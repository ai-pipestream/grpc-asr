// miniaudio is compiled here, decode-only: no devices, no threads, no
// generation. The same vendored header whisper.cpp uses for its examples.
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_DEVICE_IO
#define MA_NO_THREADING
#define MA_NO_ENGINE
#define MA_NO_NODE_GRAPH
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#include "miniaudio.h"

#include "media/audio_decoder.h"

#include <span>
#include <string>

#include "media/pcm_scrub.h"

namespace asr::media {

namespace {

// Cursor-carrying adapter between miniaudio's pull callbacks and the
// blocking ByteStream. Each decoder owns its own cursor.
struct StreamCursor {
    ByteStream* stream = nullptr;
    uint64_t position = 0;
};

ma_result stream_read(ma_decoder* decoder, void* out, size_t bytes_to_read, size_t* bytes_read) {
    StreamCursor& cursor = *static_cast<StreamCursor*>(decoder->pUserData);
    size_t total = 0;
    uint8_t* dest = static_cast<uint8_t*>(out);
    // read_at returns as soon as one byte is available; loop so miniaudio
    // sees short reads only at true end of stream.
    while (total < bytes_to_read) {
        size_t got = cursor.stream->read_at(cursor.position, dest + total, bytes_to_read - total);
        if (got == 0) {
            break;
        }
        cursor.position += got;
        total += got;
    }
    *bytes_read = total;
    if (total == 0) {
        return MA_AT_END;
    }
    return MA_SUCCESS;
}

ma_result stream_seek(ma_decoder* decoder, ma_int64 offset, ma_seek_origin origin) {
    StreamCursor& cursor = *static_cast<StreamCursor*>(decoder->pUserData);
    switch (origin) {
        case ma_seek_origin_start:
            cursor.position = static_cast<uint64_t>(offset);
            return MA_SUCCESS;
        case ma_seek_origin_current:
            cursor.position = static_cast<uint64_t>(
                static_cast<int64_t>(cursor.position) + offset);
            return MA_SUCCESS;
        case ma_seek_origin_end:
            // Always refused. The decoders seek to the end only to ask
            // where that is, and ma_decoder has no tell callback to answer
            // with, so a seek that succeeds buys nothing: dr_wav and
            // dr_mp3 both carry on without the file size. It does cost
            // something: dr_wav counts a data chunk of "unknown" size
            // (0xFFFFFFFF, what a streaming writer leaves) by reading
            // from where the cursor is, and after a successful end seek
            // that is the end, so a fully uploaded streamed wav decoded
            // to no audio at all and still ended OK. Refusing here is
            // also what an upload still in flight always did (waiting
            // for its tail would deadlock the live path), so both now
            // decode the same way.
            return MA_ERROR;
        default:
            return MA_ERROR;
    }
}

}  // namespace

struct AudioDecoder::Impl {
    ma_decoder decoder{};
    bool initialized = false;
    uint32_t output_channels = 1;
    StreamCursor cursor;

    ~Impl() {
        if (initialized) {
            ma_decoder_uninit(&decoder);
        }
    }
};

AudioDecoder::AudioDecoder(ByteStream& stream, bool header_declares_duration,
                           bool preserve_channels)
    : impl_(std::make_unique<Impl>()) {
    // First open with no conversion to learn the source format, then
    // reopen converting to the model contract. Two inits are cheaper than
    // poking miniaudio internals for the pre-conversion format.
    StreamCursor probe_cursor{&stream, 0};
    ma_decoder probe{};
    ma_decoder_config probe_config = ma_decoder_config_init_default();
    probe.pUserData = &probe_cursor;  // set before init reads the header
    ma_result result =
        ma_decoder_init(stream_read, stream_seek, &probe_cursor, &probe_config, &probe);
    if (result != MA_SUCCESS) {
        throw DecodeError(std::string("cannot decode media: ") + ma_result_description(result));
    }
    ma_format native_format = ma_format_unknown;
    ma_uint32 native_channels = 0;
    ma_uint32 native_rate = 0;
    ma_data_source_get_data_format(&probe, &native_format, &native_channels, &native_rate, nullptr,
                                   0);
    ma_decoder_uninit(&probe);

    const ma_uint32 out_channels =
        preserve_channels && native_channels > 0 ? native_channels : 1U;
    impl_->output_channels = out_channels;
    impl_->cursor = StreamCursor{&stream, 0};
    ma_decoder_config config =
        ma_decoder_config_init(ma_format_f32, out_channels, kModelSampleRate);
    result = ma_decoder_init(stream_read, stream_seek, &impl_->cursor, &config, &impl_->decoder);
    if (result != MA_SUCCESS) {
        throw DecodeError(std::string("cannot decode media: ") + ma_result_description(result));
    }
    impl_->initialized = true;

    info_.sample_rate_hz = native_rate;
    info_.channels = native_channels;

    // Asking for the length of an mp3 means scanning every frame, which
    // would block the live path until the upload ends. Only ask when the
    // header declares it (wav, flac) or the bytes are already all here.
    if (header_declares_duration || stream.is_complete()) {
        ma_uint64 total_frames = 0;
        if (ma_decoder_get_length_in_pcm_frames(&impl_->decoder, &total_frames) == MA_SUCCESS) {
            info_.duration_ms = total_frames * 1000ULL / kModelSampleRate;
        }
    }
    // A length read from the header is a claim the decode can be held to;
    // one found by scanning (mp3) is only a measurement of what is there.
    // A wav of "unknown" size is counted from its bytes by dr_wav, so its
    // claim is what it holds.
    if (header_declares_duration) {
        info_.declared_duration_ms = info_.duration_ms;
    }
}

AudioDecoder::~AudioDecoder() = default;

size_t AudioDecoder::read(float* out, size_t max_samples) {
    if (impl_->output_channels != 1) {
        throw DecodeError("AudioDecoder::read requires mono output; use read_frames");
    }
    return read_frames(out, max_samples);
}

size_t AudioDecoder::read_frames(float* interleaved_out, size_t max_frames) {
    ma_uint64 frames_read = 0;
    ma_result result =
        ma_decoder_read_pcm_frames(&impl_->decoder, interleaved_out, max_frames, &frames_read);
    if (result != MA_SUCCESS && result != MA_AT_END) {
        throw DecodeError(std::string("decode failed mid-stream: ") +
                          ma_result_description(result));
    }
    const size_t frames = static_cast<size_t>(frames_read);
    scrub_pcm(std::span<float>(interleaved_out, frames * impl_->output_channels));
    return frames;
}

bool AudioDecoder::has_buffered_input(size_t min_bytes) const {
    const ByteStream* stream = impl_->cursor.stream;
    return stream->is_complete() || stream->is_aborted() ||
           stream->available_from(impl_->cursor.position) >= min_bytes;
}

bool AudioDecoder::wait_for_input(std::chrono::milliseconds timeout, size_t min_bytes) {
    return impl_->cursor.stream->wait_for_available(impl_->cursor.position, min_bytes, timeout);
}

}  // namespace asr::media
