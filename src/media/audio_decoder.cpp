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

#include <string>

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
            // Only meaningful once the total size is known. While the
            // upload is still in flight, FAIL the seek instead of waiting:
            // dr_wav only end-seeks to clamp a lying data-chunk size
            // against the real file size and skips the clamp gracefully
            // when the seek fails — whereas blocking here deadlocks the
            // live path (server waits for the tail, client waits for a
            // segment).
            if (!cursor.stream->is_complete()) {
                return MA_ERROR;
            }
            cursor.position = static_cast<uint64_t>(
                static_cast<int64_t>(cursor.stream->size()) + offset);
            return MA_SUCCESS;
        default:
            return MA_ERROR;
    }
}

}  // namespace

struct AudioDecoder::Impl {
    ma_decoder decoder{};
    bool initialized = false;
    StreamCursor cursor;

    ~Impl() {
        if (initialized) {
            ma_decoder_uninit(&decoder);
        }
    }
};

AudioDecoder::AudioDecoder(ByteStream& stream, bool header_declares_duration)
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

    impl_->cursor = StreamCursor{&stream, 0};
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, kModelSampleRate);
    result = ma_decoder_init(stream_read, stream_seek, &impl_->cursor, &config, &impl_->decoder);
    if (result != MA_SUCCESS) {
        throw DecodeError(std::string("cannot decode media: ") + ma_result_description(result));
    }
    impl_->initialized = true;

    info_.sample_rate_hz = native_rate;
    info_.channels = native_channels;

    // Asking for the length of an mp3 means scanning every frame, which
    // would block the live path until the upload ends. Only ask when the
    // header declares it (wav) or the bytes are already all here.
    if (header_declares_duration || stream.is_complete()) {
        ma_uint64 total_frames = 0;
        if (ma_decoder_get_length_in_pcm_frames(&impl_->decoder, &total_frames) == MA_SUCCESS) {
            info_.duration_ms = total_frames * 1000ULL / kModelSampleRate;
        }
    }
}

AudioDecoder::~AudioDecoder() = default;

size_t AudioDecoder::read(float* out, size_t max_samples) {
    ma_uint64 frames_read = 0;
    ma_result result = ma_decoder_read_pcm_frames(&impl_->decoder, out, max_samples, &frames_read);
    if (result != MA_SUCCESS && result != MA_AT_END) {
        throw DecodeError(std::string("decode failed mid-stream: ") +
                          ma_result_description(result));
    }
    return static_cast<size_t>(frames_read);
}

}  // namespace asr::media
