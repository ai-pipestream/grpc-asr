// Native, seekable memory demux for macOS. No files, subprocesses or URLs.
#include "media/video_demux.h"
#include "media/audio_decoder.h"
#include "media/pcm_scrub.h"
#include "media/native_memory_seek.h"
#include "cancellation.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <span>
#include <vector>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

namespace asr::media {
namespace {
constexpr int64_t kMaxFramePixels = 16 * 1024 * 1024;
void check(int result, const char* message) {
    if (result < 0) throw DecodeError(message);
}
struct Input {
    std::span<const uint8_t> bytes;
    size_t position = 0;
    std::stop_token stop;
    std::chrono::milliseconds timeout;
    std::chrono::steady_clock::time_point deadline;
    AVIOContext* io = nullptr;
    AVFormatContext* format = nullptr;
    ~Input() {
        avformat_close_input(&format);
        if (io) av_freep(&io->buffer);
        avio_context_free(&io);
    }
    static int read(void* opaque, uint8_t* out, int size) {
        auto& s = *static_cast<Input*>(opaque);
        if (s.interrupted()) return AVERROR_EXIT;
        const auto count = std::min(s.bytes.size() - s.position, static_cast<size_t>(size));
        if (!count) return AVERROR_EOF;
        std::memcpy(out, s.bytes.data() + s.position, count);
        s.position += count;
        return static_cast<int>(count);
    }
    static int64_t seek(void* opaque, int64_t offset, int whence) {
        auto& s = *static_cast<Input*>(opaque);
        if (s.interrupted()) return AVERROR_EXIT;
        return native_memory_seek(s.bytes.size(), s.position, offset, whence);
    }
    bool interrupted() const { return stop.stop_requested() || std::chrono::steady_clock::now() >= deadline; }
    void begin_operation() { deadline = std::chrono::steady_clock::now() + timeout; cancelled(); }
    void cancelled() const {
        if (stop.stop_requested()) throw Cancelled("native media decode cancelled");
        if (std::chrono::steady_clock::now() >= deadline) throw ToolError("native media operation timed out");
    }
    void open(std::span<const uint8_t> data, std::stop_token token, std::chrono::milliseconds limit) {
        bytes = data; stop = token; timeout = limit; begin_operation();
        auto* buffer = static_cast<uint8_t*>(av_malloc(32768));
        if (!buffer) throw std::bad_alloc();
        io = avio_alloc_context(buffer, 32768, 0, this, read, nullptr, seek);
        if (!io) { av_free(buffer); throw std::bad_alloc(); }
        format = avformat_alloc_context();
        if (!format) throw std::bad_alloc();
        format->pb = io; format->flags |= AVFMT_FLAG_CUSTOM_IO;
        format->max_streams = 64;
        format->interrupt_callback = {[](void* p) { return static_cast<Input*>(p)->interrupted() ? 1 : 0; }, this};
        format->io_open = [](AVFormatContext*, AVIOContext**, const char*, int, AVDictionary**) { return AVERROR(EPERM); };
        AVDictionary* options = nullptr;
        av_dict_set(&options, "format_whitelist", "wav,mp3,flac,ogg,mov,matroska,webm,mpegts,aac,nut", 0);
        const int opened = avformat_open_input(&format, nullptr, nullptr, &options);
        av_dict_free(&options);
        cancelled(); check(opened, "native media open failed");
        // Probing may itself decode frames, before Track opens its decoder.
        std::vector<AVDictionary*> probe_options(format->nb_streams, nullptr);
        for (unsigned i = 0; i < format->nb_streams; ++i) {
            const auto* p = format->streams[i]->codecpar;
            if (p->codec_type == AVMEDIA_TYPE_VIDEO && (p->width > 8192 || p->height > 8192 ||
                static_cast<int64_t>(p->width) * p->height > kMaxFramePixels)) {
                for (auto& option : probe_options) av_dict_free(&option);
                throw DecodeError("native video dimensions exceed limit before probing");
            }
            av_dict_set_int(&probe_options[i], "max_pixels", kMaxFramePixels, 0);
            av_dict_set(&probe_options[i], "threads", "2", 0);
        }
        const int probed = avformat_find_stream_info(format, probe_options.data());
        for (auto& option : probe_options) av_dict_free(&option);
        cancelled(); check(probed, "native media probe failed");
    }
};
struct Track {
    Input input;
    AVCodecContext* codec = nullptr;
    AVFrame* frame = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();
    int index = -1;
    bool draining = false;
    ~Track() { avcodec_free_context(&codec); av_frame_free(&frame); av_packet_free(&packet); }
    void open(std::span<const uint8_t> data, std::stop_token stop, AVMediaType type, std::chrono::milliseconds timeout) {
        if (!frame || !packet) throw std::bad_alloc();
        input.open(data, stop, timeout);
        // Match probe and the established ffmpeg -map a:0 / v:0 contract.
        for (unsigned i = 0; i < input.format->nb_streams; ++i) {
            if (input.format->streams[i]->codecpar->codec_type == type) { index = static_cast<int>(i); break; }
        }
        check(index, "requested media track is missing");
        auto* params = input.format->streams[index]->codecpar;
        if (type == AVMEDIA_TYPE_VIDEO && (params->width < 0 || params->height < 0 ||
            params->width > 8192 || params->height > 8192 ||
            static_cast<int64_t>(params->width) * params->height > kMaxFramePixels))
            throw DecodeError("native video dimensions exceed limit");
        auto* decoder = avcodec_find_decoder(params->codec_id);
        if (!decoder) throw DecodeError("native codec unavailable");
        codec = avcodec_alloc_context3(decoder);
        if (!codec) throw std::bad_alloc();
        check(avcodec_parameters_to_context(codec, params), "native codec parameters failed");
        codec->thread_count = 2;
        codec->max_pixels = kMaxFramePixels;
        check(avcodec_open2(codec, decoder, nullptr), "native codec open failed");
        input.cancelled();
    }
    bool next() {
        input.begin_operation();
        while (true) {
            input.cancelled();
            const int received = avcodec_receive_frame(codec, frame);
            input.cancelled();
            if (received == 0) return true;
            if (received == AVERROR_EOF) return false;
            if (received != AVERROR(EAGAIN)) check(received, "native frame decode failed");
            if (draining) throw DecodeError("native decoder stalled during drain");
            int read_result;
            do {
                av_packet_unref(packet);
                read_result = av_read_frame(input.format, packet);
                input.cancelled();
            } while (read_result >= 0 && packet->stream_index != index);
            if (read_result == AVERROR_EOF) {
                draining = true;
                check(avcodec_send_packet(codec, nullptr), "native decoder drain failed");
            } else {
                check(read_result, "native packet read failed");
                check(avcodec_send_packet(codec, packet), "native packet decode failed");
            }
            input.cancelled();
        }
    }
};
// All tracks share the container origin. Rebasing each track independently
// would erase an intentional audio/video offset. When the demuxer cannot
// supply a container start, use the earliest known A/V stream start; with no
// timestamp metadata at all, retain the conventional zero origin.
int64_t timestamp_us(int64_t timestamp, AVRational time_base) {
    const int64_t value = av_rescale_q(timestamp, time_base, AV_TIME_BASE_Q);
    if (value == INT64_MIN || value == INT64_MAX)
        throw DecodeError("native media timestamp exceeds range");
    return value;
}
int64_t media_origin_us(const AVFormatContext* format) {
    if (format->start_time != AV_NOPTS_VALUE) return format->start_time;
    int64_t origin = AV_NOPTS_VALUE;
    for (unsigned i = 0; i < format->nb_streams; ++i) {
        const auto* stream = format->streams[i];
        if ((stream->codecpar->codec_type != AVMEDIA_TYPE_AUDIO &&
             stream->codecpar->codec_type != AVMEDIA_TYPE_VIDEO) || stream->start_time == AV_NOPTS_VALUE) continue;
        const int64_t start = timestamp_us(stream->start_time, stream->time_base);
        if (origin == AV_NOPTS_VALUE || start < origin) origin = start;
    }
    return origin == AV_NOPTS_VALUE ? 0 : origin;
}
int64_t relative_time_us(int64_t timestamp, AVRational time_base, int64_t origin) {
    const int64_t time = timestamp_us(timestamp, time_base);
    if ((origin > 0 && time < INT64_MIN + origin) ||
        (origin < 0 && time > INT64_MAX + origin))
        throw DecodeError("native media relative timestamp exceeds range");
    return time - origin;
}
// Container duration tags are not uniformly relative to start_time (notably
// authored Matroska may store the absolute endpoint). For a nonzero origin,
// measure the packet endpoint on the common timeline rather than blindly
// subtracting start_time from a duration that may already be relative.
uint64_t relative_duration_ms(Input& input, int64_t origin) {
    auto free_packet = [](AVPacket* packet) { av_packet_free(&packet); };
    std::unique_ptr<AVPacket, decltype(free_packet)> packet(av_packet_alloc(), free_packet);
    if (!packet) throw std::bad_alloc();
    input.begin_operation(); // one deadline for the entire in-memory scan
    int64_t end_us = 0;
    bool timed = false, unknown = false;
    int result;
    while ((result = av_read_frame(input.format, packet.get())) >= 0) {
        input.cancelled();
        const auto* stream = input.format->streams[packet->stream_index];
        if (stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO || stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            if (packet->pts == AV_NOPTS_VALUE || packet->duration <= 0) {
                unknown = true;
            } else {
                const int64_t start = relative_time_us(packet->pts, stream->time_base, origin);
                const int64_t duration = timestamp_us(packet->duration, stream->time_base);
                if (start > INT64_MAX - duration) throw DecodeError("native media duration exceeds range");
                end_us = std::max(end_us, start + duration);
                timed = true;
            }
        }
        av_packet_unref(packet.get());
    }
    input.cancelled();
    if (result != AVERROR_EOF) check(result, "native duration scan failed");
    // No fabricated endpoint when a packet's timing cannot be established.
    return timed && !unknown ? static_cast<uint64_t>(end_us / 1000) : 0;
}
struct Audio {
    Track track;
    SwrContext* swr = nullptr;
    std::vector<float> pending;
    size_t offset = 0;
    bool done = false;
    bool first_frame = true;
    int64_t origin_us = 0;
    uint64_t leading_silence = 0;
    uint64_t trim_samples = 0;
    ~Audio() { swr_free(&swr); }
    void open(std::span<const uint8_t> data, std::stop_token stop, std::chrono::milliseconds timeout) {
        track.open(data, stop, AVMEDIA_TYPE_AUDIO, timeout);
        origin_us = media_origin_us(track.input.format);
        AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
        auto* c = track.codec;
        if (c->sample_rate <= 0) throw DecodeError("invalid native sample rate");
        check(swr_alloc_set_opts2(&swr, &mono, AV_SAMPLE_FMT_FLT, 16000,
            &c->ch_layout, c->sample_fmt, c->sample_rate, 0, nullptr), "native resampler allocation failed");
        check(swr_init(swr), "native resampler initialization failed");
    }
    bool fill() {
        if (done) return false;
        const bool frame = track.next();
        if (frame && (track.frame->sample_rate != track.codec->sample_rate ||
            track.frame->format != track.codec->sample_fmt ||
            av_channel_layout_compare(&track.frame->ch_layout, &track.codec->ch_layout) != 0))
            throw DecodeError("native audio format changed during decode");
        if (frame && first_frame) {
            first_frame = false;
            const auto* stream = track.input.format->streams[track.index];
            const int64_t pts = track.frame->best_effort_timestamp != AV_NOPTS_VALUE
                ? track.frame->best_effort_timestamp : stream->start_time;
            if (pts != AV_NOPTS_VALUE) {
                const int64_t start = av_rescale_q(relative_time_us(pts, stream->time_base, origin_us),
                                                  AV_TIME_BASE_Q, AVRational{1, 16000});
                if (start >= 0) leading_silence = static_cast<uint64_t>(start);
                else trim_samples = static_cast<uint64_t>(-(start + 1)) + 1;
            }
        }
        const int samples = frame ? track.frame->nb_samples : 0;

        const int capacity = swr_get_out_samples(swr, samples);
        if (capacity < 0 || capacity > 16 * 1024 * 1024) throw DecodeError("native PCM frame exceeds bound");
        pending.resize(static_cast<size_t>(capacity)); offset = 0;
        auto* out = reinterpret_cast<uint8_t*>(pending.data());
        const uint8_t** source = frame ? const_cast<const uint8_t**>(track.frame->extended_data) : nullptr;
        const int converted = swr_convert(swr, &out, capacity, source, samples);
        track.input.cancelled();
        check(converted, "native audio resample failed");
        pending.resize(static_cast<size_t>(converted));
        scrub_pcm(pending);
        // Trim preroll before the shared origin; retain resampler continuity.
        offset = static_cast<size_t>(std::min<uint64_t>(trim_samples, pending.size()));
        trim_samples -= offset;
        if (!frame && converted == 0) done = true;
        return !done;
    }
};
}  // namespace

struct VideoDemux::Impl {
    std::vector<uint8_t> bytes;
    std::unique_ptr<Audio> audio;
    std::chrono::milliseconds timeout;
};
VideoDemux::VideoDemux(const uint8_t* data, size_t size, std::string, std::string,
                     std::chrono::milliseconds timeout) : impl_(std::make_unique<Impl>()) {
    if (!data || !size || size > static_cast<size_t>(INT64_MAX)) throw DecodeError("invalid media bytes");
    impl_->bytes.assign(data, data + size);
    if (timeout.count() <= 0) throw ToolError("native media timeout must be positive");
    impl_->timeout = timeout;
}
VideoDemux::~VideoDemux() = default;
ProbeInfo VideoDemux::probe(std::stop_token stop) {
    Input input; input.open(impl_->bytes, stop, impl_->timeout);
    ProbeInfo info;
    if (input.format->duration > 0) info.duration_ms = static_cast<uint64_t>(input.format->duration / 1000);
    for (unsigned i = 0; i < input.format->nb_streams; ++i) {
        const auto* p = input.format->streams[i]->codecpar;
        if (p->codec_type == AVMEDIA_TYPE_AUDIO && !info.has_audio) {
            info.has_audio = true; info.audio_codec = avcodec_get_name(p->codec_id);
            info.sample_rate_hz = static_cast<uint32_t>(std::max(0, p->sample_rate));
            info.channels = static_cast<uint32_t>(std::max(0, p->ch_layout.nb_channels));
        } else if (p->codec_type == AVMEDIA_TYPE_VIDEO && !info.has_video) {
            info.has_video = true; info.video_codec = avcodec_get_name(p->codec_id);
        }
    }
    const int64_t origin = media_origin_us(input.format);
    if (origin != 0 && (info.has_audio || info.has_video))
        info.duration_ms = relative_duration_ms(input, origin);
    return info;
}
void VideoDemux::open_audio(std::stop_token stop) {
    auto audio = std::make_unique<Audio>(); audio->open(impl_->bytes, stop, impl_->timeout); impl_->audio = std::move(audio);
}
size_t VideoDemux::read_audio(float* out, size_t size) {
    if (!impl_->audio) throw ToolError("native audio is not open");
    auto& a = *impl_->audio;
    a.track.input.begin_operation();
    while (!a.leading_silence && a.offset == a.pending.size()) if (!a.fill()) return 0;
    // Stream the initial A/V gap through the existing read boundary. Never
    // allocate proportional to an untrusted timestamp; stop/deadline and the
    // transcriber's duration cap also apply while consuming these zeros.
    if (a.leading_silence) {
        const size_t count = static_cast<size_t>(std::min<uint64_t>(size, a.leading_silence));
        std::fill_n(out, count, 0.0f); a.leading_silence -= count; return count;
    }

    const size_t count = std::min(size, a.pending.size() - a.offset);
    std::copy_n(a.pending.data() + a.offset, count, out); a.offset += count; return count;
}
void VideoDemux::close_audio() { impl_->audio.reset(); }
void VideoDemux::cancel_audio() { impl_->audio.reset(); }

void png_dimensions(const std::string& png, uint32_t* width, uint32_t* height) {
    if (png.size() < 24 || png.compare(0, 8, "\x89PNG\r\n\x1a\n", 8) != 0 || png.compare(12, 4, "IHDR") != 0)
        throw DecodeError("invalid PNG header");
    const auto read = [&](size_t offset) {
        uint32_t n = 0; for (size_t i = 0; i < 4; ++i) n = (n << 8) | static_cast<uint8_t>(png[offset + i]); return n;
    };
    *width = read(16); *height = read(20);
}

void VideoDemux::extract_keyframes(uint32_t interval,
    const std::function<bool(uint64_t, uint32_t, uint32_t, std::string)>& sink, std::stop_token stop) {
    if (!interval) throw DecodeError("keyframe interval must be positive");
    Track track; track.open(impl_->bytes, stop, AVMEDIA_TYPE_VIDEO, impl_->timeout);
    const int64_t origin_us = media_origin_us(track.input.format);
    uint64_t next_ms = 0;
    while (track.next()) {
        auto* frame = track.frame;
        if (frame->best_effort_timestamp == AV_NOPTS_VALUE) continue;
        const int64_t relative_us = relative_time_us(frame->best_effort_timestamp,
            track.input.format->streams[track.index]->time_base, origin_us);
        const int64_t ms = av_rescale_q(relative_us, AV_TIME_BASE_Q, AVRational{1, 1000});
        if (ms < 0 || static_cast<uint64_t>(ms) < next_ms) continue;
        if (frame->width <= 0 || frame->height <= 0 || frame->width > 8192 || frame->height > 8192 ||
            static_cast<int64_t>(frame->width) * frame->height > kMaxFramePixels)
            throw DecodeError("native keyframe dimensions exceed limit");
        struct Image {
            AVCodecContext* codec = nullptr; AVFrame* rgb = av_frame_alloc();
            AVPacket* packet = av_packet_alloc(); SwsContext* scale = nullptr;
            ~Image() { avcodec_free_context(&codec); av_frame_free(&rgb); av_packet_free(&packet); sws_freeContext(scale); }
        } image;
        image.codec = avcodec_alloc_context3(avcodec_find_encoder(AV_CODEC_ID_PNG));
        if (!image.codec || !image.rgb || !image.packet) throw std::bad_alloc();
        image.codec->width = frame->width; image.codec->height = frame->height;
        image.codec->pix_fmt = AV_PIX_FMT_RGB24; image.codec->time_base = {1, 1}; image.codec->thread_count = 1;
        check(avcodec_open2(image.codec, nullptr, nullptr), "PNG encoder unavailable");
        image.rgb->width = frame->width; image.rgb->height = frame->height; image.rgb->format = AV_PIX_FMT_RGB24;
        check(av_frame_get_buffer(image.rgb, 32), "PNG frame allocation failed");
        image.scale = sws_getContext(frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
            frame->width, frame->height, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!image.scale) throw DecodeError("native keyframe conversion failed");
        check(sws_scale(image.scale, frame->data, frame->linesize, 0, frame->height,
            image.rgb->data, image.rgb->linesize), "native keyframe conversion failed");
        check(avcodec_send_frame(image.codec, image.rgb), "PNG encoding failed");
        check(avcodec_receive_packet(image.codec, image.packet), "PNG output failed");
        track.input.cancelled();
        if (!sink(static_cast<uint64_t>(ms), static_cast<uint32_t>(frame->width), static_cast<uint32_t>(frame->height),
            std::string(reinterpret_cast<char*>(image.packet->data), static_cast<size_t>(image.packet->size)))) return;
        next_ms = static_cast<uint64_t>(ms) + static_cast<uint64_t>(interval) * 1000;
    }
}
}  // namespace asr::media
