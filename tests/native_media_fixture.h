#pragma once
#include "fixture.h"
#include <array>
#include <algorithm>
#include <bit>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

inline std::string pipe_fixture(const std::string& args) {
    const std::string command = fixture_ffmpeg() + " -v error -threads 1 " + args;
    FILE* pipe = ::popen(command.c_str(), "r");
    require(pipe != nullptr, "fixture generator started");
    std::string data;
    std::array<char, 4096> buffer;
    while (size_t n = ::fread(buffer.data(), 1, buffer.size(), pipe)) data.append(buffer.data(), n);
    require(::pclose(pipe) == 0 && !data.empty(), "in-memory fixture generated");
    return data;
}

// Remux authored Matroska into seekable MP4 with a trailing moov, entirely
// in RAM. avio_open_dyn_buf implements the seeks required by the MP4 muxer.
inline std::string remux_fixture(const std::string& source, bool unknown_last_video = false) {
    struct State {
        const std::string& source; size_t offset = 0;
        AVIOContext* input = nullptr;
        AVFormatContext* demux = nullptr;
        AVFormatContext* mux = nullptr;
        AVPacket* packet = av_packet_alloc();
        uint8_t* output = nullptr;
        ~State() {
            av_packet_free(&packet); avformat_close_input(&demux);
            if (input) av_freep(&input->buffer);
            avio_context_free(&input);
            if (mux && mux->pb) avio_close_dyn_buf(mux->pb, &output);
            av_free(output); avformat_free_context(mux);
        }
    } state{source};
    auto* buffer = static_cast<uint8_t*>(av_malloc(4096));
    require(buffer && state.packet, "fixture buffers allocated");
    state.input = avio_alloc_context(buffer, 4096, 0, &state,
        [](void* opaque, uint8_t* dst, int size) {
            auto& s = *static_cast<State*>(opaque);
            const auto n = std::min(static_cast<size_t>(size), s.source.size() - s.offset);
            if (!n) return AVERROR_EOF;
            std::memcpy(dst, s.source.data() + s.offset, n); s.offset += n; return static_cast<int>(n);
        }, nullptr, nullptr);
    require(state.input != nullptr, "fixture AVIO allocated");
    state.demux = avformat_alloc_context();
    require(state.demux != nullptr, "fixture demux allocated");
    state.demux->pb = state.input; state.demux->flags |= AVFMT_FLAG_CUSTOM_IO;
    require(avformat_open_input(&state.demux, nullptr, nullptr, nullptr) == 0, "fixture input opens");
    require(avformat_find_stream_info(state.demux, nullptr) >= 0, "fixture tracks found");
    require(avformat_alloc_output_context2(&state.mux, nullptr, unknown_last_video ? "matroska" : "mp4", nullptr) >= 0, "fixture MP4 allocated");
    for (unsigned i = 0; i < state.demux->nb_streams; ++i) {
        auto* stream = avformat_new_stream(state.mux, nullptr);
        require(stream != nullptr, "fixture output track allocated");
        require(avcodec_parameters_copy(stream->codecpar, state.demux->streams[i]->codecpar) >= 0, "fixture codec copied");
        stream->codecpar->codec_tag = 0; stream->time_base = state.demux->streams[i]->time_base;
        if (unknown_last_video) {
            stream->avg_frame_rate = state.demux->streams[i]->avg_frame_rate;
            stream->r_frame_rate = state.demux->streams[i]->r_frame_rate;
            if (i + 1 == state.demux->nb_streams) {
                stream->avg_frame_rate = stream->r_frame_rate = AVRational{0, 1};
#if LIBAVCODEC_VERSION_MAJOR >= 61
                stream->codecpar->framerate = AVRational{0, 1};
#endif
            }
        }

    }
    if (unknown_last_video) state.mux->avoid_negative_ts = AVFMT_AVOID_NEG_TS_DISABLED;
    require(avio_open_dyn_buf(&state.mux->pb) >= 0, "fixture seekable output opens");
    require(avformat_write_header(state.mux, nullptr) >= 0, "fixture MP4 header written");
    while (av_read_frame(state.demux, state.packet) >= 0) {
        const auto i = state.packet->stream_index;
        av_packet_rescale_ts(state.packet, state.demux->streams[i]->time_base, state.mux->streams[i]->time_base);
        state.packet->pos = -1;
        if (unknown_last_video && static_cast<unsigned>(i + 1) == state.demux->nb_streams)
            state.packet->duration = 0;
        require(av_interleaved_write_frame(state.mux, state.packet) >= 0, "fixture MP4 packet written");
        av_packet_unref(state.packet);
    }
    require(av_write_trailer(state.mux) >= 0, "fixture trailing moov written");
    const int size = avio_close_dyn_buf(state.mux->pb, &state.output); state.mux->pb = nullptr;
    require(size > 0, "fixture MP4 has bytes");
    return std::string(reinterpret_cast<char*>(state.output), static_cast<size_t>(size));
}

inline std::string trailing_moov_fixture(const std::string& source) { return remux_fixture(source); }

// A third AV track contributes one PNG packet with no duration. The main
// video ends at relative 3500ms, while audio ends at 2000ms. Neither the
// uncertain packet nor the duration tag may erase that confirmed endpoint.
inline std::string unknown_duration_fixture(int origin, bool absolute_metadata) {
    const auto source = pipe_fixture(
        " -f lavfi -i testsrc2=duration=3:size=64x48:rate=4"
        " -f lavfi -i sine=frequency=440:sample_rate=16000:duration=2"
        " -f lavfi -i color=duration=0.04:size=8x8:rate=25"
        " -map 0:v -map 1:a -map 2:v -filter:v:0 setpts=PTS+0.5/TB"
        " -c:v:0 mpeg4 -c:v:1 png -threads:v 1 -c:a pcm_s16le"
        " -output_ts_offset " + std::to_string(origin) +
        " -avoid_negative_ts disabled -f matroska pipe:1");
    auto bytes = remux_fixture(source, true);
    // The seekable Matroska muxer writes Duration as an 8-byte EBML float,
    // with its default 1ms timecode scale. Author both metadata conventions.
    const size_t at = bytes.find("\x44\x89\x88");
    require(at != std::string::npos && at < 4096 && at + 11 <= bytes.size(), "fixture Duration element");
    const double duration = 3500.0 + (absolute_metadata ? origin * 1000.0 : 0);
    const uint64_t bits = std::bit_cast<uint64_t>(duration);
    for (int i = 0; i < 8; ++i) bytes[at + 3 + i] = static_cast<char>((bits >> (56 - i * 8)) & 255);
    return bytes;
}

struct FixtureTiming {
    int64_t origin_us = AV_NOPTS_VALUE;
    int64_t declared_us = AV_NOPTS_VALUE;
    int64_t confirmed_us = 0;
    unsigned untimed_packets = 0;
};
inline FixtureTiming fixture_timing(const std::string& bytes) {
    struct Input {
        const std::string& bytes; size_t offset = 0;
        AVIOContext* io = nullptr; AVFormatContext* format = nullptr;
        AVPacket* packet = av_packet_alloc();
        ~Input() {
            av_packet_free(&packet); avformat_close_input(&format);
            if (io) av_freep(&io->buffer);
            avio_context_free(&io);
        }
    } input{bytes};
    auto* buffer = static_cast<uint8_t*>(av_malloc(4096));
    require(buffer && input.packet, "timing fixture allocation");
    input.io = avio_alloc_context(buffer, 4096, 0, &input,
        [](void* opaque, uint8_t* dst, int size) {
            auto& in = *static_cast<Input*>(opaque);
            const size_t count = std::min(static_cast<size_t>(size), in.bytes.size() - in.offset);
            if (!count) return AVERROR_EOF;
            std::memcpy(dst, in.bytes.data() + in.offset, count); in.offset += count;
            return static_cast<int>(count);
        }, nullptr, nullptr);
    if (!input.io) av_free(buffer);
    require(input.io != nullptr, "timing fixture AVIO");
    input.format = avformat_alloc_context();
    require(input.format != nullptr, "timing fixture format");
    input.format->pb = input.io; input.format->flags |= AVFMT_FLAG_CUSTOM_IO;
    require(avformat_open_input(&input.format, nullptr, nullptr, nullptr) == 0 &&
            avformat_find_stream_info(input.format, nullptr) >= 0, "timing fixture probe");
    FixtureTiming result{input.format->start_time, input.format->duration};
    while (av_read_frame(input.format, input.packet) >= 0) {
        const auto* p = input.packet;
        if (p->pts == AV_NOPTS_VALUE || p->duration <= 0) ++result.untimed_packets;
        else result.confirmed_us = std::max(result.confirmed_us,
            av_rescale_q(p->pts + p->duration, input.format->streams[p->stream_index]->time_base,
                         AV_TIME_BASE_Q) - result.origin_us);
        av_packet_unref(input.packet);
    }
    return result;
}
inline void verify_unknown_duration_fixture(const std::string& bytes, int origin, bool absolute_metadata) {
    const auto timing = fixture_timing(bytes);
    require(timing.origin_us == origin * int64_t{1000000} && timing.untimed_packets == 1 &&
            timing.confirmed_us == 3500000, "fixture retains origin, exactly one untimed packet and 3.5s endpoint");
    require(timing.declared_us == 3500000 + (absolute_metadata ? origin * int64_t{1000000} : 0),
            "fixture retains selected absolute or relative duration metadata");
}
