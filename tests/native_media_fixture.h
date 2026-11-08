#pragma once
#include "fixture.h"
#include <array>
#include <algorithm>
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
inline std::string trailing_moov_fixture(const std::string& source) {
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
    require(avformat_alloc_output_context2(&state.mux, nullptr, "mp4", nullptr) >= 0, "fixture MP4 allocated");
    for (unsigned i = 0; i < state.demux->nb_streams; ++i) {
        auto* stream = avformat_new_stream(state.mux, nullptr);
        require(stream != nullptr, "fixture output track allocated");
        require(avcodec_parameters_copy(stream->codecpar, state.demux->streams[i]->codecpar) >= 0, "fixture codec copied");
        stream->codecpar->codec_tag = 0; stream->time_base = state.demux->streams[i]->time_base;
    }
    require(avio_open_dyn_buf(&state.mux->pb) >= 0, "fixture seekable output opens");
    require(avformat_write_header(state.mux, nullptr) >= 0, "fixture MP4 header written");
    while (av_read_frame(state.demux, state.packet) >= 0) {
        const auto i = state.packet->stream_index;
        av_packet_rescale_ts(state.packet, state.demux->streams[i]->time_base, state.mux->streams[i]->time_base);
        state.packet->pos = -1;
        require(av_interleaved_write_frame(state.mux, state.packet) >= 0, "fixture MP4 packet written");
        av_packet_unref(state.packet);
    }
    require(av_write_trailer(state.mux) >= 0, "fixture trailing moov written");
    const int size = avio_close_dyn_buf(state.mux->pb, &state.output); state.mux->pb = nullptr;
    require(size > 0, "fixture MP4 has bytes");
    return std::string(reinterpret_cast<char*>(state.output), static_cast<size_t>(size));
}
