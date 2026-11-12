#include "service/asr_service.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include "cancellation.h"
#include "document/document_fold.h"
#include "engine/docling_asr_options.h"
#include "engine/remote_engine.h"
#include "engine/remote_options.h"
#include "engine/transcriber.h"
#include "media/audio_decoder.h"
#include "media/byte_stream.h"
#include "media/sniff.h"
#include "media/video_demux.h"
#include "metrics.h"
#include "whisper.h"

namespace asr {

namespace asrv1 = ai::pipestream::asr::v1;

namespace {

#ifndef GRPC_ASR_VERSION
#define GRPC_ASR_VERSION "0.0.0-dev"
#endif

// How often the RPC thread asks gRPC whether the call is still alive
// once the upload is in: the bound on how long work outlives a cancel or
// a passed deadline before its stop is requested.
constexpr std::chrono::milliseconds kCallWatchInterval{100};

// gRPC stream Write is not thread-safe; transcription workers and the
// keyframe thread share the stream through this lock. write() returning
// false means the client is gone and the producer should stop. The same
// lock feeds the optional Document fold, so segment and keyframe events
// fold in the exact order they hit the wire.
class LockedWriter {
  public:
    LockedWriter(
        grpc::ServerReaderWriter<asrv1::TranscribeResponse, asrv1::TranscribeRequest>* stream,
        doc::AsrDocumentFold* fold)
        : stream_(stream), fold_(fold) {}

    bool write(const asrv1::TranscribeResponse& response) {
        std::lock_guard<std::mutex> lock(mutex_);
        return write_locked(response);
    }

    bool folding() {
        std::lock_guard<std::mutex> lock(mutex_);
        return fold_ != nullptr;
    }

    void set_source(std::string filename, std::string mimetype, uint64_t binary_hash) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fold_ != nullptr) {
            fold_->set_source(std::move(filename), std::move(mimetype), binary_hash);
        }
    }

    bool finish(const asrv1::TranscribeResponse& trailer) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fold_ != nullptr) {
            fold_->consume(trailer);
            asrv1::TranscribeResponse document_event;
            *document_event.mutable_document() = fold_->take();
            fold_ = nullptr;
            if (!write_locked(document_event)) {
                return false;
            }
        }
        return write_locked(trailer);
    }

  private:
    bool write_locked(const asrv1::TranscribeResponse& response) {
        if (fold_ != nullptr) {
            fold_->consume(response);
        }
        if (dead_) {
            return false;
        }
        if (!stream_->Write(response)) {
            dead_ = true;
            return false;
        }
        return true;
    }

    grpc::ServerReaderWriter<asrv1::TranscribeResponse, asrv1::TranscribeRequest>* stream_;
    doc::AsrDocumentFold* fold_;
    std::mutex mutex_;
    bool dead_ = false;
};

std::string source_filename(const std::string& requested, media::MediaFamily family) {
    if (requested.empty()) {
        return "media." + std::string(media::family_name(family));
    }
    return requested.substr(requested.find_last_of('/') + 1);
}

std::optional<std::string> resolve_pipe_audio_codec(media::MediaFamily family,
                                                    media::ByteStream& upload,
                                                    const std::string& initial_prefix) {
    const size_t initial_avail = static_cast<size_t>(
        std::min<uint64_t>(upload.size(), media::kMaxPipeCodecProbeBytes));
    std::string codec_prefix =
        initial_avail > initial_prefix.size() ? upload.prefix_bytes(initial_avail) : initial_prefix;
    while (true) {
        size_t needed = 0;
        const bool complete = upload.is_complete();
        const std::string_view codec = media::sniff_pipe_audio_codec(
            family, reinterpret_cast<const uint8_t*>(codec_prefix.data()), codec_prefix.size(),
            complete, &needed);
        if (!codec.empty()) {
            return std::string(codec);
        }
        if (complete || needed <= codec_prefix.size() || needed > media::kMaxPipeCodecProbeBytes) {
            const std::string_view fallback = media::sniff_pipe_audio_codec(
                family, reinterpret_cast<const uint8_t*>(codec_prefix.data()), codec_prefix.size(),
                /*stream_complete=*/true);
            return std::string(fallback);
        }
        const size_t got = upload.wait_for_prefix(needed);
        if (upload.is_aborted()) {
            return std::nullopt;
        }
        codec_prefix = upload.prefix_bytes(got);
    }
}

void fill_segment(const engine::EngineSegment& source, asrv1::Segment* out) {
    out->set_index(source.index);
    out->set_start_ms(source.start_ms);
    out->set_end_ms(source.end_ms);
    out->set_text(source.text);
    if (source.has_avg_logprob) {
        out->set_avg_logprob(source.avg_logprob);
    }
    if (source.has_no_speech_prob) {
        out->set_no_speech_prob(source.no_speech_prob);
    }
    out->set_suppressed(source.suppressed);
    if (source.channel.has_value()) {
        out->set_channel(*source.channel);
    }
    if (!source.language.empty()) {
        out->set_language(source.language);
    }
    if (!source.speaker.empty()) {
        out->set_speaker_id(source.speaker);
    }
    out->set_speaker_turn_next(source.speaker_turn_next);
    for (const engine::EngineWord& word : source.words) {
        asrv1::Word* out_word = out->add_words();
        out_word->set_text(word.text);
        out_word->set_start_ms(word.start_ms);
        out_word->set_end_ms(word.end_ms);
        out_word->set_probability(word.probability);
    }
}

void merge_channel_results(const std::vector<engine::EngineResult>& per_channel,
                           engine::EngineResult* merged,
                           std::vector<uint32_t>* segment_count_by_channel) {
    segment_count_by_channel->clear();
    segment_count_by_channel->reserve(per_channel.size());
    std::string fallback_lang;
    for (const engine::EngineResult& ch : per_channel) {
        segment_count_by_channel->push_back(ch.final_segments);
        merged->duration_ms = std::max(merged->duration_ms, ch.duration_ms);
        merged->final_segments += ch.final_segments;
        merged->tokens += ch.tokens;
        merged->windows += ch.windows;
        merged->window_redecodes += ch.window_redecodes;
        merged->decode_wall_ns += ch.decode_wall_ns;
        merged->aborted = merged->aborted || ch.aborted;
        if (fallback_lang.empty() && !ch.language.empty()) {
            fallback_lang = ch.language;
        }
        for (const std::string& lang : ch.languages) {
            if (!lang.empty() &&
                std::find(merged->languages.begin(), merged->languages.end(), lang) ==
                    merged->languages.end()) {
                merged->languages.push_back(lang);
            }
        }
    }
    merged->language =
        merged->languages.empty() ? fallback_lang : merged->languages.front();
}

struct PreparedCall {
    asrv1::TranscribeOptions options;
    engine::DecodingConfig decoding;
    std::optional<engine::RemoteEndpointPolicy> remote_policy;
    size_t eff_min_window_seconds = 30;
    size_t chosen_window_seconds = 480;
};

grpc::Status process_stream(const Config& config_, engine::ModelPool& pool_,
                            const PreparedCall& call, media::ByteStream& upload,
                            LockedWriter& writer, std::atomic<long>& audio_ms,
                            StreamMetrics* telemetry, std::stop_token stop) {
    const auto stream_start = std::chrono::steady_clock::now();
    const asrv1::TranscribeOptions& options = call.options;
    const uint64_t max_duration_ms =
        static_cast<uint64_t>(config_.max_duration_seconds) * 1000ULL;
    const bool per_channel =
        options.channel_mode() == asrv1::CHANNEL_MODE_PER_CHANNEL;
    const std::string metric_model =
        options.has_remote() ? std::string("remote") : options.model();
    telemetry->model = metric_model;
    telemetry->chosen_window_seconds =
        options.has_remote() ? 0U : static_cast<uint32_t>(call.chosen_window_seconds);

    engine::EngineOptions base_engine_options{
        .language = options.language(),
        .translate = options.task() == asrv1::TASK_TRANSLATE,
        .word_timestamps = options.word_timestamps(),
        .diarize = options.diarize(),
        .threads = static_cast<int>(config_.threads),
        .window_seconds = call.chosen_window_seconds,
        .min_window_seconds = call.eff_min_window_seconds,
        .window_stall_ms = config_.window_stall_ms,
        .max_duration_seconds = config_.max_duration_seconds,
        .channel = std::nullopt,
        .decoding = call.decoding,
    };

    auto acquire_state = [&](std::stop_token token) {
        return pool_.acquire(options.model(), std::move(token),
                             config_.queue_timeout_seconds == 0
                                 ? std::chrono::steady_clock::time_point::max()
                                 : std::chrono::steady_clock::now() +
                                       std::chrono::seconds(config_.queue_timeout_seconds));
    };
    auto gone = [&] { return upload.is_aborted() || stop.stop_requested(); };
    const grpc::Status cancelled{grpc::StatusCode::CANCELLED,
                                 "the stream went away mid-transcription"};

    try {
        // Sniff the container and stream layout from the leading prefix;
        // 400 bytes covers three 188-byte MPEG-TS sync packets and ISO BMFF
        // top-level box headers without waiting for the whole upload.
        const size_t prefix_size = upload.wait_for_prefix(media::kSniffPrefixBytes);
        if (upload.is_aborted()) {
            telemetry->result = StreamEndResult::kCancelled;
            return {grpc::StatusCode::CANCELLED, "upload aborted"};
        }
        const std::string prefix = upload.prefix_bytes(prefix_size);
        const auto* prefix_ptr = reinterpret_cast<const uint8_t*>(prefix.data());
        media::MediaFamily family = media::sniff(prefix_ptr, prefix.size());
        bool used_extension_fallback = false;
        if (family == media::MediaFamily::kUnknown &&
            media::has_routed_media_extension(options.filename())) {
            family = media::sniff_from_filename(options.filename());
            used_extension_fallback = true;
        }
        if (family == media::MediaFamily::kUnknown && !used_extension_fallback) {
            telemetry->result = StreamEndResult::kUnknownContainer;
            return {grpc::StatusCode::UNIMPLEMENTED,
                    "unrecognized media container; supported: wav, mp3, flac, ogg, mp4/mov, "
                    "mkv/webm, avi, asf/wma/wmv, amr, mpeg-ps, mpeg-ts, aac"};
        }

        asrv1::TranscribeResponse info_response;
        asrv1::MediaInfo* info = info_response.mutable_media_info();
        info->set_container(media::to_proto_container(family));

        engine::EngineResult result;
        std::vector<uint32_t> segment_count_by_channel;
        std::atomic<uint32_t> keyframe_count{0};
        std::atomic<bool> observed_first_segment{false};
        FirstSegmentPath first_segment_path = FirstSegmentPath::kLive;

        auto segment_sink = [&](const engine::EngineSegment& segment, bool is_final) {
            if (!observed_first_segment.exchange(true, std::memory_order_relaxed)) {
                const uint64_t ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - stream_start)
                        .count());
                metrics_observe_first_segment(metric_model, first_segment_path, ns);
            }
            asrv1::TranscribeResponse response;
            fill_segment(segment, is_final ? response.mutable_final_segment()
                                           : response.mutable_partial_segment());
            return writer.write(response);
        };

        if (call.remote_policy.has_value()) {
            // Remote transcription path: emit MediaInfo from the local sniff,
            // wait for upload completion, and stream the body from ByteStream
            // via libcurl's read callback without copying it.
            first_segment_path = FirstSegmentPath::kFfmpeg;
            info->set_duration_ms(0);
            if (family != media::MediaFamily::kUnknown) {
                const auto codec = resolve_pipe_audio_codec(family, upload, prefix);
                if (!codec.has_value()) {
                    telemetry->result = StreamEndResult::kCancelled;
                    return {grpc::StatusCode::CANCELLED, "upload aborted"};
                }
                info->set_audio_codec(*codec);
            }
            if (!writer.write(info_response)) {
                telemetry->result = StreamEndResult::kCancelled;
                return cancelled;
            }
            upload.wait_complete();
            if (gone()) {
                telemetry->result = StreamEndResult::kCancelled;
                return {grpc::StatusCode::CANCELLED, "upload aborted"};
            }
            grpc::Status remote_status = engine::run_remote_transcription(
                options.remote(), *call.remote_policy, upload,
                source_filename(options.filename(), family),
                std::string(media::family_mimetype(family)), segment_sink, &result, stop);
            if (!remote_status.ok()) {
                if (remote_status.error_code() == grpc::StatusCode::CANCELLED) {
                    telemetry->result = StreamEndResult::kCancelled;
                } else if (remote_status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
                    telemetry->result = StreamEndResult::kDeadline;
                } else if (remote_status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED) {
                    telemetry->result = StreamEndResult::kByteCap;
                } else if (remote_status.error_code() == grpc::StatusCode::INVALID_ARGUMENT ||
                           remote_status.error_code() == grpc::StatusCode::PERMISSION_DENIED) {
                    telemetry->result = StreamEndResult::kInvalidMedia;
                } else {
                    telemetry->result = StreamEndResult::kInternal;
                }
                return remote_status;
            }
        } else if (!used_extension_fallback && media::decodes_in_process(family)) {
            first_segment_path = FirstSegmentPath::kLive;
            const bool wav_like = family == media::MediaFamily::kWav;
            if (!per_channel) {
                media::AudioDecoder decoder(upload, wav_like, /*preserve_channels=*/false);
                if (decoder.info().duration_ms > max_duration_ms) {
                    telemetry->result = StreamEndResult::kDurationCap;
                    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
                            "media duration exceeds GRPC_ASR_MAX_DURATION_SECONDS=" +
                                std::to_string(config_.max_duration_seconds)};
                }
                info->set_duration_ms(decoder.info().duration_ms);
                info->set_audio_codec(std::string(media::family_name(family)));
                info->set_sample_rate_hz(decoder.info().sample_rate_hz);
                info->set_channels(decoder.info().channels);
                info->set_has_video(false);
                if (!writer.write(info_response)) {
                    telemetry->result = StreamEndResult::kCancelled;
                    return cancelled;
                }

                engine::ModelPool::Lease lease = acquire_state(stop);
                result = engine::Transcriber::run(
                    lease.context(), lease.state(), base_engine_options,
                    [&](float* out, size_t max_samples) { return decoder.read(out, max_samples); },
                    segment_sink, stop,
                    [&](std::chrono::milliseconds timeout) {
                        return decoder.wait_for_input(timeout);
                    });
            } else {
                // Probe source channel count before acquiring any pool state or
                // decoding PCM, then run an independent AudioDecoder + Lease per
                // channel over the thread-safe ByteStream so a smaller pool can
                // never deadlock on a shared queue.
                uint32_t channels = 0;
                uint64_t duration_ms = 0;
                uint32_t sample_rate_hz = 0;
                {
                    media::AudioDecoder header_probe(upload, wav_like, /*preserve_channels=*/true);
                    channels = header_probe.info().channels;
                    duration_ms = header_probe.info().duration_ms;
                    sample_rate_hz = header_probe.info().sample_rate_hz;
                }
                if (channels == 0) {
                    telemetry->result = StreamEndResult::kInvalidMedia;
                    return {grpc::StatusCode::INVALID_ARGUMENT, "media reports 0 audio channels"};
                }
                if (channels > config_.max_channels) {
                    telemetry->result = StreamEndResult::kInvalidMedia;
                    return {grpc::StatusCode::INVALID_ARGUMENT,
                            "media channel count " + std::to_string(channels) +
                                " exceeds GRPC_ASR_MAX_CHANNELS=" +
                                std::to_string(config_.max_channels)};
                }
                if (duration_ms > max_duration_ms) {
                    telemetry->result = StreamEndResult::kDurationCap;
                    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
                            "media duration exceeds GRPC_ASR_MAX_DURATION_SECONDS=" +
                                std::to_string(config_.max_duration_seconds)};
                }
                info->set_duration_ms(duration_ms);
                info->set_audio_codec(std::string(media::family_name(family)));
                info->set_sample_rate_hz(sample_rate_hz);
                info->set_channels(channels);
                info->set_has_video(false);
                if (!writer.write(info_response)) {
                    telemetry->result = StreamEndResult::kCancelled;
                    return cancelled;
                }

                std::stop_source group_stop;
                std::stop_callback forward_caller(
                    stop, [&group_stop] { group_stop.request_stop(); });
                std::vector<engine::EngineResult> ch_results(channels);
                std::mutex err_mu;
                std::exception_ptr first_error;
                {
                    std::vector<std::jthread> workers;
                    workers.reserve(channels);
                    for (uint32_t ch = 0; ch < channels; ++ch) {
                        workers.emplace_back([&, ch]() {
                            try {
                                std::stop_token token = group_stop.get_token();
                                engine::ModelPool::Lease lease = acquire_state(token);
                                media::AudioDecoder ch_decoder(upload, wav_like,
                                                               /*preserve_channels=*/true);
                                engine::EngineOptions ch_opts = base_engine_options;
                                ch_opts.channel = ch;
                                std::vector<float> interleaved;
                                ch_results[ch] = engine::Transcriber::run(
                                    lease.context(), lease.state(), ch_opts,
                                    [&](float* out, size_t max_samples) -> size_t {
                                        interleaved.resize(max_samples * channels);
                                        const size_t frames =
                                            ch_decoder.read_frames(interleaved.data(), max_samples);
                                        for (size_t f = 0; f < frames; ++f) {
                                            out[f] = interleaved[f * channels + ch];
                                        }
                                        return frames;
                                    },
                                    segment_sink, token,
                                    [&](std::chrono::milliseconds timeout) {
                                        return ch_decoder.wait_for_input(timeout);
                                    });
                                if (ch_results[ch].aborted) {
                                    group_stop.request_stop();
                                }
                            } catch (...) {
                                std::lock_guard<std::mutex> lock(err_mu);
                                if (!first_error) {
                                    first_error = std::current_exception();
                                }
                                group_stop.request_stop();
                            }
                        });
                    }
                }
                if (first_error) {
                    std::rethrow_exception(first_error);
                }
                merge_channel_results(ch_results, &result, &segment_count_by_channel);
            }
        } else {
            const auto inactivity =
                std::chrono::milliseconds(config_.tool_inactivity_seconds * 1000);
            media::StreamLayout layout = media::StreamLayout::kSeekableMemfd;
            std::string layout_prefix = prefix;
            if (!used_extension_fallback) {
                constexpr size_t kMaxLayoutProbeBytes = 1024 * 1024;
                while (true) {
                    size_t needed = 0;
                    layout = media::sniff_stream_layout(
                        family, reinterpret_cast<const uint8_t*>(layout_prefix.data()),
                        layout_prefix.size(), upload.is_complete(), &needed);
                    if (layout != media::StreamLayout::kNeedMoreBytes) {
                        break;
                    }
                    if (needed <= layout_prefix.size() || needed > kMaxLayoutProbeBytes) {
                        layout = media::StreamLayout::kSeekableMemfd;
                        break;
                    }
                    const size_t got = upload.wait_for_prefix(needed);
                    if (upload.is_aborted()) {
                        telemetry->result = StreamEndResult::kCancelled;
                        return {grpc::StatusCode::CANCELLED, "upload aborted"};
                    }
                    layout_prefix = upload.prefix_bytes(got);
                }
            }

            if (layout == media::StreamLayout::kPipeable && !options.emit_keyframes() &&
                !per_channel) {
                // Live ffmpeg pipe path: ogg, mkv/webm, fast-start mp4/mov, amr,
                // mpeg-ps/ts, and ADTS aac demux directly from pipe:0 while the
                // upload is still in flight, skipping ffprobe and allocating no
                // memfd.
                first_segment_path = FirstSegmentPath::kLive;
                info->set_duration_ms(0);
                const auto codec = resolve_pipe_audio_codec(family, upload, layout_prefix);
                if (!codec.has_value()) {
                    telemetry->result = StreamEndResult::kCancelled;
                    return {grpc::StatusCode::CANCELLED, "upload aborted"};
                }
                info->set_audio_codec(*codec);
                info->set_sample_rate_hz(0);
                info->set_channels(0);
                info->set_has_video(false);
                if (!writer.write(info_response)) {
                    telemetry->result = StreamEndResult::kCancelled;
                    return cancelled;
                }

                std::optional<engine::ModelPool::Lease> lease(acquire_state(stop));
                media::VideoDemux demux(config_.ffmpeg, config_.ffprobe, inactivity);
                demux.open_audio_from_pipe(upload, stop);
                try {
                    result = engine::Transcriber::run(
                        lease->context(), lease->state(), base_engine_options,
                        [&](float* out, size_t max_samples) {
                            return demux.read_audio(out, max_samples);
                        },
                        segment_sink, stop,
                        [&](std::chrono::milliseconds timeout) {
                            return demux.wait_for_audio(timeout, stop);
                        });
                } catch (...) {
                    lease.reset();
                    demux.cancel_audio();
                    throw;
                }
                lease.reset();
                if (result.aborted) {
                    demux.cancel_audio();
                } else {
                    demux.close_audio();
                }
                // Wait for any trailing container footer bytes on the upload
                // before hashing for the Document fold.
                if (!result.aborted) {
                    upload.wait_complete();
                }
            } else {
                // Seekable memfd ffmpeg path: migrate ByteStream to a single
                // anonymous memfd now so remaining upload chunks land directly in
                // kernel pages with strictly 1x resident media memory.
                first_segment_path = FirstSegmentPath::kFfmpeg;
                upload.switch_to_memfd();
                upload.wait_complete();
                if (gone()) {
                    telemetry->result = StreamEndResult::kCancelled;
                    return {grpc::StatusCode::CANCELLED, "upload aborted"};
                }
                media::VideoDemux demux(upload.memfd(), config_.ffmpeg, config_.ffprobe,
                                        inactivity);
                media::ProbeInfo probe = demux.probe(stop);
                if (!probe.has_audio) {
                    telemetry->result = StreamEndResult::kInvalidMedia;
                    return {grpc::StatusCode::INVALID_ARGUMENT,
                            "the media has no audio track to transcribe"};
                }
                if (per_channel && (probe.channels == 0 || probe.channels > config_.max_channels)) {
                    telemetry->result = StreamEndResult::kInvalidMedia;
                    return {grpc::StatusCode::INVALID_ARGUMENT,
                            "media channel count " + std::to_string(probe.channels) +
                                " exceeds GRPC_ASR_MAX_CHANNELS=" +
                                std::to_string(config_.max_channels)};
                }
                if (probe.duration_ms > max_duration_ms) {
                    telemetry->result = StreamEndResult::kDurationCap;
                    return {grpc::StatusCode::RESOURCE_EXHAUSTED,
                            "media duration exceeds GRPC_ASR_MAX_DURATION_SECONDS=" +
                                std::to_string(config_.max_duration_seconds)};
                }
                info->set_duration_ms(probe.duration_ms);
                info->set_audio_codec(probe.audio_codec);
                info->set_sample_rate_hz(probe.sample_rate_hz);
                info->set_channels(probe.channels);
                info->set_has_video(probe.has_video);
                info->set_video_codec(probe.video_codec);
                if (!writer.write(info_response)) {
                    telemetry->result = StreamEndResult::kCancelled;
                    return cancelled;
                }

                std::exception_ptr keyframe_error;
                std::jthread keyframe_thread;
                auto start_keyframes_if_requested = [&]() {
                    if (options.emit_keyframes() && probe.has_video) {
                        const uint32_t interval =
                            options.keyframe_interval_seconds() != 0
                                ? options.keyframe_interval_seconds()
                                : static_cast<uint32_t>(config_.keyframe_interval_seconds);
                        keyframe_thread = std::jthread([&, interval](std::stop_token kf_stop) {
                            try {
                                demux.extract_keyframes(
                                    interval,
                                    [&](uint64_t timestamp_ms, uint32_t width, uint32_t height,
                                        std::string png) {
                                        if (kf_stop.stop_requested()) {
                                            return false;
                                        }
                                        asrv1::TranscribeResponse response;
                                        asrv1::Keyframe* frame = response.mutable_keyframe();
                                        frame->set_timestamp_ms(timestamp_ms);
                                        frame->set_width(width);
                                        frame->set_height(height);
                                        frame->set_png(std::move(png));
                                        if (!writer.write(response)) {
                                            return false;
                                        }
                                        keyframe_count++;
                                        return true;
                                    },
                                    kf_stop);
                            } catch (...) {
                                keyframe_error = std::current_exception();
                            }
                        });
                    }
                };

                if (!per_channel) {
                    std::optional<engine::ModelPool::Lease> lease(acquire_state(stop));
                    start_keyframes_if_requested();
                    std::stop_callback forward_stop(
                        stop, [source = keyframe_thread.get_stop_source()]() mutable {
                            source.request_stop();
                        });

                    demux.open_audio(stop);
                    result = engine::Transcriber::run(
                        lease->context(), lease->state(), base_engine_options,
                        [&](float* out, size_t max_samples) {
                            return demux.read_audio(out, max_samples);
                        },
                        segment_sink, stop);
                    lease.reset();
                    if (result.aborted) {
                        demux.cancel_audio();
                        keyframe_thread.request_stop();
                    } else {
                        demux.close_audio();
                    }
                } else {
                    start_keyframes_if_requested();
                    std::stop_callback forward_kf_stop(
                        stop, [source = keyframe_thread.get_stop_source()]() mutable {
                            source.request_stop();
                        });

                    const uint32_t channels = probe.channels;
                    std::stop_source group_stop;
                    std::stop_callback forward_caller(
                        stop, [&group_stop] { group_stop.request_stop(); });
                    std::vector<engine::EngineResult> ch_results(channels);
                    std::mutex err_mu;
                    std::exception_ptr first_error;
                    {
                        std::vector<std::jthread> workers;
                        workers.reserve(channels);
                        for (uint32_t ch = 0; ch < channels; ++ch) {
                            workers.emplace_back([&, ch]() {
                                try {
                                    std::stop_token token = group_stop.get_token();
                                    std::optional<engine::ModelPool::Lease> lease(
                                        acquire_state(token));
                                    media::VideoDemux ch_demux(upload.memfd(), config_.ffmpeg,
                                                               config_.ffprobe, inactivity);
                                    ch_demux.open_audio(token, ch);
                                    engine::EngineOptions ch_opts = base_engine_options;
                                    ch_opts.channel = ch;
                                    ch_results[ch] = engine::Transcriber::run(
                                        lease->context(), lease->state(), ch_opts,
                                        [&](float* out, size_t max_samples) {
                                            return ch_demux.read_audio(out, max_samples);
                                        },
                                        segment_sink, token);
                                    lease.reset();
                                    if (ch_results[ch].aborted) {
                                        ch_demux.cancel_audio();
                                        group_stop.request_stop();
                                    } else {
                                        ch_demux.close_audio();
                                    }
                                } catch (...) {
                                    std::lock_guard<std::mutex> lock(err_mu);
                                    if (!first_error) {
                                        first_error = std::current_exception();
                                    }
                                    group_stop.request_stop();
                                }
                            });
                        }
                    }
                    if (first_error) {
                        keyframe_thread.request_stop();
                        if (keyframe_thread.joinable()) {
                            keyframe_thread.join();
                        }
                        std::rethrow_exception(first_error);
                    }
                    merge_channel_results(ch_results, &result, &segment_count_by_channel);
                    if (result.aborted) {
                        keyframe_thread.request_stop();
                    }
                }

                if (keyframe_thread.joinable()) {
                    keyframe_thread.join();
                }
                if (!result.aborted && keyframe_error != nullptr) {
                    std::rethrow_exception(keyframe_error);
                }
            }
        }

        if (result.aborted || gone()) {
            telemetry->result = StreamEndResult::kCancelled;
            return cancelled;
        }

        if (writer.folding()) {
            writer.set_source(source_filename(options.filename(), family),
                              std::string(media::family_mimetype(family)),
                              upload.is_complete() ? upload.content_hash() : 0);
        }

        asrv1::TranscribeResponse trailer;
        asrv1::TranscriptComplete* complete = trailer.mutable_complete();
        complete->set_language(result.language);
        complete->set_duration_ms(result.duration_ms);
        complete->set_segment_count(result.final_segments);
        complete->set_token_count(result.tokens);
        complete->set_keyframe_count(keyframe_count.load());
        complete->set_window_seconds(telemetry->chosen_window_seconds);
        for (uint32_t count : segment_count_by_channel) {
            complete->add_segment_count_by_channel(count);
        }
        for (const std::string& lang : result.languages) {
            complete->add_languages(lang);
        }
        if (!writer.finish(trailer)) {
            telemetry->result = StreamEndResult::kCancelled;
            return cancelled;
        }

        if (!call.remote_policy.has_value() && result.duration_ms > 0 &&
            result.decode_wall_ns > 0) {
            const double audio_sec = static_cast<double>(result.duration_ms) / 1000.0;
            const double wall_sec = static_cast<double>(result.decode_wall_ns) / 1e9;
            if (wall_sec > 0.0) {
                metrics_observe_realtime_factor(options.model(), pool_.backend(),
                                                audio_sec / wall_sec);
            }
        }

        audio_ms += static_cast<long>(result.duration_ms);
        telemetry->result = StreamEndResult::kOk;
        telemetry->audio_ms = result.duration_ms;
        telemetry->keyframes = keyframe_count.load();
        telemetry->windows = result.windows;
        telemetry->window_redecodes = result.window_redecodes;
        return grpc::Status::OK;
    } catch (const Cancelled&) {
        telemetry->result = StreamEndResult::kCancelled;
        return cancelled;
    } catch (const engine::QueueTimeout& error) {
        telemetry->result = StreamEndResult::kQueueTimeout;
        return {grpc::StatusCode::RESOURCE_EXHAUSTED,
                std::string(error.what()) + " for GRPC_ASR_QUEUE_TIMEOUT_SECONDS=" +
                    std::to_string(config_.queue_timeout_seconds)};
    } catch (const media::DecodeError& error) {
        if (gone()) {
            telemetry->result = StreamEndResult::kCancelled;
            return {grpc::StatusCode::CANCELLED, "upload aborted mid-decode"};
        }
        telemetry->result = StreamEndResult::kInvalidMedia;
        return {grpc::StatusCode::INVALID_ARGUMENT, error.what()};
    } catch (const engine::DurationCapExceeded& error) {
        telemetry->result = StreamEndResult::kDurationCap;
        return {grpc::StatusCode::RESOURCE_EXHAUSTED, error.what()};
    } catch (const media::ToolError& error) {
        if (gone()) {
            telemetry->result = StreamEndResult::kCancelled;
            return cancelled;
        }
        telemetry->result = StreamEndResult::kToolError;
        return {grpc::StatusCode::INTERNAL, error.what()};
    } catch (const std::exception& error) {
        if (gone()) {
            telemetry->result = StreamEndResult::kCancelled;
            return cancelled;
        }
        telemetry->result = StreamEndResult::kInternal;
        return {grpc::StatusCode::INTERNAL,
                std::string("transcription failed: ") + error.what()};
    }
}

}  // namespace

AsrServiceImpl::AsrServiceImpl(const Config& config, engine::ModelPool& pool)
    : config_(config), pool_(pool) {}

grpc::Status AsrServiceImpl::Transcribe(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<asrv1::TranscribeResponse, asrv1::TranscribeRequest>* stream) {
    metrics_inc_streams_active();
    struct ActiveGuard {
        ~ActiveGuard() { metrics_dec_streams_active(); }
    } active_guard;

    StreamMetrics telemetry;
    auto reject = [&](StreamEndResult kind, grpc::Status status) {
        telemetry.result = kind;
        metrics_record_stream_end(telemetry);
        rejected++;
        return status;
    };

    asrv1::TranscribeRequest request;
    if (!stream->Read(&request) || request.payload_case() != asrv1::TranscribeRequest::kOptions) {
        return reject(StreamEndResult::kInvalidMedia,
                      {grpc::StatusCode::INVALID_ARGUMENT,
                       "the first message must carry TranscribeOptions"});
    }
    const asrv1::TranscribeOptions options = request.options();
    if (grpc::Status status = engine::validate_docling_asr_options(options); !status.ok()) {
        const StreamEndResult kind =
            status.error_code() == grpc::StatusCode::UNIMPLEMENTED
                ? StreamEndResult::kUnknownContainer
                : StreamEndResult::kInvalidMedia;
        return reject(kind, std::move(status));
    }
    if (options.channel_mode() != asrv1::CHANNEL_MODE_UNSPECIFIED &&
        options.channel_mode() != asrv1::CHANNEL_MODE_MIXDOWN &&
        options.channel_mode() != asrv1::CHANNEL_MODE_PER_CHANNEL) {
        return reject(StreamEndResult::kInvalidMedia,
                      {grpc::StatusCode::INVALID_ARGUMENT, "unknown TranscribeOptions.channel_mode"});
    }

    PreparedCall call;
    call.eff_min_window_seconds =
        std::max<size_t>(1, std::min(config_.min_window_seconds, config_.window_seconds));
    const size_t eff_max_window = std::max<size_t>(1, config_.window_seconds);
    call.chosen_window_seconds =
        options.window_seconds() == 0
            ? eff_max_window
            : std::clamp<size_t>(options.window_seconds(), call.eff_min_window_seconds,
                                 eff_max_window);

    if (options.has_remote()) {
        if (grpc::Status status = engine::validate_remote_options(options.remote()); !status.ok()) {
            return reject(StreamEndResult::kInvalidMedia, std::move(status));
        }
        engine::RemoteEndpointPolicy policy;
        if (grpc::Status status = engine::authorize_remote_endpoint(
                options.remote().url(), config_.remote_allowed_hosts, &policy);
            !status.ok()) {
            const StreamEndResult kind =
                status.error_code() == grpc::StatusCode::UNAVAILABLE
                    ? StreamEndResult::kInternal
                    : StreamEndResult::kInvalidMedia;
            telemetry.result = kind;
            telemetry.model = "remote";
            metrics_record_stream_end(telemetry);
            if (status.error_code() == grpc::StatusCode::UNAVAILABLE) {
                failed++;
            } else {
                rejected++;
            }
            return status;
        }
        call.options = options;
        engine::apply_model_type_default(&call.options);
        call.remote_policy = std::move(policy);
    } else {
        if (const auto err = engine::parse_decoding_options(options.decoding(), &call.decoding);
            err.has_value()) {
            return reject(StreamEndResult::kInvalidMedia,
                          {grpc::StatusCode::INVALID_ARGUMENT, *err});
        }
        asrv1::TranscribeOptions resolved = options;
        engine::apply_model_type_default(&resolved);
        if (resolved.model().empty()) {
            return reject(StreamEndResult::kInvalidMedia,
                          {grpc::StatusCode::INVALID_ARGUMENT,
                           "TranscribeOptions.model is required"});
        }
        if (!pool_.has_model(resolved.model())) {
            std::string loaded;
            for (const std::string& name : pool_.model_names()) {
                loaded += loaded.empty() ? name : ", " + name;
            }
            return reject(StreamEndResult::kInvalidMedia,
                          {grpc::StatusCode::INVALID_ARGUMENT,
                           "model '" + resolved.model() + "' is not loaded (loaded: " + loaded +
                               ")"});
        }
        call.options = std::move(resolved);
    }

    media::ByteStream upload;
    std::optional<doc::AsrDocumentFold> fold;
    if (call.options.emit_document()) {
        const std::string fold_model =
            call.options.has_remote()
                ? (call.options.remote().model().empty() ? std::string("remote")
                                                         : call.options.remote().model())
                : call.options.model();
        fold.emplace(fold_model, GRPC_ASR_VERSION,
                     doc::FoldOptions{.word_provenance = config_.document_word_provenance});
    }
    LockedWriter writer(stream, fold.has_value() ? &*fold : nullptr);

    std::stop_source stop;
    grpc::Status worker_status = grpc::Status::OK;
    std::mutex worker_mutex;
    std::condition_variable worker_finished;
    bool worker_done = false;
    std::thread worker([&] {
        worker_status = process_stream(config_, pool_, call, upload, writer, audio_ms, &telemetry,
                                       stop.get_token());
        std::lock_guard<std::mutex> lock(worker_mutex);
        worker_done = true;
        worker_finished.notify_all();
    });

    grpc::Status reader_status = grpc::Status::OK;
    StreamEndResult reader_end = StreamEndResult::kOk;
    bool saw_bytes = false;
    while (stream->Read(&request)) {
        if (request.payload_case() != asrv1::TranscribeRequest::kChunk) {
            reader_status = {grpc::StatusCode::INVALID_ARGUMENT,
                             "only media chunks may follow TranscribeOptions"};
            reader_end = StreamEndResult::kInvalidMedia;
            break;
        }
        const std::string& data = request.chunk().data();
        if (upload.size() + data.size() > config_.max_media_bytes) {
            reader_status = {grpc::StatusCode::RESOURCE_EXHAUSTED,
                             "media exceeds GRPC_ASR_MAX_MEDIA_BYTES=" +
                                 std::to_string(config_.max_media_bytes)};
            reader_end = StreamEndResult::kByteCap;
            break;
        }
        saw_bytes = saw_bytes || !data.empty();
        upload.append(data.data(), data.size());
    }

    if (!reader_status.ok() || context->IsCancelled()) {
        stop.request_stop();
        upload.abort();
    } else if (!saw_bytes) {
        reader_status = {grpc::StatusCode::INVALID_ARGUMENT,
                         "stream ended without media bytes"};
        reader_end = StreamEndResult::kInvalidMedia;
        stop.request_stop();
        upload.abort();
    } else {
        upload.complete();
    }

    {
        std::unique_lock<std::mutex> lock(worker_mutex);
        while (!worker_finished.wait_for(lock, kCallWatchInterval, [&] { return worker_done; })) {
            if (context->IsCancelled()) {
                stop.request_stop();
                break;
            }
        }
    }
    worker.join();

    grpc::Status status = reader_status.ok() ? worker_status : reader_status;
    if (!reader_status.ok()) {
        telemetry.result = reader_end;
    }
    if (telemetry.result == StreamEndResult::kCancelled &&
        context->deadline() <= std::chrono::system_clock::now()) {
        telemetry.result = StreamEndResult::kDeadline;
    }
    metrics_record_stream_end(telemetry);

    if (status.ok()) {
        transcribed++;
    } else if (status.error_code() == grpc::StatusCode::INVALID_ARGUMENT ||
               status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED ||
               status.error_code() == grpc::StatusCode::UNIMPLEMENTED ||
               status.error_code() == grpc::StatusCode::FAILED_PRECONDITION ||
               status.error_code() == grpc::StatusCode::PERMISSION_DENIED) {
        rejected++;
    } else {
        failed++;
    }
    return status;
}

grpc::Status AsrServiceImpl::GetServiceInfo(grpc::ServerContext* /*context*/,
                                            const asrv1::GetServiceInfoRequest* /*request*/,
                                            asrv1::GetServiceInfoResponse* response) {
    response->set_version(GRPC_ASR_VERSION);
    response->set_whisper_version(whisper_version());
    response->set_backend(pool_.backend());
    for (const std::string& name : pool_.model_names()) {
        response->add_models(name);
    }
    response->set_max_media_bytes(config_.max_media_bytes);
    response->set_max_duration_ms(static_cast<uint64_t>(config_.max_duration_seconds) * 1000ULL);
    asrv1::UiInfo* ui = response->mutable_ui();
    ui->set_title("ASR");
    ui->set_path("/ui/asr");
    ui->set_description("Speech-to-text via embedded whisper.cpp");
    const uint32_t eff_max_window =
        static_cast<uint32_t>(std::max<size_t>(1, config_.window_seconds));
    const uint32_t eff_min_window = static_cast<uint32_t>(
        std::max<size_t>(1, std::min(config_.min_window_seconds, config_.window_seconds)));
    response->set_min_window_seconds(eff_min_window);
    response->set_max_window_seconds(eff_max_window);
    response->set_warmed_up(pool_.warmed_up());
    response->set_remote_enabled(!config_.remote_allowed_hosts.empty());
    return grpc::Status::OK;
}

}  // namespace asr
