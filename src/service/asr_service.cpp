#include "service/asr_service.h"

#include <atomic>
#include <exception>
#include <mutex>
#include <string>
#include <thread>

#include "engine/transcriber.h"
#include "media/audio_decoder.h"
#include "media/byte_stream.h"
#include "media/sniff.h"
#include "media/video_demux.h"
#include "whisper.h"

namespace asr {

namespace asrv1 = ai::pipestream::asr::v1;

namespace {

#ifndef GRPC_ASR_VERSION
#define GRPC_ASR_VERSION "0.0.0-dev"
#endif

// gRPC stream Write is not thread-safe; the transcription worker and the
// keyframe thread share the stream through this lock. write() returning
// false means the client is gone and the producer should stop.
class LockedWriter {
  public:
    explicit LockedWriter(
        grpc::ServerReaderWriter<asrv1::TranscribeResponse, asrv1::TranscribeRequest>* stream)
        : stream_(stream) {}

    bool write(const asrv1::TranscribeResponse& response) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (dead_) {
            return false;
        }
        if (!stream_->Write(response)) {
            dead_ = true;
            return false;
        }
        return true;
    }

  private:
    grpc::ServerReaderWriter<asrv1::TranscribeResponse, asrv1::TranscribeRequest>* stream_;
    std::mutex mutex_;
    bool dead_ = false;
};

void fill_segment(const engine::EngineSegment& source, asrv1::Segment* out) {
    out->set_index(source.index);
    out->set_start_ms(source.start_ms);
    out->set_end_ms(source.end_ms);
    out->set_text(source.text);
    out->set_avg_logprob(source.avg_logprob);
    for (const engine::EngineWord& word : source.words) {
        asrv1::Word* out_word = out->add_words();
        out_word->set_text(word.text);
        out_word->set_start_ms(word.start_ms);
        out_word->set_end_ms(word.end_ms);
        out_word->set_probability(word.probability);
    }
}

// The transcription pipeline, run on a worker thread concurrently with
// the upload so segments stream while media is still arriving. Returns
// the final RPC status; on success the Complete trailer has been written.
grpc::Status process_stream(const Config& config_, engine::ModelPool& pool_,
                            const asrv1::TranscribeOptions& options, media::ByteStream& upload,
                            LockedWriter& writer, std::atomic<long>& audio_ms) {
    const uint64_t max_duration_ms =
        static_cast<uint64_t>(config_.max_duration_seconds) * 1000ULL;

    engine::EngineOptions engine_options;
    engine_options.language = options.language();
    engine_options.translate = options.task() == asrv1::TASK_TRANSLATE;
    engine_options.word_timestamps = options.word_timestamps();
    engine_options.threads = static_cast<int>(config_.threads);
    engine_options.window_seconds = config_.window_seconds;
    engine_options.max_duration_seconds = config_.max_duration_seconds;

    try {
        // Sniff the container from the first bytes; blocks only until the
        // first chunk lands, not the whole upload.
        uint8_t prefix[16] = {};
        size_t prefix_size = upload.wait_for_prefix(sizeof prefix);
        if (upload.is_aborted()) {
            return {grpc::StatusCode::CANCELLED, "upload aborted"};
        }
        upload.read_at(0, prefix, prefix_size);
        const media::MediaFamily family = media::sniff(prefix, prefix_size);
        if (family == media::MediaFamily::kUnknown) {
            return {grpc::StatusCode::UNIMPLEMENTED,
                    "unrecognized media container; supported: wav, mp3, flac, ogg, mp4/mov, "
                    "mkv/webm"};
        }

        asrv1::TranscribeResponse info_response;
        asrv1::MediaInfo* info = info_response.mutable_media_info();
        engine::EngineResult result;
        std::atomic<uint32_t> keyframe_count{0};

        auto segment_sink = [&](const engine::EngineSegment& segment, bool is_final) {
            asrv1::TranscribeResponse response;
            fill_segment(segment, is_final ? response.mutable_final_segment()
                                           : response.mutable_partial_segment());
            return writer.write(response);
        };

        if (media::is_audio_family(family)) {
            // Live path: the decoder pulls from the growing upload, so
            // MediaInfo and the first segments go out before half-close.
            media::AudioDecoder decoder(upload, family == media::MediaFamily::kWav);
            if (decoder.info().duration_ms > max_duration_ms) {
                return {grpc::StatusCode::RESOURCE_EXHAUSTED,
                        "media duration exceeds GRPC_ASR_MAX_DURATION_SECONDS=" +
                            std::to_string(config_.max_duration_seconds)};
            }
            info->set_duration_ms(decoder.info().duration_ms);
            info->set_audio_codec(std::string(media::family_name(family)));
            info->set_sample_rate_hz(decoder.info().sample_rate_hz);
            info->set_channels(decoder.info().channels);
            info->set_has_video(false);
            writer.write(info_response);

            engine::ModelPool::Lease lease = pool_.acquire(options.model());
            result = engine::Transcriber::run(
                lease.context(), lease.state(), engine_options,
                [&](float* out, size_t max_samples) { return decoder.read(out, max_samples); },
                segment_sink);
        } else {
            // Video path: a classic mp4's moov index can trail the file,
            // so ffmpeg needs the complete bytes in the (seekable) memfd.
            // Streamable containers (mpeg-ts, fragmented mp4) via a pipe
            // are the designed follow-up; see docs/design.md.
            upload.wait_complete();
            if (upload.is_aborted()) {
                return {grpc::StatusCode::CANCELLED, "upload aborted"};
            }
            const std::string& media_bytes = upload.completed_bytes();
            media::VideoDemux demux(
                reinterpret_cast<const uint8_t*>(media_bytes.data()), media_bytes.size(),
                config_.ffmpeg, config_.ffprobe,
                std::chrono::milliseconds(config_.tool_inactivity_seconds * 1000));
            media::ProbeInfo probe = demux.probe();
            if (!probe.has_audio) {
                return {grpc::StatusCode::INVALID_ARGUMENT,
                        "video has no audio track to transcribe"};
            }
            if (probe.duration_ms > max_duration_ms) {
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
            writer.write(info_response);

            engine::ModelPool::Lease lease = pool_.acquire(options.model());

            // Keyframes stream from their own ffmpeg child concurrently
            // with transcription; the LockedWriter interleaves the two.
            std::thread keyframe_thread;
            std::exception_ptr keyframe_error;
            if (options.emit_keyframes() && probe.has_video) {
                uint32_t interval =
                    options.keyframe_interval_seconds() != 0
                        ? options.keyframe_interval_seconds()
                        : static_cast<uint32_t>(config_.keyframe_interval_seconds);
                keyframe_thread = std::thread([&, interval] {
                    try {
                        demux.extract_keyframes(
                            interval, [&](uint64_t timestamp_ms, uint32_t width, uint32_t height,
                                          std::string png) {
                                asrv1::TranscribeResponse response;
                                asrv1::Keyframe* frame = response.mutable_keyframe();
                                frame->set_timestamp_ms(timestamp_ms);
                                frame->set_width(width);
                                frame->set_height(height);
                                frame->set_png(std::move(png));
                                writer.write(response);
                                keyframe_count++;
                            });
                    } catch (...) {
                        keyframe_error = std::current_exception();
                    }
                });
            }

            demux.open_audio();
            try {
                result = engine::Transcriber::run(
                    lease.context(), lease.state(), engine_options,
                    [&](float* out, size_t max_samples) {
                        return demux.read_audio(out, max_samples);
                    },
                    segment_sink);
            } catch (...) {
                if (keyframe_thread.joinable()) {
                    keyframe_thread.join();
                }
                throw;
            }
            demux.close_audio();
            if (keyframe_thread.joinable()) {
                keyframe_thread.join();
            }
            if (keyframe_error != nullptr) {
                std::rethrow_exception(keyframe_error);
            }
        }

        if (result.aborted || upload.is_aborted()) {
            return {grpc::StatusCode::CANCELLED, "the stream went away mid-transcription"};
        }

        asrv1::TranscribeResponse trailer;
        asrv1::TranscriptComplete* complete = trailer.mutable_complete();
        complete->set_language(result.language);
        complete->set_duration_ms(result.duration_ms);
        complete->set_segment_count(result.final_segments);
        complete->set_token_count(result.tokens);
        complete->set_keyframe_count(keyframe_count.load());
        writer.write(trailer);

        audio_ms += static_cast<long>(result.duration_ms);
        return grpc::Status::OK;
    } catch (const media::DecodeError& error) {
        // An aborted upload surfaces to the decoder as truncation; report
        // the disconnect, not a media defect.
        if (upload.is_aborted()) {
            return {grpc::StatusCode::CANCELLED, "upload aborted mid-decode"};
        }
        return {grpc::StatusCode::INVALID_ARGUMENT, error.what()};
    } catch (const engine::DurationCapExceeded& error) {
        return {grpc::StatusCode::RESOURCE_EXHAUSTED, error.what()};
    } catch (const media::ToolError& error) {
        return {grpc::StatusCode::INTERNAL, error.what()};
    } catch (const std::exception& error) {
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
    // The first message must carry the options; everything about the
    // request validates before any media byte is accepted or a pool slot
    // is touched.
    asrv1::TranscribeRequest request;
    if (!stream->Read(&request) || request.payload_case() != asrv1::TranscribeRequest::kOptions) {
        rejected++;
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "the first message must carry TranscribeOptions"};
    }
    const asrv1::TranscribeOptions options = request.options();
    if (options.model().empty()) {
        rejected++;
        return {grpc::StatusCode::INVALID_ARGUMENT, "TranscribeOptions.model is required"};
    }
    if (!pool_.has_model(options.model())) {
        std::string loaded;
        for (const std::string& name : pool_.model_names()) {
            loaded += loaded.empty() ? name : ", " + name;
        }
        rejected++;
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "model '" + options.model() + "' is not loaded (loaded: " + loaded + ")"};
    }

    media::ByteStream upload;
    LockedWriter writer(stream);
    grpc::Status worker_status = grpc::Status::OK;
    std::thread worker([&] {
        worker_status = process_stream(config_, pool_, options, upload, writer, audio_ms);
    });

    // Upload loop, concurrent with the worker. Reader-side failures
    // (double options, byte cap) abort the worker and win over its
    // status.
    grpc::Status reader_status = grpc::Status::OK;
    bool saw_bytes = false;
    while (stream->Read(&request)) {
        if (request.payload_case() != asrv1::TranscribeRequest::kChunk) {
            reader_status = {grpc::StatusCode::INVALID_ARGUMENT,
                             "only media chunks may follow TranscribeOptions"};
            break;
        }
        const std::string& data = request.chunk().data();
        if (upload.size() + data.size() > config_.max_media_bytes) {
            reader_status = {grpc::StatusCode::RESOURCE_EXHAUSTED,
                             "media exceeds GRPC_ASR_MAX_MEDIA_BYTES=" +
                                 std::to_string(config_.max_media_bytes)};
            break;
        }
        saw_bytes = saw_bytes || !data.empty();
        upload.append(data.data(), data.size());
    }

    if (!reader_status.ok() || context->IsCancelled()) {
        upload.abort();
    } else if (!saw_bytes) {
        reader_status = {grpc::StatusCode::INVALID_ARGUMENT,
                         "stream ended without media bytes"};
        upload.abort();
    } else {
        upload.complete();
    }
    worker.join();

    grpc::Status status = reader_status.ok() ? worker_status : reader_status;
    if (status.ok()) {
        transcribed++;
    } else if (status.error_code() == grpc::StatusCode::INVALID_ARGUMENT ||
               status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED ||
               status.error_code() == grpc::StatusCode::UNIMPLEMENTED) {
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
    return grpc::Status::OK;
}

}  // namespace asr
