// End-to-end over gRPC loopback: upload → MediaInfo → partial/final
// segments → keyframes → TranscriptComplete, plus the full error-code
// matrix and the wire-level anti-batch proof (a final segment arrives
// while the client still holds the last media chunk).
//
// Needs GRPC_ASR_TEST_MODEL and GRPC_ASR_TEST_SAMPLE; skips with 77
// otherwise. Video cases additionally need ffmpeg/ffprobe on PATH.

#include "service/asr_service.h"

#include <grpcpp/grpcpp.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <print>
#include <thread>
#include <vector>

#include "config.h"
#include "document/document_fold.h"
#include "engine/model_pool.h"
#include "fixture.h"
#include "ggml-backend.h"

namespace asrv1 = ai::pipestream::asr::v1;

namespace {

constexpr size_t kChunk = 64 * 1024;

// FNV-1a 64, written out here so the origin hash is checked against a
// second implementation rather than the server's own.
uint64_t content_hash(const std::string& bytes) {
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return text;
}

struct StreamResult {
    bool got_media_info = false;
    bool media_info_first = false;
    asrv1::MediaInfo media_info;
    std::vector<asrv1::Segment> partials;
    std::vector<asrv1::Segment> finals;
    size_t keyframes = 0;
    std::vector<asrv1::Keyframe> keyframe_samples;
    bool got_complete = false;
    bool complete_last = true;
    asrv1::TranscriptComplete complete;
    size_t documents = 0;
    ai::pipestream::document::v1::Document document;
    // Set when a final segment was received before the upload finished —
    // the wire-level proof the server is not batching.
    bool final_before_upload_done = false;
    grpc::Status status;
};

void consume_event(StreamResult& result, const asrv1::TranscribeResponse& response,
                   bool upload_done, size_t events_seen) {
    if (result.got_complete) {
        result.complete_last = false;  // something followed the trailer
    }
    switch (response.event_case()) {
        case asrv1::TranscribeResponse::kMediaInfo:
            result.got_media_info = true;
            result.media_info_first = events_seen == 0;
            result.media_info = response.media_info();
            break;
        case asrv1::TranscribeResponse::kPartialSegment:
            result.partials.push_back(response.partial_segment());
            break;
        case asrv1::TranscribeResponse::kFinalSegment:
            if (!upload_done) {
                result.final_before_upload_done = true;
            }
            result.finals.push_back(response.final_segment());
            break;
        case asrv1::TranscribeResponse::kKeyframe:
            result.keyframes++;
            if (result.keyframe_samples.size() < 2) {
                result.keyframe_samples.push_back(response.keyframe());
            }
            break;
        case asrv1::TranscribeResponse::kComplete:
            result.got_complete = true;
            result.complete = response.complete();
            break;
        case asrv1::TranscribeResponse::kDocument:
            result.documents++;
            result.document = response.document();
            break;
        default:
            break;
    }
}

// Drives one Transcribe stream. hold_back > 0 withholds that many bytes
// of the media tail until a final segment (or stream end) has been
// observed, proving transcription runs during the upload.
StreamResult transcribe(const std::shared_ptr<grpc::Channel>& channel,
                        const asrv1::TranscribeOptions& options, const std::string& media,
                        size_t hold_back = 0, bool skip_options = false) {
    StreamResult result;
    auto stub = asrv1::AsrService::NewStub(channel);
    grpc::ClientContext context;
    auto stream = stub->Transcribe(&context);

    asrv1::TranscribeRequest request;
    if (!skip_options) {
        *request.mutable_options() = options;
        stream->Write(request);
    }
    size_t send_now = media.size() > hold_back ? media.size() - hold_back : 0;
    for (size_t offset = 0; offset < send_now; offset += kChunk) {
        request.Clear();
        request.mutable_chunk()->set_data(media.substr(offset, std::min(kChunk, send_now - offset)));
        if (!stream->Write(request)) {
            break;
        }
    }

    size_t events_seen = 0;
    asrv1::TranscribeResponse response;
    if (hold_back > 0) {
        // Read until the server proves it is transcribing what it has.
        while (!result.final_before_upload_done && stream->Read(&response)) {
            consume_event(result, response, /*upload_done=*/false, events_seen++);
        }
        request.Clear();
        request.mutable_chunk()->set_data(media.substr(send_now));
        stream->Write(request);
    }
    stream->WritesDone();
    while (stream->Read(&response)) {
        consume_event(result, response, /*upload_done=*/true, events_seen++);
    }
    result.status = stream->Finish();
    return result;
}

asrv1::TranscribeOptions options_for(const std::string& model) {
    asrv1::TranscribeOptions options;
    options.set_model(model);
    options.set_language("en");
    return options;
}

std::string final_text(const StreamResult& result) {
    std::string text;
    for (const asrv1::Segment& segment : result.finals) {
        text += segment.text();
    }
    return text;
}

bool have_ffmpeg() {
    return std::system("ffmpeg -version >/dev/null 2>&1") == 0 &&
           std::system("ffprobe -version >/dev/null 2>&1") == 0;
}

std::string generate_media(const std::string& args, const std::string& suffix) {
    std::string path = "/tmp/grpc-asr-e2e-XXXXXX" + suffix;
    int fd = mkstemps(path.data(), static_cast<int>(suffix.size()));
    require(fd >= 0, "temp fixture path");
    ::close(fd);
    std::string command = "ffmpeg -v error -y " + args + " " + path + " >/dev/null 2>&1";
    require(std::system(command.c_str()) == 0, "fixture generation: " + command);
    std::string bytes = slurp(path);
    std::remove(path.c_str());
    require(!bytes.empty(), "fixture has bytes");
    return bytes;
}

// ---- cases -------------------------------------------------------------

void verify_wav(const std::shared_ptr<grpc::Channel>& channel, const std::string& jfk) {
    StreamResult result = transcribe(channel, options_for("tiny.en"), jfk);
    require(result.status.ok(), "wav transcribes OK: " + result.status.error_message());
    require(result.got_media_info && result.media_info_first, "MediaInfo is the first event");
    require(result.media_info.duration_ms() > 10000 && result.media_info.duration_ms() < 12000,
            "MediaInfo duration close to 11s");
    require(!result.media_info.has_video(), "wav has no video");
    require(!result.partials.empty(), "partials streamed");
    require(!result.finals.empty(), "finals streamed");
    require(lower(final_text(result)).find("country") != std::string::npos,
            "transcript contains 'country', got: " + final_text(result));
    uint64_t last_start = 0;
    for (size_t i = 0; i < result.finals.size(); i++) {
        require(result.finals[i].index() == i, "final indexes dense and ordered");
        require(result.finals[i].start_ms() >= last_start, "timestamps monotonic");
        last_start = result.finals[i].start_ms();
    }
    require(result.got_complete && result.complete_last, "Complete is the last event");
    require(result.complete.segment_count() == result.finals.size(),
            "trailer counts the finals");
    require(result.complete.language() == "en", "language in the trailer");
    require(result.keyframes == 0 && result.complete.keyframe_count() == 0,
            "audio-only media never emits keyframes");
    require(result.documents == 0, "no document event unless emit_document was set");
}

void verify_document(const std::shared_ptr<grpc::Channel>& channel, const std::string& jfk) {
    namespace docv1 = ai::pipestream::document::v1;
    asrv1::TranscribeOptions options = options_for("tiny.en");
    options.set_emit_document(true);
    options.set_word_timestamps(true);
    options.set_filename("/uploads/address.wav");
    StreamResult result = transcribe(channel, options, jfk);
    require(result.status.ok(), "document run transcribes OK: " + result.status.error_message());
    require(result.documents == 1, "exactly one document event");
    require(result.got_complete && result.complete_last,
            "the document precedes Complete, which stays the trailer");

    const docv1::Document& document = result.document;
    std::vector<std::string> errors = asr::doc::document_integrity_errors(document);
    require(errors.empty(), "document integrity: " + (errors.empty() ? "" : errors.front()));
    require(document.schema_name() == "docling_document_v2", "schema name");
    require(static_cast<size_t>(document.texts_size()) == result.finals.size(),
            "one text item per final segment");
    require(document.body().children_size() == document.texts_size(),
            "body children mirror the text arena");

    // Source identity: the client named the media, the server sniffed the
    // container, and the hash is over the bytes that arrived.
    require(document.name() == "address.wav" && document.origin().filename() == "address.wav",
            "the document is named after the upload, basename only");
    require(document.origin().mimetype() == "audio/wav", "origin mimetype from the sniff");
    require(document.origin().binary_hash() == content_hash(jfk),
            "origin hash is the content hash of the uploaded bytes");

    std::string all_text;
    double last_start = -1.0;
    size_t items_with_words = 0;
    for (size_t i = 0; i < static_cast<size_t>(document.texts_size()); i++) {
        const docv1::TextItemBase& base = document.texts(i).text().base();
        all_text += base.text() + " ";
        require(base.source_size() == 2, "every text item carries track + collector sources");
        require(base.source(0).has_track() && base.source(1).has_collector(),
                "source order is track then collector");
        const docv1::TrackSource& track = base.source(0).track();
        require(track.start_time() >= last_start, "track times monotonic");
        require(track.end_time() > track.start_time(),
                "track range strictly positive (the upstream TrackSource validator)");
        last_start = track.start_time();
        const docv1::CollectorSource& collector = base.source(1).collector();
        require(collector.collector() == "asr" && collector.model() == "tiny.en",
                "collector attribution names asr and the model");
        require(collector.confidence() > 0.0 && collector.confidence() <= 1.0,
                "confidence derived from the segment's avg logprob");
        require(!base.meta().custom_fields().contains("pipestream__avg_logprob"),
                "the retired meta custom field is gone from a real transcription");

        // Time provenance: the segment entry, then one entry per word the
        // decoder aligned.
        const asrv1::Segment& segment = result.finals[i];

        // The raw decoder score is typed on the collector source and is the
        // typed stream's own number, byte for byte, not a rescale of it.
        require(collector.has_raw_score() == segment.has_avg_logprob(),
                "raw_score is set exactly when the decoder scored the segment");
        if (segment.has_avg_logprob()) {
            require(collector.raw_score() == static_cast<double>(segment.avg_logprob()),
                    "raw_score carries the typed segment's avg logprob verbatim");
            require(collector.raw_score_kind() == "avg_logprob",
                    "raw_score_kind names the engine's signal");
            require(collector.raw_score() <= 0.0, "a mean token logprob is at most zero");
        } else {
            require(!collector.has_raw_score_kind(),
                    "an unscored segment names no raw score kind");
        }
        require(base.prov_size() == 1 + segment.words_size(),
                "one provenance entry for the segment plus one per word");
        const docv1::ProvenanceItem& span = base.prov(0);
        require(span.time().start_ms() == static_cast<double>(segment.start_ms()) &&
                    span.time().end_ms() == static_cast<double>(segment.end_ms()),
                "segment provenance matches the typed segment, in milliseconds");
        require(span.charspan().start() == 0 && span.charspan().end() > 0,
                "segment charspan covers the item text");
        items_with_words += segment.words_size() > 0 ? 1 : 0;
        int32_t last_word_start = -1;
        for (int w = 0; w < segment.words_size(); w++) {
            const docv1::ProvenanceItem& word = base.prov(w + 1);
            require(word.time().start_ms() == static_cast<double>(segment.words(w).start_ms()),
                    "word provenance keeps the typed word timing");
            require(word.has_charspan(), "every aligned word is located in the item text");
            require(word.charspan().start() >= last_word_start &&
                        word.charspan().end() <= span.charspan().end(),
                    "word charspans advance and stay inside the item text");
            last_word_start = word.charspan().start();
        }
    }
    require(items_with_words > 0, "the run actually produced word timings to locate");
    require(lower(all_text).find("country") != std::string::npos,
            "document text matches the transcript, got: " + all_text);
    require(document.body().meta().language().code_raw() == "en" &&
                document.body().meta().language().code() == docv1::HUMAN_LANGUAGE_LABEL_EN,
            "trailer language folded into the body meta language slots");
    require(document.source_meta().language() == "en", "and into the document meta");
    require(document.media().duration_ms() > 10000.0 && document.media().duration_ms() < 12000.0,
            "media meta carries the decoded duration");
}

void verify_streaming_during_upload(const std::shared_ptr<grpc::Channel>& channel,
                                    const std::string& jfk) {
    // Hold back the last ~0.5 seconds — the ~10.5s already sent overfill
    // the server's 10s window, which must transcribe and deliver a final
    // segment while we still hold the tail. Whoever makes the server
    // buffer the whole upload before transcribing breaks this test (it
    // deadlocks, visibly, under the ctest timeout).
    StreamResult result = transcribe(channel, options_for("tiny.en"), jfk, /*hold_back=*/16000);
    require(result.final_before_upload_done,
            "a final segment must arrive before the upload is finished");
    require(result.status.ok(), "streamed upload still completes OK");
    require(lower(final_text(result)).find("country") != std::string::npos,
            "held-back tail still transcribed");
}

void verify_silence(const std::shared_ptr<grpc::Channel>& channel) {
    StreamResult result = transcribe(channel, options_for("tiny.en"), make_wav(12.0, 0.0));
    require(result.status.ok(), "silence transcribes OK");
    require(result.got_complete, "silence reaches Complete");
    std::string transcript = final_text(result);
    for (char c : transcript) {
        require(!std::isalnum(static_cast<unsigned char>(c)) ||
                    transcript.find('[') != std::string::npos,
                "silence produced words: " + transcript);
    }
}

void verify_error_matrix(const std::shared_ptr<grpc::Channel>& channel) {
    StreamResult result =
        transcribe(channel, options_for("tiny.en"), make_truncated_mp3());
    require(result.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
            "truncated mp3 is INVALID_ARGUMENT, got " + result.status.error_message());

    result = transcribe(channel, options_for("tiny.en"), "plain text, definitely not media");
    require(result.status.error_code() == grpc::StatusCode::UNIMPLEMENTED,
            "unknown container is UNIMPLEMENTED");

    result = transcribe(channel, options_for("no-such-model"), make_wav(1.0, 440.0));
    require(result.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
            "unknown model is INVALID_ARGUMENT");
    require(result.status.error_message().find("tiny.en") != std::string::npos,
            "the error names the loaded models");

    result = transcribe(channel, options_for("tiny.en"), make_wav(1.0, 440.0), 0,
                        /*skip_options=*/true);
    require(result.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
            "missing options is INVALID_ARGUMENT");

    result = transcribe(channel, options_for("tiny.en"), "");
    require(result.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
            "empty media is INVALID_ARGUMENT");
}

void verify_byte_cap(asr::Config config, asr::engine::ModelPool& pool) {
    // A dedicated server whose cap is smaller than the upload.
    config.max_media_bytes = 4096;
    asr::AsrServiceImpl service(config, pool);
    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    require(server != nullptr, "cap server started");
    auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                                       grpc::InsecureChannelCredentials());
    StreamResult result = transcribe(channel, options_for("tiny.en"), make_wav(4.0, 440.0));
    require(result.status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED,
            "over-cap media is RESOURCE_EXHAUSTED");
    server->Shutdown();
}

void verify_video(const std::shared_ptr<grpc::Channel>& channel, const std::string& jfk_path) {
    // testsrc2 video muxed with the spoken fixture audio, generated at
    // test time.
    std::string media = generate_media(
        "-f lavfi -i testsrc2=duration=11:size=320x240:rate=10 -i " + jfk_path +
            " -shortest -pix_fmt yuv420p", ".mp4");

    asrv1::TranscribeOptions options = options_for("tiny.en");
    options.set_emit_keyframes(true);
    options.set_keyframe_interval_seconds(2);
    StreamResult result = transcribe(channel, options, media);
    require(result.status.ok(), "video transcribes OK: " + result.status.error_message());
    require(result.got_media_info && result.media_info.has_video(), "MediaInfo reports video");
    require(lower(final_text(result)).find("country") != std::string::npos,
            "video audio track transcribed");
    require(result.keyframes >= 4 && result.keyframes <= 7,
            "keyframes match the 2s interval over 11s, got " +
                std::to_string(result.keyframes));
    require(result.complete.keyframe_count() == result.keyframes,
            "trailer counts the keyframes");
    for (const asrv1::Keyframe& frame : result.keyframe_samples) {
        require(frame.width() == 320 && frame.height() == 240, "keyframe dimensions");
        require(frame.png().starts_with("\x89PNG"), "keyframe is a PNG");
    }

    // Same media without opting in: zero keyframes.
    StreamResult quiet = transcribe(channel, options_for("tiny.en"), media);
    require(quiet.status.ok(), "video without keyframes OK");
    require(quiet.keyframes == 0 && quiet.complete.keyframe_count() == 0,
            "no keyframes without opt-in");

    std::string video_only = generate_media(
        "-f lavfi -i testsrc2=duration=3:size=320x240:rate=10 -an -pix_fmt yuv420p", ".mp4");
    StreamResult no_audio = transcribe(channel, options_for("tiny.en"), video_only);
    require(no_audio.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
            "video without an audio track is INVALID_ARGUMENT");
}

void verify_service_info(const std::shared_ptr<grpc::Channel>& channel) {
    auto stub = asrv1::AsrService::NewStub(channel);
    grpc::ClientContext context;
    asrv1::GetServiceInfoRequest request;
    asrv1::GetServiceInfoResponse info;
    require(stub->GetServiceInfo(&context, request, &info).ok(), "GetServiceInfo OK");
    require(!info.version().empty(), "version reported");
    require(!info.whisper_version().empty(), "whisper version reported");
    require(info.backend() == "cpu", "backend reported");
    require(info.models_size() == 1 && info.models(0) == "tiny.en", "models reported");
    require(info.max_media_bytes() > 0 && info.max_duration_ms() > 0, "caps reported");
    require(info.ui().title() == "ASR" && info.ui().path() == "/ui/asr" &&
                !info.ui().description().empty(),
            "ui advertisement reported for the demo shell");
}

void verify_startup_fails_loud(const std::string& models_dir) {
    const char* binary = env_or_null("GRPC_ASR_SERVER_BINARY");
    require(binary != nullptr, "GRPC_ASR_SERVER_BINARY set by ctest");

    // A missing model must stop the process at startup, whatever the
    // backend.
    std::string command = std::string("timeout 30 ") + binary +
                          " >/dev/null 2>&1";
    std::string env = "GRPC_ASR_BACKEND=cpu GRPC_ASR_MODELS_DIR=" + models_dir +
                      " GRPC_ASR_MODELS=no-such-model ";
    int code = std::system((env + command).c_str());
    require(WIFEXITED(code) && WEXITSTATUS(code) == 1,
            "missing model exits 1 at startup, got " + std::to_string(WEXITSTATUS(code)));

    // When this build has no CUDA backend, asking for it must also die at
    // startup, before any RPC.
    if (ggml_backend_reg_by_name("CUDA") == nullptr) {
        env = "GRPC_ASR_BACKEND=cuda GRPC_ASR_MODELS_DIR=" + models_dir +
              " GRPC_ASR_MODELS=tiny.en ";
        code = std::system((env + command).c_str());
        require(WIFEXITED(code) && WEXITSTATUS(code) == 1,
                "unavailable backend exits 1 at startup");
    } else {
        std::println("note: CUDA backend present in this build; skipping the "
                     "missing-backend sub-case");
    }
}

}  // namespace

int main() {
    const char* model = env_or_null("GRPC_ASR_TEST_MODEL");
    const char* sample = env_or_null("GRPC_ASR_TEST_SAMPLE");
    if (model == nullptr || slurp(model).empty()) {
        return skip("no model weights at $GRPC_ASR_TEST_MODEL");
    }
    std::string jfk = sample == nullptr ? "" : slurp(sample);
    if (jfk.empty()) {
        return skip("no fixture wav at $GRPC_ASR_TEST_SAMPLE");
    }

    asr::Config config;
    config.backend = "cpu";
    config.models_dir = std::filesystem::path(model).parent_path().string();
    config.models = {"tiny.en"};
    config.concurrency = 1;
    // A window smaller than the fixture makes the during-upload assertion
    // possible.
    config.window_seconds = 10;
    config.threads = 4;

    try {
        asr::engine::ModelPool pool(config);
        asr::AsrServiceImpl service(config, pool);

        int port = 0;
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service);
        auto server = builder.BuildAndStart();
        require(server != nullptr, "server started");
        auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                                           grpc::InsecureChannelCredentials());

        verify_wav(channel, jfk);
        verify_streaming_during_upload(channel, jfk);
        verify_document(channel, jfk);
        verify_silence(channel);
        verify_error_matrix(channel);
        verify_byte_cap(config, pool);
        verify_service_info(channel);
        if (have_ffmpeg()) {
            verify_video(channel, sample);
        } else {
            std::println("note: ffmpeg not on PATH; video cases not run");
        }
        verify_startup_fails_loud(config.models_dir);

        require(service.transcribed.load() > 0, "transcribed counter moved");
        require(service.rejected.load() > 0, "rejected counter moved");
        server->Shutdown();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("asr-service-test passed");
    return 0;
}
