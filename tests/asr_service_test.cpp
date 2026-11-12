// End-to-end over gRPC loopback: upload → MediaInfo → partial/final
// segments → keyframes → TranscriptComplete, plus the full error-code
// matrix and the wire-level anti-batch proof (a final segment arrives
// while the client still holds the last media chunk).
//
// Needs GRPC_ASR_TEST_MODEL and GRPC_ASR_TEST_SAMPLE; skips with 77
// otherwise. Video cases additionally need ffmpeg/ffprobe on PATH.

#include "service/asr_service.h"

#include <arpa/inet.h>
#include <grpcpp/grpcpp.h>
#include <netinet/in.h>
#include <csignal>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <print>
#include <string_view>
#include <thread>
#include <vector>

#include "config.h"
#include "document/document_fold.h"
#include "engine/model_pool.h"
#include "fixture.h"
#include "ggml-backend.h"
#include "media/sniff.h"
#include "metrics.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/util/message_differencer.h>

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

// Uploads the whole media, reads events until cancel_on says the server is
// mid-transcription, waits settle, and cancels the call from the client
// side. Returns the status the client saw.
grpc::Status transcribe_and_cancel(
    const std::shared_ptr<grpc::Channel>& channel, const asrv1::TranscribeOptions& options,
    const std::string& media,
    const std::function<bool(const asrv1::TranscribeResponse&)>& cancel_on,
    std::chrono::milliseconds settle = std::chrono::milliseconds(0)) {
    auto stub = asrv1::AsrService::NewStub(channel);
    grpc::ClientContext context;
    auto stream = stub->Transcribe(&context);
    asrv1::TranscribeRequest request;
    *request.mutable_options() = options;
    stream->Write(request);
    for (size_t offset = 0; offset < media.size(); offset += kChunk) {
        request.Clear();
        request.mutable_chunk()->set_data(
            media.substr(offset, std::min(kChunk, media.size() - offset)));
        if (!stream->Write(request)) {
            break;
        }
    }
    stream->WritesDone();
    asrv1::TranscribeResponse response;
    while (stream->Read(&response)) {
        if (cancel_on(response)) {
            break;
        }
    }
    std::this_thread::sleep_for(settle);
    context.TryCancel();
    while (stream->Read(&response)) {
    }
    return stream->Finish();
}

// Streams of this service whose handler has returned, whatever the
// status: it moves by one per finished stream.
long handlers_finished(const asr::AsrServiceImpl& service) {
    return service.transcribed.load() + service.rejected.load() + service.failed.load();
}

// Polls until condition holds; false when limit passed first.
bool eventually(const std::function<bool()>& condition, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!condition()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return true;
}

// Uploads the whole media under a client deadline and reads to the end.
grpc::Status transcribe_with_deadline(const std::shared_ptr<grpc::Channel>& channel,
                                      const asrv1::TranscribeOptions& options,
                                      const std::string& media,
                                      std::chrono::milliseconds deadline) {
    auto stub = asrv1::AsrService::NewStub(channel);
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + deadline);
    auto stream = stub->Transcribe(&context);
    asrv1::TranscribeRequest request;
    *request.mutable_options() = options;
    stream->Write(request);
    for (size_t offset = 0; offset < media.size(); offset += kChunk) {
        request.Clear();
        request.mutable_chunk()->set_data(
            media.substr(offset, std::min(kChunk, media.size() - offset)));
        if (!stream->Write(request)) {
            break;
        }
    }
    stream->WritesDone();
    asrv1::TranscribeResponse response;
    while (stream->Read(&response)) {
    }
    return stream->Finish();
}

// Keeps the pool's only state (the suite runs with concurrency 1) busy
// with a long speech-free transcription on its own thread until cancel().
class StateHolder {
  public:
    StateHolder(const std::shared_ptr<grpc::Channel>& channel, const std::string& hiss)
        : thread_([this, channel, &hiss] {
              auto stub = asrv1::AsrService::NewStub(channel);
              auto stream = stub->Transcribe(&context_);
              asrv1::TranscribeRequest request;
              request.mutable_options()->set_model("tiny.en");
              stream->Write(request);
              for (size_t offset = 0; offset < hiss.size(); offset += kChunk) {
                  request.Clear();
                  request.mutable_chunk()->set_data(
                      hiss.substr(offset, std::min(kChunk, hiss.size() - offset)));
                  if (!stream->Write(request)) {
                      break;
                  }
              }
              stream->WritesDone();
              asrv1::TranscribeResponse response;
              while (stream->Read(&response)) {
                  if (response.has_media_info()) {
                      started_ = true;
                  }
              }
              status_ = stream->Finish();
          }) {
        require(eventually([&] { return started_.load(); }, std::chrono::seconds(30)),
                "the holding stream started");
        // MediaInfo goes out just before the state is taken.
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    ~StateHolder() {
        if (thread_.joinable()) {
            cancel();
        }
    }

    grpc::Status cancel() {
        context_.TryCancel();
        thread_.join();
        return status_;
    }

  private:
    grpc::ClientContext context_;
    std::atomic<bool> started_{false};
    grpc::Status status_;
    std::thread thread_;
};

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
    const std::string generator = fixture_ffmpeg() + " -version >/dev/null 2>&1";
    return std::system("ffmpeg -version >/dev/null 2>&1") == 0 &&
           std::system("ffprobe -version >/dev/null 2>&1") == 0 &&
           std::system(generator.c_str()) == 0;
}

std::string generate_media(const std::string& args, const std::string& suffix) {
    std::string path = "/tmp/grpc-asr-e2e-XXXXXX" + suffix;
    int fd = mkstemps(path.data(), static_cast<int>(suffix.size()));
    require(fd >= 0, "temp fixture path");
    ::close(fd);
    std::string command =
        fixture_ffmpeg() + " -v error -y " + args + " " + path + " >/dev/null 2>&1";
    require(std::system(command.c_str()) == 0, "fixture generation: " + command);
    std::string bytes = slurp(path);
    std::remove(path.c_str());
    require(!bytes.empty(), "fixture has bytes");
    return bytes;
}

std::string http_get_metrics(uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    require(fd >= 0, "metrics client socket");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    require(::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "inet_pton 127.0.0.1");
    require(::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0,
            "connect to metrics HTTP server");
    const std::string req = "GET /metrics HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    size_t sent = 0;
    while (sent < req.size()) {
        const ssize_t n = ::send(fd, req.data() + sent, req.size() - sent, MSG_NOSIGNAL);
        require(n > 0, "send GET /metrics");
        sent += static_cast<size_t>(n);
    }
    std::string raw;
    char buf[4096];
    while (true) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            break;
        }
        raw.append(buf, static_cast<size_t>(n));
    }
    ::close(fd);
    require(raw.starts_with("HTTP/1.1 200 OK\r\n"), "GET /metrics returned 200 OK");
    const size_t sep = raw.find("\r\n\r\n");
    require(sep != std::string::npos, "HTTP header delimiter present");
    return raw.substr(sep + 4);
}

std::optional<double> scrape_metric_value(const std::string& body, std::string_view series_name) {
    const std::string needle = std::string(series_name) + " ";
    size_t pos = 0;
    while (pos < body.size()) {
        const size_t line_end = body.find('\n', pos);
        const std::string_view line =
            line_end == std::string::npos
                ? std::string_view(body).substr(pos)
                : std::string_view(body).substr(pos, line_end - pos);
        if (line.starts_with(needle)) {
            const std::string_view tail = line.substr(needle.size());
            double value = 0.0;
            const auto [ptr, ec] =
                std::from_chars(tail.data(), tail.data() + tail.size(), value);
            if (ec == std::errc{}) {
                return value;
            }
        }
        if (line_end == std::string::npos) {
            break;
        }
        pos = line_end + 1;
    }
    return std::nullopt;
}

// Extracts raw 16-bit mono PCM bytes from a RIFF WAVE container.
std::string extract_wav_pcm16_data(const std::string& wav) {
    require(wav.size() >= 44 && wav.compare(0, 4, "RIFF") == 0 && wav.compare(8, 4, "WAVE") == 0,
            "fixture is RIFF WAVE");
    auto read_u32 = [&](size_t off) -> uint32_t {
        return static_cast<uint32_t>(static_cast<uint8_t>(wav[off])) |
               (static_cast<uint32_t>(static_cast<uint8_t>(wav[off + 1])) << 8) |
               (static_cast<uint32_t>(static_cast<uint8_t>(wav[off + 2])) << 16) |
               (static_cast<uint32_t>(static_cast<uint8_t>(wav[off + 3])) << 24);
    };
    size_t off = 12;
    while (off + 8 <= wav.size()) {
        const std::string_view tag(wav.data() + off, 4);
        const uint32_t sz = read_u32(off + 4);
        off += 8;
        if (tag == "data") {
            const size_t avail = std::min<size_t>(sz, wav.size() - off);
            return wav.substr(off, avail & ~size_t{1});
        }
        off += sz + (sz & 1U);
    }
    throw std::runtime_error("FAIL: no data chunk in WAV");
}

std::string wrap_wav_pcm16(const std::string& pcm, uint16_t channels, uint32_t sample_rate = 16000) {
    const auto data_bytes = static_cast<uint32_t>(pcm.size());
    std::string out;
    out.reserve(44 + pcm.size());
    auto push_u32 = [&](uint32_t v) {
        for (int shift = 0; shift < 32; shift += 8) {
            out.push_back(static_cast<char>((v >> shift) & 0xFF));
        }
    };
    auto push_u16 = [&](uint16_t v) {
        out.push_back(static_cast<char>(v & 0xFF));
        out.push_back(static_cast<char>((v >> 8) & 0xFF));
    };
    out += "RIFF";
    push_u32(36 + data_bytes);
    out += "WAVEfmt ";
    push_u32(16);
    push_u16(1);
    push_u16(channels);
    push_u32(sample_rate);
    push_u32(sample_rate * channels * 2U);
    push_u16(static_cast<uint16_t>(channels * 2U));
    push_u16(16);
    out += "data";
    push_u32(data_bytes);
    out += pcm;
    return out;
}

// Builds a stereo 16 kHz WAV with mono `jfk` speech on channel 0 and digital
// silence on channel 1.
std::string make_stereo_jfk_ch0_wav(const std::string& jfk) {
    const std::string mono = extract_wav_pcm16_data(jfk);
    std::string stereo;
    stereo.reserve(mono.size() * 2);
    for (size_t i = 0; i + 1 < mono.size(); i += 2) {
        stereo.push_back(mono[i]);
        stereo.push_back(mono[i + 1]);
        stereo.push_back('\0');
        stereo.push_back('\0');
    }
    return wrap_wav_pcm16(stereo, /*channels=*/2, 16000);
}

// Builds a stereo 16 kHz WAV with mono `jfk` speech on both channels so
// concurrent PER_CHANNEL worker threads emit overlapping spoken segments.
std::string make_stereo_jfk_both_wav(const std::string& jfk) {
    const std::string mono = extract_wav_pcm16_data(jfk);
    std::string stereo;
    stereo.reserve(mono.size() * 2);
    for (size_t i = 0; i + 1 < mono.size(); i += 2) {
        stereo.push_back(mono[i]);
        stereo.push_back(mono[i + 1]);
        stereo.push_back(mono[i]);
        stereo.push_back(mono[i + 1]);
    }
    return wrap_wav_pcm16(stereo, /*channels=*/2, 16000);
}

std::string concat_jfk_twice_wav(const std::string& jfk) {
    const std::string mono = extract_wav_pcm16_data(jfk);
    return wrap_wav_pcm16(mono + mono, /*channels=*/1, 16000);
}

uint16_t pick_free_port() {
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    require(fd >= 0, "ephemeral port socket");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    require(::bind(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0, "bind :0");
    socklen_t len = sizeof(addr);
    require(::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0, "getsockname :0");
    const uint16_t port = ntohs(addr.sin_port);
    ::close(fd);
    return port;
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

void verify_trailer_to_gone_client(const std::string& target,
                                   const asr::AsrServiceImpl& service) {
    // A client that leaves after the last segment fails only the trailer
    // write, and never sees TranscriptComplete: the stream must end
    // CANCELLED, not OK, and not count as transcribed. The Document event
    // goes out with the trailer and carries the filename, so a 1 MiB name
    // makes it far larger than the window a client that stops reading
    // grants with BDP probing off. The server then sits in that write,
    // past every earlier check for a gone call, until the cancel fails it.
    grpc::ChannelArguments arguments;
    arguments.SetInt(GRPC_ARG_HTTP2_BDP_PROBE, 0);
    auto channel =
        grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), arguments);
    asrv1::TranscribeOptions options = options_for("tiny.en");
    options.set_emit_document(true);
    options.set_filename(std::string(1 << 20, 'a') + ".wav");
    // No samples, so no whisper pass: the stream reaches its trailer at
    // once, whatever the weights.
    const std::string media = make_wav(0.0, 0.0);

    const long transcribed = service.transcribed.load();
    const long finished = handlers_finished(service);
    auto stub = asrv1::AsrService::NewStub(channel);
    grpc::ClientContext context;
    auto stream = stub->Transcribe(&context);
    asrv1::TranscribeRequest request;
    *request.mutable_options() = options;
    stream->Write(request);
    request.Clear();
    request.mutable_chunk()->set_data(media);
    stream->Write(request);
    stream->WritesDone();
    asrv1::TranscribeResponse response;
    while (stream->Read(&response) && !response.has_media_info()) {
    }
    // Long enough for the stream to park in the Document write.
    std::this_thread::sleep_for(std::chrono::seconds(1));
    require(handlers_finished(service) == finished,
            "the server waits on the client that stopped reading");
    context.TryCancel();
    while (stream->Read(&response)) {
    }
    stream->Finish();
    require(eventually([&] { return handlers_finished(service) > finished; },
                       std::chrono::seconds(10)),
            "the stream ends once its client is gone");
    require(service.transcribed.load() == transcribed,
            "a stream whose trailer never reached the client is not transcribed");
}

void verify_cancel_without_speech(const std::shared_ptr<grpc::Channel>& channel,
                                  const asr::AsrServiceImpl& service, const std::string& hiss,
                                  const std::string& jfk) {
    // A speech-free stretch commits no segment, so nothing is written and
    // a gone client never showed up as a failed write: whisper went on
    // decoding it for nobody, holding the only model state.
    const long finished = handlers_finished(service);
    grpc::Status status = transcribe_and_cancel(
        channel, options_for("tiny.en"), hiss,
        [](const asrv1::TranscribeResponse& response) { return response.has_media_info(); },
        std::chrono::milliseconds(500));
    require(status.error_code() == grpc::StatusCode::CANCELLED,
            "the client sees its own cancel, got " + status.error_message());
    require(eventually([&] { return handlers_finished(service) > finished; },
                       std::chrono::seconds(10)),
            "a cancel during a speech-free stretch stops the stream within seconds");
    StreamResult next = transcribe(channel, options_for("tiny.en"), jfk);
    require(next.status.ok() && lower(final_text(next)).find("country") != std::string::npos,
            "the state serves the next stream after that cancel: " +
                next.status.error_message());
}

void verify_deadline_without_speech(const std::shared_ptr<grpc::Channel>& channel,
                                    const asr::AsrServiceImpl& service, const std::string& hiss) {
    // The same for a client deadline, which is how gRParse bounds an ASR
    // call: once it passes the server stops too, rather than finishing a
    // transcription nobody waits for.
    const long finished = handlers_finished(service);
    grpc::Status status = transcribe_with_deadline(channel, options_for("tiny.en"), hiss,
                                                   std::chrono::seconds(4));
    require(status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED,
            "the client sees its deadline pass, got " + status.error_message());
    require(eventually([&] { return handlers_finished(service) > finished; },
                       std::chrono::seconds(10)),
            "the server stops the stream within seconds of its deadline");
}

void verify_cancel_while_queued(const std::shared_ptr<grpc::Channel>& channel,
                                const asr::AsrServiceImpl& service, const std::string& hiss,
                                const std::string& jfk) {
    StateHolder holder(channel, hiss);
    // This stream queues for the only state; its MediaInfo goes out before
    // it starts waiting. Cancelled there, it must leave the queue at once
    // rather than hold its RPC thread and upload until the holder is done.
    const long finished = handlers_finished(service);
    grpc::Status status = transcribe_and_cancel(
        channel, options_for("tiny.en"), jfk,
        [](const asrv1::TranscribeResponse& response) { return response.has_media_info(); },
        std::chrono::milliseconds(300));
    require(status.error_code() == grpc::StatusCode::CANCELLED,
            "the queued client sees its own cancel, got " + status.error_message());
    require(eventually([&] { return handlers_finished(service) > finished; },
                       std::chrono::seconds(10)),
            "a cancelled stream leaves the queue for a model state at once");
    require(handlers_finished(service) == finished + 1,
            "while the holder still has the state");
    holder.cancel();
    require(eventually([&] { return handlers_finished(service) > finished + 1; },
                       std::chrono::seconds(10)),
            "the holder stops on its own cancel");
}

void verify_queue_timeout(asr::Config config, asr::engine::ModelPool& pool,
                          const std::shared_ptr<grpc::Channel>& channel,
                          const std::string& hiss, const std::string& jfk,
                          uint16_t metrics_port) {
    // A second service over the same pool whose streams wait at most two
    // seconds for a state, long enough to observe grpc_asr_queue_depth == 1
    // on /metrics while queued.
    config.queue_timeout_seconds = 2;
    asr::AsrServiceImpl impatient(config, pool);
    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&impatient);
    auto server = builder.BuildAndStart();
    require(server != nullptr, "queue timeout server started");
    auto impatient_channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                                                 grpc::InsecureChannelCredentials());

    StateHolder holder(channel, hiss);
    const auto started = std::chrono::steady_clock::now();
    StreamResult result;
    std::thread waiter([&] { result = transcribe(impatient_channel, options_for("tiny.en"), jfk); });
    require(eventually(
                [&] {
                    const auto v = scrape_metric_value(http_get_metrics(metrics_port),
                                                       "grpc_asr_queue_depth{model=\"tiny.en\"}");
                    return v.has_value() && *v >= 1.0;
                },
                std::chrono::milliseconds(1800)),
            "grpc_asr_queue_depth reached 1 while stream waited");
    waiter.join();
    const auto waited = std::chrono::steady_clock::now() - started;
    require(result.status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED,
            "a stream that waited out the queue timeout is RESOURCE_EXHAUSTED, got " +
                result.status.error_message());
    require(result.status.error_message().find("GRPC_ASR_QUEUE_TIMEOUT_SECONDS") !=
                std::string::npos,
            "the error names the knob");
    require(waited >= std::chrono::seconds(2) && waited < std::chrono::seconds(10),
            "the wait lasted about the queue timeout");
    holder.cancel();
    server->Shutdown();
}

void verify_error_matrix(const std::shared_ptr<grpc::Channel>& channel, asr::Config base_config,
                         asr::engine::ModelPool& pool) {
    StreamResult result =
        transcribe(channel, options_for("tiny.en"), make_truncated_mp3());
    require(result.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
            "truncated mp3 is INVALID_ARGUMENT, got " + result.status.error_message());

    result = transcribe(channel, options_for("tiny.en"), "plain text, definitely not media");
    require(result.status.error_code() == grpc::StatusCode::UNIMPLEMENTED,
            "unknown container is UNIMPLEMENTED");

    if (have_ffmpeg()) {
        asrv1::TranscribeOptions ext_fallback = options_for("tiny.en");
        ext_fallback.set_filename("broken.wma");
        result = transcribe(channel, ext_fallback, "plain text, definitely not media");
        require(result.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
                "routed extension fallback routes corrupt .wma to ffmpeg INVALID_ARGUMENT, got " +
                    result.status.error_message());
    }

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

    // Local DecodingOptions validation (#14): beam_size > 8 and prompt > 1024 B
    // are rejected before any media is decoded.
    asrv1::TranscribeOptions bad_beam = options_for("tiny.en");
    bad_beam.mutable_decoding()->set_beam_size(64);
    result = transcribe(channel, bad_beam, make_wav(1.0, 440.0));
    require(result.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT &&
                result.finals.empty() && !result.got_media_info,
            "beam_size=64 is INVALID_ARGUMENT before decoding, got " +
                result.status.error_message());

    asrv1::TranscribeOptions long_prompt = options_for("tiny.en");
    long_prompt.mutable_decoding()->set_initial_prompt(std::string(1025, 'p'));
    result = transcribe(channel, long_prompt, make_wav(1.0, 440.0));
    require(result.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT &&
                result.finals.empty() && !result.got_media_info,
            "over-long initial_prompt is INVALID_ARGUMENT before decoding, got " +
                result.status.error_message());

    // Multi-channel cap (#15): 3 channels under default GRPC_ASR_MAX_CHANNELS=2
    // fails INVALID_ARGUMENT before MediaInfo or PCM decoding.
    asrv1::TranscribeOptions three_ch = options_for("tiny.en");
    three_ch.set_channel_mode(asrv1::CHANNEL_MODE_PER_CHANNEL);
    result = transcribe(channel, three_ch, make_multichannel_wav(1.0, {440.0, 550.0, 660.0}));
    require(result.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT &&
                result.status.error_message().find("GRPC_ASR_MAX_CHANNELS=2") !=
                    std::string::npos &&
                result.finals.empty() && !result.got_media_info,
            "3-channel WAV under PER_CHANNEL is INVALID_ARGUMENT before decoding, got " +
                result.status.error_message());

    // Remote options (#19): malformed URL is INVALID_ARGUMENT; valid URL with no
    // allowlist is FAILED_PRECONDITION; unlisted host or loopback without an
    // explicit loopback pattern is PERMISSION_DENIED.
    asrv1::TranscribeOptions remote = options_for("");
    remote.mutable_remote()->set_url("file:///etc/passwd");
    result = transcribe(channel, remote, make_wav(1.0, 440.0));
    require(result.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT &&
                result.status.error_message().find("RemoteTranscriptionOptions.url") !=
                    std::string::npos,
            "a file:// remote url is INVALID_ARGUMENT, got " + result.status.error_message());
    remote.mutable_remote()->set_url("http://localhost:8000/v1/audio/transcriptions");
    result = transcribe(channel, remote, make_wav(1.0, 440.0));
    require(result.status.error_code() == grpc::StatusCode::FAILED_PRECONDITION &&
                result.status.error_message().find("GRPC_ASR_REMOTE_ALLOWED_HOSTS") !=
                    std::string::npos,
            "remote with no allowlist is FAILED_PRECONDITION, got " +
                result.status.error_message());
    require(result.finals.empty() && !result.got_media_info,
            "disabled remote options never run the local pool");

    {
        asr::Config strict_cfg = base_config;
        strict_cfg.remote_allowed_hosts = {"allowed.example:8443"};
        asr::AsrServiceImpl strict_svc(strict_cfg, pool);
        int strict_port = 0;
        grpc::ServerBuilder sb;
        sb.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &strict_port);
        sb.RegisterService(&strict_svc);
        auto strict_srv = sb.BuildAndStart();
        require(strict_srv != nullptr, "strict remote allowlist server started");
        auto strict_ch = grpc::CreateChannel("127.0.0.1:" + std::to_string(strict_port),
                                             grpc::InsecureChannelCredentials());

        remote.mutable_remote()->set_url("https://other.example:8443/v1/audio/transcriptions");
        result = transcribe(strict_ch, remote, make_wav(1.0, 440.0));
        require(result.status.error_code() == grpc::StatusCode::PERMISSION_DENIED,
                "unlisted remote host is PERMISSION_DENIED, got " +
                    result.status.error_message());
        strict_srv->Shutdown();
    }

    {
        asr::Config wildcard_cfg = base_config;
        wildcard_cfg.remote_allowed_hosts = {"*"};
        asr::AsrServiceImpl wildcard_svc(wildcard_cfg, pool);
        int wildcard_port = 0;
        grpc::ServerBuilder wb;
        wb.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &wildcard_port);
        wb.RegisterService(&wildcard_svc);
        auto wildcard_srv = wb.BuildAndStart();
        require(wildcard_srv != nullptr, "wildcard remote allowlist server started");
        auto wildcard_ch = grpc::CreateChannel("127.0.0.1:" + std::to_string(wildcard_port),
                                               grpc::InsecureChannelCredentials());

        auto stub = asrv1::AsrService::NewStub(wildcard_ch);
        grpc::ClientContext info_ctx;
        asrv1::GetServiceInfoRequest info_req;
        asrv1::GetServiceInfoResponse info_resp;
        require(stub->GetServiceInfo(&info_ctx, info_req, &info_resp).ok() &&
                    info_resp.remote_enabled(),
                "GetServiceInfo reports remote_enabled=true when allowlist is set");

        remote.mutable_remote()->set_url("http://127.0.0.1:8000/v1/audio/transcriptions");
        result = transcribe(wildcard_ch, remote, make_wav(1.0, 440.0));
        require(result.status.error_code() == grpc::StatusCode::PERMISSION_DENIED,
                "loopback remote URL under wildcard without explicit loopback pattern is "
                "PERMISSION_DENIED, got " +
                    result.status.error_message());
        wildcard_srv->Shutdown();
    }

    // Docling-slim engine / model_type: unsupported arms are UNIMPLEMENTED
    // before any media is decoded; a whisper.cpp-servable preset fills model.
    asrv1::TranscribeOptions conflicting = options_for("tiny.en");
    conflicting.set_engine(asrv1::ASR_ENGINE_KIND_WHISPER_CPP);
    conflicting.mutable_remote()->set_url("https://example.invalid/transcriptions");
    result = transcribe(channel, conflicting, make_wav(1.0, 440.0));
    require(result.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT &&
                result.status.error_message().find("engine") != std::string::npos &&
                result.finals.empty() && !result.got_media_info,
            "conflicting remote engine rejected before decoding");

    asrv1::TranscribeOptions transformers = options_for("tiny.en");
    transformers.set_engine(asrv1::ASR_ENGINE_KIND_TRANSFORMERS);
    result = transcribe(channel, transformers, make_wav(1.0, 440.0));
    require(result.status.error_code() == grpc::StatusCode::UNIMPLEMENTED &&
                result.status.error_message().find("TRANSFORMERS") != std::string::npos,
            "TRANSFORMERS engine is UNIMPLEMENTED, got " + result.status.error_message());

    asrv1::TranscribeOptions mlx = options_for("");
    mlx.set_model_type(asrv1::ASR_MODEL_TYPE_WHISPER_TINY_MLX);
    result = transcribe(channel, mlx, make_wav(1.0, 440.0));
    require(result.status.error_code() == grpc::StatusCode::UNIMPLEMENTED &&
                result.status.error_message().find("model_type") != std::string::npos,
            "unmapped model_type is UNIMPLEMENTED, got " + result.status.error_message());

    asrv1::TranscribeOptions preset = options_for("");
    preset.set_model_type(asrv1::ASR_MODEL_TYPE_WHISPER_TINY);
    // This fixture loads tiny.en only; resolving the preset must name missing tiny.
    result = transcribe(channel, preset, make_wav(1.0, 440.0));
    require(result.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT &&
                result.status.error_message().find("model 'tiny' is not loaded") != std::string::npos,
            "whisper_tiny preset resolves to model tiny, got " + result.status.error_message());

    asrv1::TranscribeOptions bad_task_raw = options_for("tiny.en");
    bad_task_raw.set_task_raw("summarize");
    result = transcribe(channel, bad_task_raw, make_wav(1.0, 440.0));
    require(result.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT &&
                result.status.error_message().find("task_raw") != std::string::npos &&
                result.finals.empty() && !result.got_media_info,
            "unknown task_raw is INVALID_ARGUMENT before decoding, got " +
                result.status.error_message());

    asrv1::TranscribeOptions conflicting_task = options_for("tiny.en");
    conflicting_task.set_task(asrv1::TASK_TRANSCRIBE);
    conflicting_task.set_task_raw("translate");
    result = transcribe(channel, conflicting_task, make_wav(1.0, 440.0));
    require(result.status.error_code() == grpc::StatusCode::INVALID_ARGUMENT &&
                result.status.error_message().find("conflicts") != std::string::npos &&
                result.finals.empty() && !result.got_media_info,
            "conflicting task and task_raw is INVALID_ARGUMENT before decoding, got " +
                result.status.error_message());
}

void verify_window_stall_and_clamping(asr::Config config, asr::engine::ModelPool& pool,
                                      const std::string& jfk) {
    config.window_seconds = 60;
    config.min_window_seconds = 10;
    config.window_stall_ms = 250;
    asr::AsrServiceImpl service(config, pool);
    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    require(server != nullptr, "window clamping server started");
    auto ch =
        grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials());

    // Client value above server max (60) is clamped, not rejected, and reported
    // on TranscriptComplete.window_seconds.
    asrv1::TranscribeOptions over = options_for("tiny.en");
    over.set_window_seconds(9999);
    StreamResult over_res = transcribe(ch, over, jfk);
    require(over_res.status.ok(), "over-max window_seconds clamped OK");
    require(over_res.complete.window_seconds() == 60,
            "TranscriptComplete reports clamped max window_seconds=60, got " +
                std::to_string(over_res.complete.window_seconds()));

    asrv1::TranscribeOptions under = options_for("tiny.en");
    under.set_window_seconds(2);
    StreamResult under_res = transcribe(ch, under, jfk);
    require(under_res.status.ok(), "under-min window_seconds clamped OK");
    require(under_res.complete.window_seconds() == 10,
            "TranscriptComplete reports clamped min window_seconds=10, got " +
                std::to_string(under_res.complete.window_seconds()));

    // Real-time paced upload (#13): stream ~22 s of 16 kHz mono WAV in 1 s
    // chunks with client window_seconds=10 and assert the first FinalSegment
    // arrives before 20 s of media has been uploaded.
    const std::string wav22 = concat_jfk_twice_wav(jfk);
    auto stub = asrv1::AsrService::NewStub(ch);
    grpc::ClientContext ctx;
    auto stream = stub->Transcribe(&ctx);

    asrv1::TranscribeOptions paced_opts = options_for("tiny.en");
    paced_opts.set_window_seconds(10);
    asrv1::TranscribeRequest req;
    *req.mutable_options() = paced_opts;
    require(stream->Write(req), "write paced options");

    std::atomic<bool> saw_first_final{false};
    StreamResult paced_res;
    std::thread reader([&] {
        asrv1::TranscribeResponse resp;
        size_t seen = 0;
        while (stream->Read(&resp)) {
            if (resp.has_final_segment()) {
                saw_first_final.store(true, std::memory_order_release);
            }
            consume_event(paced_res, resp, /*upload_done=*/true, seen++);
        }
    });

    // Send 44-byte WAV header, then 1 s chunks (32000 B at 16 kHz s16le) at
    // real-time 1 s/chunk pace until the first final segment arrives, then
    // flush the remaining bytes cleanly.
    constexpr size_t kOneSecondPcmBytes = 32000;
    req.Clear();
    req.mutable_chunk()->set_data(wav22.substr(0, 44));
    require(stream->Write(req), "write wav header");

    double uploaded_seconds_at_first_final = 999.0;
    size_t offset = 44;
    size_t seconds_sent = 0;
    while (offset < wav22.size()) {
        const size_t n = std::min(kOneSecondPcmBytes, wav22.size() - offset);
        req.Clear();
        req.mutable_chunk()->set_data(wav22.substr(offset, n));
        if (!stream->Write(req)) {
            break;
        }
        offset += n;
        seconds_sent++;

        if (saw_first_final.load(std::memory_order_acquire)) {
            if (uploaded_seconds_at_first_final > 900.0) {
                uploaded_seconds_at_first_final = static_cast<double>(seconds_sent);
            }
            continue;
        }
        // Real-time 1 s pacing in 25 ms slices so we notice the first final
        // immediately without unnecessary wall-clock delay.
        for (int slice = 0; slice < 40; ++slice) {
            if (saw_first_final.load(std::memory_order_acquire)) {
                uploaded_seconds_at_first_final = static_cast<double>(seconds_sent);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    }
    stream->WritesDone();
    reader.join();
    paced_res.status = stream->Finish();

    require(paced_res.status.ok(), "paced 22s upload OK: " + paced_res.status.error_message());
    require(saw_first_final.load() && uploaded_seconds_at_first_final < 20.0,
            "first FinalSegment arrived before 20 s uploaded, got at " +
                std::to_string(uploaded_seconds_at_first_final) + " s");
    require(paced_res.complete.window_seconds() == 10, "trailer reports window_seconds=10");
    require(lower(final_text(paced_res)).find("country") != std::string::npos,
            "paced transcript contains 'country'");
    server->Shutdown();
}

void verify_per_channel_stereo(const std::shared_ptr<grpc::Channel>& channel,
                               asr::Config base_config, const std::string& jfk) {
    const std::string stereo = make_stereo_jfk_ch0_wav(jfk);
    asrv1::TranscribeOptions options = options_for("tiny.en");
    options.set_channel_mode(asrv1::CHANNEL_MODE_PER_CHANNEL);
    options.set_task_raw("transcribe");
    options.set_emit_document(true);

    StreamResult result = transcribe(channel, options, stereo);
    require(result.status.ok(),
            "stereo PER_CHANNEL transcribes OK: " + result.status.error_message());
    require(result.media_info.channels() == 2, "MediaInfo reports 2 channels");
    require(result.complete.segment_count_by_channel_size() == 2,
            "TranscriptComplete reports segment_count_by_channel for both channels");

    size_t spoken_ch0 = 0;
    size_t spoken_ch1 = 0;
    for (const asrv1::Segment& seg : result.finals) {
        require(seg.channel() < 2, "every final segment under PER_CHANNEL has valid channel");
        bool has_word_char = false;
        for (unsigned char c : seg.text()) {
            if (std::isalnum(c)) {
                has_word_char = true;
                break;
            }
        }
        const bool is_spoken = has_word_char && seg.text().find('[') == std::string::npos;
        if (is_spoken) {
            if (seg.channel() == 0) {
                spoken_ch0++;
                require(seg.speaker_id() == "C1",
                        "channel 0 speech is attributed to speaker C1, got " + seg.speaker_id());
            } else {
                spoken_ch1++;
            }
        }
    }
    require(spoken_ch0 > 0, "channel 0 produced spoken segments");
    require(spoken_ch1 == 0, "silent channel 1 produced zero spoken segments");
    require(lower(final_text(result)).find("country") != std::string::npos,
            "stereo ch0 transcript contains 'country'");

    const std::vector<std::string> errors = asr::doc::document_integrity_errors(result.document);
    require(errors.empty(), "stereo Document integrity clean");
    require(result.document.media().speakers_size() == 1 &&
                result.document.media().speakers(0) == "C1",
            "Document registers only C1 when channel 1 did not speak, got " +
                std::to_string(result.document.media().speakers_size()) + " speaker(s)");

    // Dual-spoken-channel concurrency=2 determinism: both channel worker
    // threads run concurrently on distinct ModelPool states and fold into a
    // byte-identical Document with sorted ["C1", "C2"] speakers.
    base_config.concurrency = 2;
    base_config.threads = 2;
    base_config.warmup = false;
    asr::engine::ModelPool dual_pool(base_config);
    asr::AsrServiceImpl dual_svc(base_config, dual_pool);
    int dual_port = 0;
    grpc::ServerBuilder sb;
    sb.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &dual_port);
    sb.RegisterService(&dual_svc);
    auto dual_srv = sb.BuildAndStart();
    require(dual_srv != nullptr, "dual-channel concurrency=2 server started");
    auto dual_ch = grpc::CreateChannel("127.0.0.1:" + std::to_string(dual_port),
                                       grpc::InsecureChannelCredentials());

    const std::string both_wav = make_stereo_jfk_both_wav(jfk);
    const StreamResult run1 = transcribe(dual_ch, options, both_wav);
    const StreamResult run2 = transcribe(dual_ch, options, both_wav);
    dual_srv->Shutdown();

    require(run1.status.ok() && run2.status.ok(),
            "dual-spoken stereo PER_CHANNEL runs completed OK");
    require(asr::doc::document_integrity_errors(run1.document).empty(),
            "dual-spoken stereo Document integrity clean");
    require(run1.document.media().speakers_size() == 2 &&
                run1.document.media().speakers(0) == "C1" &&
                run1.document.media().speakers(1) == "C2",
            "dual-spoken stereo Document registers canonical [C1, C2] speakers");
    auto deterministic_bytes = [](const google::protobuf::MessageLite& msg) {
        std::string out;
        {
            google::protobuf::io::StringOutputStream sink(&out);
            google::protobuf::io::CodedOutputStream coded(&sink);
            coded.SetSerializationDeterministic(true);
            require(msg.SerializeToCodedStream(&coded) && !coded.HadError(),
                    "deterministic protobuf serialization succeeded");
        }
        return out;
    };
    require(google::protobuf::util::MessageDifferencer::Equals(run1.document, run2.document),
            "concurrent PER_CHANNEL Document fold equals across runs");
    require(deterministic_bytes(run1.document) == deterministic_bytes(run2.document),
            "concurrent PER_CHANNEL Document fold is byte-identical across runs");
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
    options.set_emit_document(true);
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

    // #20: Document.body.children must be ordered by media time (prov[0].time.start_ms)
    // across interleaved keyframes and text segments, with integrity preserved.
    require(result.documents == 1, "video emitted folded Document");
    const std::vector<std::string> doc_errors =
        asr::doc::document_integrity_errors(result.document);
    require(doc_errors.empty(),
            "video Document integrity: " + (doc_errors.empty() ? "" : doc_errors.front()));
    double prev_start_ms = -1.0;
    size_t seen_texts = 0;
    size_t seen_pictures = 0;
    for (const auto& child : result.document.body().children()) {
        const std::string& ref = child.ref();
        double child_start_ms = 0.0;
        if (ref.starts_with("#/texts/")) {
            const int idx = std::stoi(ref.substr(8));
            child_start_ms = result.document.texts(idx).text().base().prov(0).time().start_ms();
            seen_texts++;
        } else if (ref.starts_with("#/pictures/")) {
            const int idx = std::stoi(ref.substr(11));
            child_start_ms = result.document.pictures(idx).prov(0).time().start_ms();
            seen_pictures++;
        } else {
            require(false, "unexpected body child ref: " + ref);
        }
        require(child_start_ms >= prev_start_ms,
                "video Document body children non-decreasing in start_ms: " +
                    std::to_string(child_start_ms) + " < " + std::to_string(prev_start_ms));
        prev_start_ms = child_start_ms;
    }
    require(seen_texts > 0 && seen_pictures == result.keyframes,
            "video Document body contains all text and picture items");

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

void verify_pipe_streaming_and_mp4_layouts(const std::shared_ptr<grpc::Channel>& channel,
                                           const std::string& jfk_path) {
    // #11: Loop the 11 s spoken fixture twice (~22 s > 10 s window) into ogg,
    // webm, and fast-start mp4; hold back the final 4 KiB until a FinalSegment
    // arrives, proving ffmpeg demuxes from pipe:0 while the upload is in flight.
    struct PipeCase {
        const char* label;
        std::string args;
        const char* suffix;
        asr::media::MediaFamily family;
        const char* expected_codec;
    };
    const std::vector<PipeCase> pipe_cases = {
        {"ogg", "-stream_loop 1 -i " + jfk_path + " -c:a libopus", ".ogg",
         asr::media::MediaFamily::kOgg, "opus"},
        {"webm", "-stream_loop 1 -i " + jfk_path + " -c:a libopus", ".webm",
         asr::media::MediaFamily::kMkv, "opus"},
        {"faststart-mp4", "-stream_loop 1 -i " + jfk_path + " -c:a aac -movflags +faststart",
         ".mp4", asr::media::MediaFamily::kMp4, "aac"},
    };
    for (const PipeCase& c : pipe_cases) {
        const std::string bytes = generate_media(c.args, c.suffix);
        const asr::media::StreamLayout layout = asr::media::sniff_stream_layout(
            c.family, reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(),
            /*stream_complete=*/true);
        require(layout == asr::media::StreamLayout::kPipeable,
                std::string(c.label) + " classified as kPipeable");

        asrv1::TranscribeOptions opts = options_for("tiny.en");
        opts.set_emit_document(true);
        StreamResult res = transcribe(channel, opts, bytes, /*hold_back=*/4096);
        require(res.final_before_upload_done,
                std::string(c.label) + " emitted a FinalSegment before upload finished");
        require(res.status.ok(),
                std::string(c.label) + " live pipe stream completed OK: " +
                    res.status.error_message());
        require(res.media_info.audio_codec() == c.expected_codec,
                std::string(c.label) + " MediaInfo.audio_codec expected " + c.expected_codec +
                    ", got '" + res.media_info.audio_codec() + "'");
        require(res.document.media().codec() == c.expected_codec,
                std::string(c.label) + " Document.media.codec expected " + c.expected_codec +
                    ", got '" + res.document.media().codec() + "'");
        require(lower(final_text(res)).find("country") != std::string::npos,
                std::string(c.label) + " transcript contains 'country'");
    }

    // Trailing-moov mp4 (default muxer order: mdat before moov) still
    // transcribes cleanly through the seekable memfd path and reports the
    // identical "aac" audio codec as fast-start mp4.
    const std::string trailing_mp4 = generate_media("-i " + jfk_path + " -c:a aac", ".mp4");
    const asr::media::StreamLayout trailing_layout = asr::media::sniff_stream_layout(
        asr::media::MediaFamily::kMp4, reinterpret_cast<const uint8_t*>(trailing_mp4.data()),
        trailing_mp4.size(), /*stream_complete=*/true);
    require(trailing_layout == asr::media::StreamLayout::kSeekableMemfd,
            "default mp4 fixture has trailing moov (kSeekableMemfd)");
    StreamResult trailing_res = transcribe(channel, options_for("tiny.en"), trailing_mp4);
    require(trailing_res.status.ok(),
            "trailing-moov mp4 transcribes OK: " + trailing_res.status.error_message());
    require(trailing_res.media_info.audio_codec() == "aac",
            "trailing-moov mp4 MediaInfo.audio_codec matches fast-start ('aac')");
    require(lower(final_text(trailing_res)).find("country") != std::string::npos,
            "trailing-moov mp4 transcript contains 'country'");
}

void verify_new_container_families(const std::shared_ptr<grpc::Channel>& channel,
                                   const std::string& jfk_path) {
    // #12: Author jfk in each newly supported container family and transcribe
    // end-to-end against the configured ffmpeg.
    struct ContainerCase {
        const char* label;
        std::string bytes;
        asrv1::ContainerFamily expected_container;
        const char* filename;
    };
    std::vector<ContainerCase> cases;
    cases.push_back({"avi-mp3", generate_media("-i " + jfk_path + " -c:a libmp3lame", ".avi"),
                     asrv1::CONTAINER_FAMILY_AVI, "clip.avi"});
    const std::string asf_wma = generate_media("-i " + jfk_path + " -c:a wmav2", ".asf");
    cases.push_back({"asf-wmav2", asf_wma, asrv1::CONTAINER_FAMILY_ASF, "clip.wma"});
    cases.push_back({"mpegps-mp2", generate_media("-i " + jfk_path + " -c:a mp2", ".mpg"),
                     asrv1::CONTAINER_FAMILY_MPEG_PS, "clip.mpg"});
    cases.push_back({"mpegts-aac", generate_media("-i " + jfk_path + " -c:a aac", ".ts"),
                     asrv1::CONTAINER_FAMILY_MPEG_TS, "clip.ts"});
    cases.push_back(
        {"adts-aac", generate_media("-i " + jfk_path + " -c:a aac -f adts", ".aac"),
         asrv1::CONTAINER_FAMILY_ADTS_AAC, "clip.aac"});

    // Resample jfk to 8 kHz mono s16le and encode AMR-NB in memory.
    const std::string raw_8k =
        generate_media("-i " + jfk_path + " -ac 1 -ar 8000 -f s16le", ".pcm");
    std::vector<int16_t> pcm_8k(raw_8k.size() / sizeof(int16_t));
    std::memcpy(pcm_8k.data(), raw_8k.data(), pcm_8k.size() * sizeof(int16_t));
    cases.push_back({"amr-nb", encode_amr_nb_8k(pcm_8k), asrv1::CONTAINER_FAMILY_AMR, "clip.amr"});

    for (const ContainerCase& c : cases) {
        asrv1::TranscribeOptions opts = options_for("tiny.en");
        opts.set_filename(c.filename);
        opts.set_emit_document(true);
        StreamResult res = transcribe(channel, opts, c.bytes);
        require(res.status.ok(),
                std::string(c.label) + " transcribes OK: " + res.status.error_message());
        require(res.media_info.container() == c.expected_container,
                std::string(c.label) + " reports expected MediaInfo.container");
        require(lower(final_text(res)).find("country") != std::string::npos,
                std::string(c.label) + " transcript contains 'country', got: " +
                    final_text(res));
        require(asr::doc::document_integrity_errors(res.document).empty(),
                std::string(c.label) + " folded Document passes integrity check");
    }
}

void verify_ogg(const std::shared_ptr<grpc::Channel>& channel, const std::string& jfk_path) {
    // Ogg was sniffed as in-process audio, but miniaudio decodes neither
    // Vorbis nor Opus here, so every .ogg/.opus upload failed
    // INVALID_ARGUMENT "cannot decode media". ffmpeg decodes both.
    for (const char* codec : {"libopus", "libvorbis"}) {
        const std::string ogg =
            generate_media("-i " + jfk_path + " -c:a " + std::string(codec), ".ogg");
        asrv1::TranscribeOptions options = options_for("tiny.en");
        options.set_emit_document(true);
        StreamResult result = transcribe(channel, options, ogg);
        const std::string expected = std::string(codec).substr(3);  // opus, vorbis
        require(result.status.ok(),
                expected + " in ogg transcribes OK: " + result.status.error_message());
        require(result.media_info.audio_codec() == expected && !result.media_info.has_video(),
                "MediaInfo names the " + expected + " codec, got " +
                    result.media_info.audio_codec());
        require(lower(final_text(result)).find("country") != std::string::npos,
                expected + " transcript contains 'country', got: " + final_text(result));
        require(result.document.origin().mimetype() == "audio/ogg",
                "the document origin keeps the sniffed ogg mimetype");
    }
}

void verify_cancel_video(const std::shared_ptr<grpc::Channel>& channel,
                         const asr::AsrServiceImpl& service, const std::string& jfk_path,
                         const std::string& jfk) {
    // Five minutes of the spoken fixture, looped under a moving picture, so
    // both ffmpeg children are still busy when the client cancels. That
    // path used to throw from the audio child's exit status past a running
    // keyframe thread: std::terminate, and every stream on the server gone.
    std::string media = generate_media(
        "-f lavfi -i testsrc2=duration=300:size=160x120:rate=5 -stream_loop -1 -i " +
            jfk_path + " -t 300 -pix_fmt yuv420p",
        ".mp4");
    asrv1::TranscribeOptions options = options_for("tiny.en");
    options.set_emit_keyframes(true);
    options.set_keyframe_interval_seconds(1);

    const long finished = handlers_finished(service);
    grpc::Status status = transcribe_and_cancel(
        channel, options, media,
        [](const asrv1::TranscribeResponse& response) { return response.has_final_segment(); });
    require(status.error_code() == grpc::StatusCode::CANCELLED,
            "the client sees its own cancel, got " + status.error_message());
    require(eventually([&] { return handlers_finished(service) > finished; },
                       std::chrono::seconds(30)),
            "the cancelled video stream ends long before its five minutes are transcribed");

    // The single model state came back: the next stream runs normally.
    StreamResult next = transcribe(channel, options_for("tiny.en"), jfk);
    require(next.status.ok() && lower(final_text(next)).find("country") != std::string::npos,
            "the server serves the next stream after the cancel: " + next.status.error_message());
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
    require(info.min_window_seconds() == 10 && info.max_window_seconds() == 10,
            "window bounds reported on GetServiceInfo");
    require(info.warmed_up(), "warmed_up=true reported on GetServiceInfo");
    require(!info.remote_enabled(), "remote_enabled=false reported when allowlist is unset");
}

void verify_prometheus_metrics(uint16_t metrics_port) {
    const std::string body = http_get_metrics(metrics_port);
    auto expect_ge = [&](std::string_view series, double min_val) {
        const auto got = scrape_metric_value(body, series);
        require(got.has_value() && *got >= min_val,
                "metric " + std::string(series) + " >= " + std::to_string(min_val) +
                    ", got " + (got.has_value() ? std::to_string(*got) : "missing"));
    };
    auto expect_eq = [&](std::string_view series, double expected) {
        const auto got = scrape_metric_value(body, series);
        require(got.has_value() && *got == expected,
                "metric " + std::string(series) + " == " + std::to_string(expected) +
                    ", got " + (got.has_value() ? std::to_string(*got) : "missing"));
    };

    expect_ge("grpc_asr_stream_end_total{result=\"ok\"}", 1.0);
    expect_ge("grpc_asr_stream_end_total{result=\"unknown_container\"}", 1.0);
    expect_eq("grpc_asr_states_total{model=\"tiny.en\"}", 1.0);
    expect_eq("grpc_asr_states_busy{model=\"tiny.en\"}", 0.0);
    expect_eq("grpc_asr_streams_active", 0.0);
    expect_eq("grpc_asr_upload_bytes_resident", 0.0);
    expect_eq("grpc_asr_memfd_bytes_resident", 0.0);
    expect_ge("grpc_asr_first_segment_seconds_count{model=\"tiny.en\",path=\"live\"}", 1.0);
    expect_ge("grpc_asr_realtime_factor_count{model=\"tiny.en\",backend=\"cpu\"}", 1.0);
    expect_ge("grpc_asr_windows_total", 1.0);
    expect_ge("grpc_asr_audio_seconds_total{model=\"tiny.en\"}", 1.0);
    expect_ge("grpc_asr_warmup_seconds", 0.0);
}

std::string run_server_until_listening(const std::string& binary, const std::string& models_dir,
                                       bool warmup) {
    std::string log_path = "/tmp/grpc-asr-boot-XXXXXX.log";
    const int log_fd = mkstemps(log_path.data(), 4);
    require(log_fd >= 0, "temp boot log");
    const uint16_t listen_port = pick_free_port();

    const pid_t pid = ::fork();
    require(pid >= 0, "fork server subprocess");
    if (pid == 0) {
        ::dup2(log_fd, STDOUT_FILENO);
        ::dup2(log_fd, STDERR_FILENO);
        ::close(log_fd);
        ::setenv("GRPC_ASR_BACKEND", "cpu", 1);
        ::setenv("GRPC_ASR_MODELS_DIR", models_dir.c_str(), 1);
        ::setenv("GRPC_ASR_MODELS", "tiny.en", 1);
        ::setenv("GRPC_ASR_CONCURRENCY", "1", 1);
        ::setenv("GRPC_ASR_THREADS", "2", 1);
        const std::string addr = "127.0.0.1:" + std::to_string(listen_port);
        ::setenv("GRPC_ASR_LISTEN_ADDRESS", addr.c_str(), 1);
        ::setenv("GRPC_ASR_WARMUP", warmup ? "1" : "0", 1);
        ::execl(binary.c_str(), binary.c_str(), static_cast<char*>(nullptr));
        ::_exit(127);
    }
    ::close(log_fd);

    const bool ready = eventually(
        [&] { return slurp(log_path).find("grpc-asr listening on") != std::string::npos; },
        std::chrono::seconds(30));

    if (ready) {
        auto ch = grpc::CreateChannel("127.0.0.1:" + std::to_string(listen_port),
                                      grpc::InsecureChannelCredentials());
        auto stub = asrv1::AsrService::NewStub(ch);
        grpc::ClientContext ctx;
        asrv1::GetServiceInfoRequest req;
        asrv1::GetServiceInfoResponse info;
        require(stub->GetServiceInfo(&ctx, req, &info).ok(), "subprocess GetServiceInfo OK");
        require(info.warmed_up() == warmup, "subprocess GetServiceInfo.warmed_up matches env");
    }

    ::kill(pid, SIGTERM);
    int status = 0;
    ::waitpid(pid, &status, 0);
    const std::string output = slurp(log_path);
    std::remove(log_path.c_str());
    require(ready, "server reached listening state; log:\n" + output);
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "server exited cleanly on SIGTERM");
    return output;
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

    // #18: Corrupt/stub model weights fail fast with exit 1 before listening.
    std::string bad_dir_tpl = "/tmp/grpc-asr-bad-model-XXXXXX";
    char* bad_dir_raw = ::mkdtemp(bad_dir_tpl.data());
    require(bad_dir_raw != nullptr, "mkdtemp for corrupt model dir");
    const std::filesystem::path bad_dir(bad_dir_raw);
    const std::filesystem::path bad_file = bad_dir / "ggml-bad.bin";
    {
        std::ofstream out(bad_file, std::ios::binary);
        out << "not a valid ggml whisper weight file";
    }
    std::string bad_log = (bad_dir / "boot.log").string();
    env = "GRPC_ASR_BACKEND=cpu GRPC_ASR_MODELS_DIR=" + bad_dir.string() +
          " GRPC_ASR_MODELS=bad ";
    code = std::system((env + "timeout 30 " + binary + " >" + bad_log + " 2>&1").c_str());
    const std::string bad_output = slurp(bad_log);
    std::filesystem::remove_all(bad_dir);
    require(WIFEXITED(code) && WEXITSTATUS(code) == 1,
            "corrupt model exits 1 at startup, got " + std::to_string(WEXITSTATUS(code)));
    require(bad_output.find("grpc-asr listening on") == std::string::npos,
            "corrupt model fails before listening");

    // #18: Startup warmup enabled logs one warmup line per state before
    // listening; GRPC_ASR_WARMUP=0 logs none.
    const std::string warm_log = run_server_until_listening(binary, models_dir, /*warmup=*/true);
    const size_t warm_pos = warm_log.find("grpc-asr warmup: tiny.en state 0 ");
    const size_t listen_pos = warm_log.find("grpc-asr listening on");
    require(warm_pos != std::string::npos && warm_pos < listen_pos,
            "warmup=1 logs warmup line before listening; log:\n" + warm_log);
    require(warm_log.find("grpc-asr warmup:", warm_pos + 1) == std::string::npos,
            "warmup=1 logs exactly one warmup line for concurrency=1");

    const std::string cold_log = run_server_until_listening(binary, models_dir, /*warmup=*/false);
    require(cold_log.find("grpc-asr warmup:") == std::string::npos,
            "GRPC_ASR_WARMUP=0 suppresses warmup log lines; log:\n" + cold_log);

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
        return missing_model_input("no model weights at $GRPC_ASR_TEST_MODEL");
    }
    std::string jfk = sample == nullptr ? "" : slurp(sample);
    if (jfk.empty()) {
        return missing_model_input("no fixture wav at $GRPC_ASR_TEST_SAMPLE");
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
        asr::MetricsHttpServer metrics("127.0.0.1:0");
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
        verify_per_channel_stereo(channel, config, jfk);
        verify_window_stall_and_clamping(config, pool, jfk);
        verify_trailer_to_gone_client("127.0.0.1:" + std::to_string(port), service);
        {
            // Forty minutes of low hiss: whisper decodes it into no segment
            // at all, so nothing is ever written, and a server that ignored
            // the call would decode it for half a minute or more after the
            // client left, well past the bounds asserted below.
            const std::string hiss = make_hiss_wav(2400.0);
            verify_cancel_without_speech(channel, service, hiss, jfk);
            verify_deadline_without_speech(channel, service, hiss);
            verify_cancel_while_queued(channel, service, hiss, jfk);
            verify_queue_timeout(config, pool, channel, hiss, jfk, metrics.port());
        }
        verify_error_matrix(channel, config, pool);
        verify_byte_cap(config, pool);
        verify_service_info(channel);
        if (have_ffmpeg()) {
            verify_video(channel, sample);
            verify_ogg(channel, sample);
            verify_pipe_streaming_and_mp4_layouts(channel, sample);
            verify_new_container_families(channel, sample);
            verify_cancel_video(channel, service, sample, jfk);
        } else {
            std::println("note: ffmpeg not on PATH; video cases not run");
        }
        verify_prometheus_metrics(metrics.port());
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
