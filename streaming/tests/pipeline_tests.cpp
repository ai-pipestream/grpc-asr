#include "pipeline.h"
#include "parakeet_model.h"
#include "http_fixture.h"
#include "service.h"
#include "worker.h"
#include "decoder.h"
#include "media_fixture.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <charconv>
#include <cstring>
#include <grpcpp/grpcpp.h>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <thread>
#include <map>

using namespace std::chrono_literals;
namespace mediapb = media::mediapb;
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
std::string read(const char* path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("pipeline asset missing");
  return {std::istreambuf_iterator<char>(file), {}};
}
struct Server {
  std::unique_ptr<grpc::Server> server;
  int port = 0;
  explicit Server(grpc::Service& service) {
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service); server = builder.BuildAndStart();
    check(server != nullptr, "pipeline server bind failed");
  }
  ~Server() { server->Shutdown(); server->Wait(); }
  std::string address() const { return "127.0.0.1:" + std::to_string(port); }
};
void check_reference(const std::string& text) {
  std::string normalized; bool space = false;
  for (unsigned char c : text) {
    if (c >= 'A' && c <= 'Z') c += 'a'-'A';
    if (c >= 'a' && c <= 'z') { if (space && !normalized.empty()) normalized += ' '; normalized += c; space = false; }
    else if (c == ' ') space = true;
  }
  const std::string expected = "and so my fellow americans ask not what your country can do for you ask what you can do for your country";
  if (normalized != expected) std::cerr << "Expected: " << expected << "\nActual: " << normalized << '\n';
  check(normalized == expected,
        "streamed final transcript lost or duplicated boundary words");
}
void http_pipeline(const media::SpeechPipelineFactory& factory, const media::SpeechBackend& model,
                   const std::string& data, size_t expected_video_packets) {
  const auto& model_hash = model.model_hash();
  const std::string language = model.reports_language() ? "en" : "";
  const char* external = std::getenv("MEDIA_PIPELINE_HTTP_TARGET");
  uint16_t feed_port = 0;
  if (const char* configured = std::getenv("MEDIA_PIPELINE_HTTP_PORT")) {
    const auto end = configured + std::strlen(configured);
    const auto parsed = std::from_chars(configured, end, feed_port);
    check(parsed.ec == std::errc() && parsed.ptr == end && feed_port != 0,
          "invalid HTTP fixture port");
  }
  Fixture feed([&](int fd, Fixture& fixture) {
    send_bytes(fd, "HTTP/1.1 200 OK\r\nContent-Type: video/mp4\r\nContent-Length: " + std::to_string(data.size()) + "\r\n\r\n");
    const auto split = data.size()*3/4;
    send_bytes(fd, data.substr(0, split));
    while (!fixture.release && !fixture.stop) std::this_thread::sleep_for(1ms);
    if (!fixture.stop) send_bytes(fd, data.substr(split));
  }, feed_port);
  media::Registry registry;
  media::Service service(registry, &factory, feed.policy()); Server server(service);
  auto stub = mediapb::MediaService::NewStub(grpc::CreateChannel(external ? external : server.address(), grpc::InsecureChannelCredentials()));
  grpc::ClientContext live_context; live_context.set_deadline(std::chrono::system_clock::now()+90s);
  mediapb::SubscribeRequest subscription; subscription.add_payload_kinds(mediapb::PAYLOAD_KIND_TRANSCRIPT);
  subscription.mutable_hydration()->set_metadata(true);
  // No UUID selector: register before the stream exists and follow future streams.
  auto live = stub->Subscribe(&live_context, subscription);
  mediapb::SubscribeResponse delivered;
  check(live->Read(&delivered) && delivered.has_snapshot_begin(), "global HTTP subscription begin missing");
  check(live->Read(&delivered), "HTTP snapshot ended early");
  if (external) {
    // The same packaged registry can retain the preceding direct-input cases.
    while (delivered.has_snapshot()) check(live->Read(&delivered), "HTTP snapshot ended early");
  }
  check(delivered.has_snapshot_end(), "HTTP snapshot missing end");
  grpc::ClientContext context; context.set_deadline(std::chrono::system_clock::now()+90s);
  mediapb::PullRequest request; request.mutable_open()->mutable_http()->set_url(feed.url);
  request.mutable_open()->mutable_decode()->set_emit_packets(true);
  auto* speech = request.mutable_open()->add_speech(); speech->set_model_id(model.model_id());
  speech->mutable_windowing()->set_window_ms(4000); speech->mutable_windowing()->set_overlap_ms(1000);
  auto pull = stub->Pull(&context, request);
  mediapb::PullResponse response;
  std::string source, final_text, id;
  bool http_info = false, closed = false, hypothesis = false;
  size_t packets = 0;
  std::map<std::string, mediapb::TranscriptSegment> revisions;
  while (pull->Read(&response)) {
    const auto& event = response.event();
    if (event.has_opened()) {
      id = event.cursor().stream_id();
      check(live->Read(&delivered) && delivered.event().has_opened() && delivered.event().cursor().stream_id()==id,
            "global subscriber missed future HTTP stream");
    }
    if (event.has_http_info()) http_info = event.http_info().origin()==media::HttpInput::origin(feed.url) &&
        event.http_info().status_code()==200 && event.producer().library_version().starts_with("libcurl/");
    if (event.has_source()) source += event.source().data();
    if (event.has_packet() && event.packet().track_id()==0) ++packets;
    if (event.has_hypothesis()) {
      check(event.hypothesis().track_id()==1 && event.producer().model_sha256()==model_hash, "HTTP speech provenance mismatch");
      const auto prepared=model.encoder_context(event.hypothesis().sample_count());
      check(event.hypothesis().has_encoder_context()==prepared.has_value() &&
            (!prepared || event.hypothesis().encoder_context().SerializeAsString()==prepared->SerializeAsString()),
            "HTTP encoder preparation metadata mismatch");
      hypothesis = true; feed.release = true;
    }
    if (event.has_transcript()) {
      const auto& t = event.transcript();
      check(t.source_language()==language && t.output_language()==language && t.contributors_size()==2 &&
            t.contributors(0).model_sha256()==model_hash, "HTTP transcript provenance missing");
      if (revisions.contains(t.segment_id()))
        check(!revisions.at(t.segment_id()).final() && t.revision()==revisions.at(t.segment_id()).revision()+1,
              "HTTP transcript revision is not contiguous");
      else check(t.revision()==1, "HTTP transcript started at wrong revision");
      revisions[t.segment_id()] = t;
      if (t.final()) final_text += t.text();
      check(live->Read(&delivered) && delivered.event().transcript().SerializeAsString()==t.SerializeAsString(),
            "HTTP transcript differs on global subscription");
    }
    if (event.has_closed()) closed = event.closed().status_code()==0 && event.closed().source_sha256()==media::sha256(data);
  }
  check(pull->Finish().ok() && hypothesis && http_info && closed && source==data && packets==expected_video_packets,
        "HTTP video speech pipeline incomplete");
  check_reference(final_text);
  check(live->Read(&delivered) && delivered.event().has_closed(), "global subscriber missed HTTP close");
  live_context.TryCancel(); while (live->Read(&delivered)) {} live->Finish();
  grpc::ClientContext hydrate_context; hydrate_context.set_deadline(std::chrono::system_clock::now()+5s);
  subscription.add_stream_ids(id);
  auto hydrate = stub->Subscribe(&hydrate_context, subscription);
  check(hydrate->Read(&delivered) && delivered.has_snapshot_begin(), "HTTP hydration begin missing");
  check(hydrate->Read(&delivered) && delivered.has_snapshot() && delivered.snapshot().has_closed() &&
        delivered.snapshot().http_info().status_code()==200 &&
        delivered.snapshot().transcripts_size()==static_cast<int>(revisions.size()), "HTTP hydration incomplete");
  for (const auto& t : delivered.snapshot().transcripts())
    check(t.final() && revisions.at(t.segment_id()).SerializeAsString()==t.SerializeAsString(), "HTTP hydration lost final transcript");
  hydrate_context.TryCancel(); while (hydrate->Read(&delivered)) {} hydrate->Finish();
  std::cout << "Local HTTP H.264/AAC -> worker -> global subscriber and hydration passed\n";
}
int main(int argc, char** argv) {
  try {
    if (argc != 4 && argc != 5) throw std::runtime_error("usage: pipeline_tests MODEL VAD_MODEL JFK_WAV [whisper|parakeet]");
    const std::string backend = argc == 5 ? argv[4] : "whisper";
    check(backend == "whisper" || backend == "parakeet", "unknown pipeline test backend");
    const char* selected_case = std::getenv("MEDIA_PIPELINE_TEST_CASE");
    const std::string case_name = selected_case ? selected_case : "all";
    check(case_name == "all" || case_name == "0" || case_name == "1" || case_name == "2" ||
          case_name == "3" || case_name == "4" || case_name == "http", "unknown pipeline test case");
    const bool trace = std::getenv("MEDIA_PIPELINE_TRACE") != nullptr;
    const auto weights = read(argv[1]), vad = read(argv[2]), wav = read(argv[3]);
    size_t audio_offset = 0; std::string pcm16;
    mediapb::DecodeOptions decode; decode.set_emit_audio(true);
    media::decode_media([&](uint8_t* out, int capacity) {
      const auto n = std::min(wav.size()-audio_offset, static_cast<size_t>(capacity));
      std::memcpy(out, wav.data()+audio_offset, n); audio_offset+=n; return static_cast<int>(n);
    }, [&](mediapb::StreamEvent event) {
      if (!event.has_audio()) return;
      const auto& pcm = event.audio().pcm_f32le();
      for (size_t i=0; i<pcm.size(); i+=4) {
        uint32_t bits=0;
        for (int b=0; b<4; ++b) bits |= static_cast<uint32_t>(static_cast<unsigned char>(pcm[i+b])) << (8*b);
        const auto value = static_cast<int16_t>(std::clamp(std::lround(std::bit_cast<float>(bits)*32768), -32768L, 32767L));
        pcm16.push_back(static_cast<char>(static_cast<uint16_t>(value)&255));
        pcm16.push_back(static_cast<char>(static_cast<uint16_t>(value)>>8));
      }
    }, [] {return false;}, decode);
    const auto video_fixture = matroska_fixture(pcm16);
    const auto mp4_fixture = fragmented_mp4_fixture(pcm16);
    const char* configured_id = std::getenv("MEDIA_PIPELINE_TEST_MODEL_ID");
    const std::string model_id = configured_id ? configured_id : (backend == "parakeet" ? "parakeet-test" : "test-tiny.en");
    std::unique_ptr<media::SpeechBackend> selected;
    if (backend == "parakeet") selected = std::make_unique<media::ParakeetModel>(weights, model_id, 4);
    else selected = std::make_unique<media::SpeechModel>(weights, model_id, 4);
    auto& model = *selected;
    const std::string language = model.reports_language() ? "en" : "";
    media::SpeechWorker worker(model); Server worker_server(worker);
    media::SpeechPipelineFactory factory(worker_server.address(), vad);
    media::Registry registry; media::Service ingress(registry, &factory); Server ingress_server(ingress);
    const char* external = std::getenv("MEDIA_PIPELINE_TEST_TARGET");
    auto stub = mediapb::MediaService::NewStub(grpc::CreateChannel(external ? external : ingress_server.address(), grpc::InsecureChannelCredentials()));
    mediapb::OpenStream raw_only; auto* raw = raw_only.add_speech();
    raw->set_model_id(model.model_id()); raw->set_reconcile_transcript(false);
    factory.resolve(raw_only);
    check(!raw_only.speech(0).reconcile_transcript() && !raw_only.speech(0).word_timestamps(), "raw-only option overridden");
    for (int mode : {0, 1, 2, 3, 4}) {
      if (case_name != "all" && case_name != std::to_string(mode)) continue;
      const bool smart = mode==2 || mode==4, video = mode!=0, mp4 = mode>=3;
      const auto& data = mp4 ? mp4_fixture : video ? video_fixture.bytes : wav;
      size_t video_packets = 0;
      grpc::ClientContext context; context.set_deadline(std::chrono::system_clock::now() + 90s);
      auto rpc = stub->Ingest(&context);
      mediapb::IngestRequest open;
      open.mutable_open()->mutable_direct()->set_name(mp4 ? "reference.mp4" : video ? "reference.mkv" : "reference.wav");
      open.mutable_open()->mutable_decode()->set_emit_packets(video);
      auto* speech = open.mutable_open()->add_speech();
      speech->set_model_id(model.model_id());
      if (smart && model.supports_language_hint()) speech->set_language("en");
      auto* w = speech->mutable_windowing(); w->set_window_ms(4000); w->set_overlap_ms(smart ? 500 : 1000);
      w->set_strategy(smart ? mediapb::CHUNKING_STRATEGY_SPEECH_AWARE : mediapb::CHUNKING_STRATEGY_FIXED);
      if (smart) {
        auto* b = w->mutable_speech_boundary(); b->set_vad_model_id("silero-v5.1.2");
        b->set_max_window_ms(6000); b->set_min_pause_ms(250); b->set_speech_padding_ms(80);
        b->set_no_pause_policy(mediapb::NO_PAUSE_POLICY_SPLIT_WITH_OVERLAP);
      }
      check(rpc->Write(open), "ingress speech open failed");
      mediapb::IngestResponse response;
      check(rpc->Read(&response) && response.event().has_opened(), "ingress handshake missing");
      const auto id = response.event().cursor().stream_id();
      check(response.event().opened().speech_size() == 1, "speech options not echoed");
      check(response.event().opened().speech(0).reconcile_transcript() &&
            response.event().opened().speech(0).word_timestamps(), "resolved reconciliation options missing");
      mediapb::SubscribeRequest live_request; live_request.add_stream_ids(id);
      live_request.add_payload_kinds(mediapb::PAYLOAD_KIND_TRANSCRIPT);
      live_request.mutable_hydration()->set_metadata(true);
      grpc::ClientContext live_context; live_context.set_deadline(std::chrono::system_clock::now() + 90s);
      auto live = stub->Subscribe(&live_context, live_request);
      mediapb::SubscribeResponse initial;
      check(live->Read(&initial) && initial.has_snapshot_begin(), "live subscription boundary missing");
      check(live->Read(&initial) && initial.has_snapshot(), "live subscription metadata missing");
      check(live->Read(&initial) && initial.has_snapshot_end(), "live subscription registration incomplete");
      std::map<std::string, mediapb::TranscriptSegment> revisions;
      std::atomic<bool> saw_hypothesis{false}, done{false}, writes_ok{true};
      std::jthread writer([&] {
        size_t offset = 0; uint64_t count = 0;
        const size_t split = data.size() * 3 / 4;
        auto send_until = [&](size_t end) {
          while (offset < end) {
            mediapb::IngestRequest request;
            auto* chunk = request.mutable_chunk();
            const auto n = std::min(size_t(8192), end - offset);
            chunk->set_chunk_number(++count); chunk->set_byte_offset(offset); chunk->set_data(data.substr(offset, n));
            if (!rpc->Write(request)) { writes_ok = false; return false; }
            offset += n;
          }
          return true;
        };
        if (!send_until(split)) return;
        const auto deadline = std::chrono::steady_clock::now() + 75s;
        while (!saw_hypothesis && !done && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(5ms);
        if (!saw_hypothesis) { writes_ok = false; context.TryCancel(); return; }
        if (!send_until(data.size())) return;
        mediapb::IngestRequest finish;
        finish.mutable_finish()->set_chunk_count(count); finish.mutable_finish()->set_byte_count(data.size());
        finish.mutable_finish()->set_sha256(media::sha256(data));
        if (!rpc->Write(finish)) writes_ok = false;
        rpc->WritesDone();
      });
      std::string source, text, final_text;
      int hypotheses = 0; bool closed = false, evidence_ok = true;
      while (rpc->Read(&response)) {
        const auto& event = response.event();
        if (event.has_source()) source += event.source().data();
        if (video && event.has_packet() && event.packet().track_id()==(mp4 ? 0u : 1u)) {
          if (mp4) evidence_ok &= !event.packet().data().empty() && event.packet().has_pts();
          else evidence_ok &= video_packets < video_fixture.video_packets.size() &&
                              event.packet().data()==video_fixture.video_packets[video_packets];
          ++video_packets;
        }
        if (event.has_hypothesis()) {
          if (trace) std::cerr << "HYPOTHESIS " << event.hypothesis().DebugString() << '\n';
          saw_hypothesis = true; ++hypotheses;
          const auto& h = event.hypothesis();
          const auto prepared=model.encoder_context(h.sample_count());
          evidence_ok &= h.has_encoder_context()==prepared.has_value() &&
              (!prepared || h.encoder_context().SerializeAsString()==prepared->SerializeAsString());
          evidence_ok &= h.track_id()==(mp4 ? 1u : 0u);
          evidence_ok &= event.producer().model_sha256() == model.model_hash() && h.contributors_size() == 1;
          if (smart) evidence_ok &= h.contributors(0).model_sha256() == media::sha256(vad);
          for (const auto& segment : h.segments()) {
            text += segment.text();
            evidence_ok &= !segment.final() && segment.source_language()==language &&
                           segment.output_language()==language && segment.task()==mediapb::SPEECH_TASK_TRANSCRIBE;
          }
        }
        if (event.has_transcript()) {
          if (trace) std::cerr << "TRANSCRIPT " << event.transcript().DebugString() << '\n';
          const auto& t = event.transcript();
          if (revisions.contains(t.segment_id())) {
            const auto& prior = revisions.at(t.segment_id());
            evidence_ok &= !prior.final() && t.revision() == prior.revision()+1;
          } else evidence_ok &= t.revision()==1;
          evidence_ok &= t.source_language()==language && t.output_language()==language &&
                         t.task()==mediapb::SPEECH_TASK_TRANSCRIBE && t.producer().service()=="grpc-asr-reconciliation" && t.contributors_size()==2 &&
                         t.contributors(0).model_sha256()==model.model_hash();
          if (t.final()) final_text += t.text();
          revisions[t.segment_id()] = t;
          mediapb::SubscribeResponse delivered;
          evidence_ok &= live->Read(&delivered) && delivered.event().transcript().SerializeAsString()==t.SerializeAsString();
        }
        if (event.has_closed()) closed = event.closed().status_code() == 0 && event.closed().source_sha256() == media::sha256(data);
      }
      done = true; writer.join();
      live_context.TryCancel(); while (live->Read(&initial)) {} live->Finish();
      const auto status = rpc->Finish();
      if (!status.ok()) std::cerr << "pipeline status: " << status.error_code() << ' ' << status.error_message() << '\n';
      check(status.ok() && writes_ok && closed, "pipeline did not close successfully");
      if (!smart) check_reference(final_text);
      if (video) check(video_packets==(mp4 ? (pcm16.size()/2+1599)/1600 : video_fixture.video_packets.size()),
                       "speech processing lost compressed video packets");
      check(!revisions.empty(), "no reconciled transcript published");
      for (const auto& [segment_id, segment] : revisions) check(segment.final(), "provisional tail not flushed before close");
      check(source == data && hypotheses >= 2 && evidence_ok, "pipeline source or provenance mismatch");
      check(text.find("country") != std::string::npos, "pipeline did not recognize reference speech");
      grpc::ClientContext subscribe_context; subscribe_context.set_deadline(std::chrono::system_clock::now() + 5s);
      mediapb::SubscribeRequest subscribe; subscribe.add_stream_ids(id); subscribe.mutable_hydration()->set_metadata(true);
      auto listener = stub->Subscribe(&subscribe_context, subscribe);
      mediapb::SubscribeResponse item;
      check(listener->Read(&item) && item.has_snapshot_begin(), "speech hydration begin missing");
      check(listener->Read(&item) && item.has_snapshot(), "speech hydration missing");
      check(item.snapshot().hypotheses_size() == hypotheses && item.snapshot().has_closed(), "speech results not hydrated");
      check(item.snapshot().transcripts_size()==static_cast<int>(revisions.size()), "transcript hydration duplicated revisions");
      for (const auto& segment : item.snapshot().transcripts())
        check(revisions.at(segment.segment_id()).SerializeAsString()==segment.SerializeAsString(), "hydration lost final revision or provenance");
      subscribe_context.TryCancel(); while (listener->Read(&item)) {} listener->Finish();
      std::cout << (mp4 ? (smart ? "Smart H.264/AAC MP4" : "Fixed H.264/AAC MP4") : smart ? "Smart Matroska video" : video ? "Fixed Matroska video" : "Fixed WAV") << " ingress -> worker -> hydrated output passed\n";
    }
    if (case_name == "all" || case_name == "http")
      http_pipeline(factory, model, mp4_fixture, (pcm16.size()/2+1599)/1600);
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
