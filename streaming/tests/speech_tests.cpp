#include "speech.h"
#include "decoder.h"
#include "worker.h"
#include <grpcpp/grpcpp.h>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <bit>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
std::string read(const char* path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("test asset missing");
  return {std::istreambuf_iterator<char>(file), {}};
}
int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: speech_tests MODEL JFK_WAV");
    const auto weights = read(argv[1]);
    media::SpeechModel model(weights, "test-tiny.en", 4);
    check(model.model_hash().size() == 32, "missing model fingerprint");
    media::mediapb::SpeechOptions unsupported_translation;
    unsupported_translation.set_model_id("test-tiny.en");
    unsupported_translation.set_task(media::mediapb::SPEECH_TASK_TRANSLATE_TO_ENGLISH);
    for (const std::string& language : {std::string(), std::string("en")}) {
      unsupported_translation.set_language(language);
      bool rejected_translation = false;
      try { model.validate_options(unsupported_translation); }
      catch (const media::Failure& f) { rejected_translation = f.code == grpc::StatusCode::INVALID_ARGUMENT; }
      check(rejected_translation, "English-only weights accepted unsupported translation");
    }
    const auto bytes = read(argv[2]);
    size_t offset = 0;
    media::SpeechWindow window{1, 16000, 16000, media::WindowBoundary::End, {}};
    media::mediapb::DecodeOptions decode;
    decode.set_emit_audio(true);
    media::decode_media([&](uint8_t* out, int cap) {
      const auto n = std::min(bytes.size() - offset, static_cast<size_t>(cap));
      std::memcpy(out, bytes.data() + offset, n); offset += n; return static_cast<int>(n);
    }, [&](media::mediapb::StreamEvent event) {
      if (!event.has_audio()) return;
      const auto& pcm = event.audio().pcm_f32le();
      for (size_t i = 0; i < pcm.size(); i += 4) {
        uint32_t bits = 0;
        for (int b = 0; b < 4; ++b) bits |= static_cast<uint32_t>(static_cast<unsigned char>(pcm[i+b])) << (8*b);
        window.samples.push_back(std::bit_cast<float>(bits));
      }
    }, [] { return false; }, decode);
    media::mediapb::SpeechOptions options;
    options.set_model_id("test-tiny.en"); options.set_word_timestamps(true);
    const auto segments = model.infer(window, 7, options, [] { return false; });
    std::string transcript;
    for (const auto& s : segments) {
      check(s.track_id() == 7 && !s.final() && s.revision() == 1, "incorrect segment identity");
      check(s.start_us() >= 1000000 && s.end_us() >= s.start_us(), "incorrect absolute timestamp");
      check(s.tokens_size() > 0 && s.source_language() == "en", "missing native token metadata");
      transcript += s.text();
    }
    std::cout << transcript << '\n';
    std::transform(transcript.begin(), transcript.end(), transcript.begin(), [](unsigned char c) { return std::tolower(c); });
    check(transcript.find("ask not") != std::string::npos && transcript.find("country") != std::string::npos,
          "reference speech not recognized");
    bool cancelled = false;
    try { model.infer(window, 7, options, [] { return true; }); }
    catch (const media::Failure& f) { cancelled = f.code == grpc::StatusCode::CANCELLED; }
    check(cancelled, "pre-cancelled inference ran");
    options.set_model_id("unregistered");
    bool rejected = false;
    try { model.infer(window, 7, options, [] { return false; }); }
    catch (const media::Failure& f) { rejected = f.code == grpc::StatusCode::INVALID_ARGUMENT; }
    check(rejected, "model mismatch ignored");
    options.set_model_id("test-tiny.en");
    options.set_language("fr");
    rejected = false;
    try { model.infer(window, 7, options, [] { return false; }); }
    catch (const media::Failure& f) { rejected = f.code == grpc::StatusCode::INVALID_ARGUMENT; }
    check(rejected, "English-only model accepted another language");
    options.set_language("en");
    media::SpeechWorker worker(model);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&worker);
    struct ServerGuard {
      std::unique_ptr<grpc::Server> server;
      ~ServerGuard() { if (server) { server->Shutdown(); server->Wait(); } }
    } guard;
    const char* external = std::getenv("MEDIA_WORKER_TEST_TARGET");
    if (!external) {
      guard.server = builder.BuildAndStart();
      check(guard.server != nullptr, "worker server did not bind");
    }
    auto stub = media::mediapb::SpeechWorkerService::NewStub(grpc::CreateChannel(
        external ? external : "127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
    {
      grpc::ClientContext context;
      context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
      auto rpc = stub->Transcribe(&context);
      media::mediapb::TranscribeRequest request;
      request.mutable_open()->set_stream_id(media::uuid());
      *request.mutable_open()->mutable_options() = unsupported_translation;
      check(rpc->Write(request), "translation rejection request write failed");
      rpc->WritesDone();
      media::mediapb::TranscribeResponse response;
      check(!rpc->Read(&response), "English-only worker acknowledged translation");
      check(rpc->Finish().error_code() == grpc::StatusCode::INVALID_ARGUMENT,
            "English-only worker did not reject translation");
    }
    {
      grpc::ClientContext context;
      context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(60));
      auto rpc = stub->Transcribe(&context);
      media::mediapb::TranscribeRequest request;
      request.mutable_open()->set_stream_id(media::uuid());
      request.mutable_open()->set_track_id(7);
      *request.mutable_open()->mutable_options() = options;
      check(rpc->Write(request), "worker open failed");
      media::mediapb::TranscribeResponse response;
      check(rpc->Read(&response) && response.has_opened(), "worker handshake absent");
      check(response.opened().producer().model_sha256() == model.model_hash() &&
            response.opened().producer().configuration_sha256().size() == 32 &&
            response.opened().producer().build_id().size() == 64, "worker provenance incomplete");
      {
        std::vector<std::unique_ptr<grpc::ClientContext>> contexts;
        std::vector<std::unique_ptr<grpc::ClientReaderWriter<media::mediapb::TranscribeRequest, media::mediapb::TranscribeResponse>>> idle;
        for (int i = 0; i < 31; ++i) {
          auto pending = std::make_unique<grpc::ClientContext>();
          pending->set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));
          auto call = stub->Transcribe(pending.get());
          check(call->Write(request), "idle worker track open failed");
          media::mediapb::TranscribeResponse opened;
          check(call->Read(&opened) && opened.has_opened(), "idle tracks consumed inference slot");
          contexts.push_back(std::move(pending)); idle.push_back(std::move(call));
        }
        grpc::ClientContext second_context;
        second_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        auto second = stub->Transcribe(&second_context);
        second->WritesDone();
        media::mediapb::TranscribeResponse ignored;
        check(!second->Read(&ignored), "overloaded worker accepted a second stream");
        const auto overflow_status = second->Finish();
        if (overflow_status.error_code() != grpc::StatusCode::RESOURCE_EXHAUSTED)
          std::cerr << "worker overload status: " << overflow_status.error_code() << " " << overflow_status.error_message() << '\n';
        check(overflow_status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED, "worker overload status wrong");
        media::mediapb::TranscribeRequest finish_idle;
        finish_idle.mutable_finish()->set_window_count(0);
        for (auto& call : idle) {
          check(call->Write(finish_idle), "idle track close failed"); call->WritesDone();
          check(call->Read(&ignored) && ignored.has_closed(), "idle track close absent");
          check(!call->Read(&ignored) && call->Finish().ok(), "idle track closure failed");
        }
      }
      request.Clear();
      auto* input = request.mutable_window();
      input->set_window_number(1); input->set_first_sample(window.first_sample);
      input->set_first_new_sample(window.first_sample); input->set_boundary(media::mediapb::INFERENCE_BOUNDARY_END);
      auto* encoded = input->mutable_pcm_f32le();
      for (float sample : window.samples) {
        const auto bits = std::bit_cast<uint32_t>(sample);
        for (int b = 0; b < 4; ++b) encoded->push_back(static_cast<char>(bits >> (8*b)));
      }
      check(rpc->Write(request), "worker audio write failed");
      // Finish is deliberately withheld until the actual model output arrives.
      check(rpc->Read(&response) && response.has_hypothesis(), "worker buffered inference until finish");
      check(response.hypothesis().window_number() == 1 && response.hypothesis().track_id() == 7,
            "worker output identity mismatch");
      std::string wire_text;
      for (const auto& segment : response.hypothesis().segments()) {
        check(!segment.final(), "worker falsely finalized unreconciled text");
        wire_text += segment.text();
      }
      check(wire_text.find("country") != std::string::npos, "wire hypothesis missing reference speech");
      request.Clear(); request.mutable_finish()->set_window_count(1);
      check(rpc->Write(request), "worker finish failed"); rpc->WritesDone();
      check(rpc->Read(&response) && response.has_closed() && response.closed().window_count() == 1,
            "worker close accounting missing");
      check(!rpc->Read(&response) && rpc->Finish().ok(), "worker stream did not close successfully");
    }
    for (bool malformed_window : {false, true}) {
      grpc::ClientContext context;
      context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
      auto rpc = stub->Transcribe(&context);
      media::mediapb::TranscribeRequest request;
      request.mutable_open()->set_stream_id(media::uuid());
      *request.mutable_open()->mutable_options() = options;
      check(rpc->Write(request), "worker rejection test open failed");
      media::mediapb::TranscribeResponse response;
      check(rpc->Read(&response) && response.has_opened(), "worker slot not released");
      if (malformed_window) {
        request.Clear(); request.mutable_window()->set_window_number(2);
        check(rpc->Write(request), "malformed request write failed");
      }
      rpc->WritesDone();
      check(!rpc->Read(&response), "worker accepted malformed close or window");
      check(rpc->Finish().error_code() == (malformed_window ? grpc::StatusCode::INVALID_ARGUMENT : grpc::StatusCode::CANCELLED),
            "worker lifecycle rejection status wrong");
    }
    std::cout << "Live worker inference, overload and lifecycle tests passed\n";
    std::cout << "Native speech recognition, cancellation and model identity tests passed\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
