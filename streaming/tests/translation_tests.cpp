#include "speech.h"
#include "worker.h"
#include "decoder.h"
#include <grpcpp/grpcpp.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>

namespace mediapb = media::mediapb;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
std::string read(const char* path) {
  std::ifstream input(path, std::ios::binary);
  check(input.good(), "translation asset missing");
  return {std::istreambuf_iterator<char>(input), {}};
}
int main(int argc, char** argv) {
  try {
    check(argc == 4, "usage: translation_tests MULTILINGUAL_MODEL SPANISH_WAV minds14|synthetic");
    const bool synthetic = std::string(argv[3]) == "synthetic";
    check(synthetic || std::string(argv[3]) == "minds14", "unknown translation fixture");
    const auto weights = read(argv[1]), source = read(argv[2]);
    media::SpeechModel model(weights, "test-multilingual", 4);
    std::string pcm;
    size_t offset = 0;
    mediapb::DecodeOptions decode; decode.set_emit_audio(true);
    media::decode_media([&](uint8_t* out, int capacity) {
      const auto n = std::min(source.size()-offset, static_cast<size_t>(capacity));
      std::memcpy(out, source.data()+offset, n); offset += n; return static_cast<int>(n);
    }, [&](mediapb::StreamEvent event) {
      if (event.has_audio()) pcm += event.audio().pcm_f32le();
    }, [] {return false;}, decode);
    check(!pcm.empty() && pcm.size()/4 <= 480000, "translation fixture exceeds one model window");
    media::SpeechWorker worker(model);
    grpc::ServerBuilder builder; int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&worker);
    struct Guard {
      std::unique_ptr<grpc::Server> server;
      ~Guard() { if (server) {server->Shutdown(); server->Wait();} }
    } guard{builder.BuildAndStart()};
    check(guard.server != nullptr, "translation worker bind failed");
    auto stub = mediapb::SpeechWorkerService::NewStub(grpc::CreateChannel(
        "127.0.0.1:"+std::to_string(port), grpc::InsecureChannelCredentials()));
    grpc::ClientContext info_context;
    info_context.set_deadline(std::chrono::system_clock::now()+std::chrono::seconds(5));
    mediapb::GetModelInfoResponse info;
    check(stub->GetModelInfo(&info_context, {}, &info).ok() && info.multilingual() && info.multilingual_known() && info.accepts_language_hint() &&
          info.reports_language() && info.backend()==mediapb::SPEECH_BACKEND_KIND_WHISPER && info.tasks_size()==2 &&
          info.tasks(0)==mediapb::SPEECH_TASK_TRANSCRIBE && info.tasks(1)==mediapb::SPEECH_TASK_TRANSLATE_TO_ENGLISH &&
          info.producer().model_sha256()==model.model_hash(), "multilingual model discovery mismatch");
    for (const auto task : {mediapb::SPEECH_TASK_TRANSCRIBE, mediapb::SPEECH_TASK_TRANSLATE_TO_ENGLISH}) {
      grpc::ClientContext context;
      context.set_deadline(std::chrono::system_clock::now()+std::chrono::seconds(180));
      auto rpc = stub->Transcribe(&context);
      mediapb::TranscribeRequest request;
      request.mutable_open()->set_stream_id(media::uuid()); request.mutable_open()->set_track_id(3);
      auto* options = request.mutable_open()->mutable_options();
      options->set_model_id("test-multilingual"); options->set_task(task);
      options->set_word_timestamps(true); // Also exercise native timings for future reconciliation.
      // Leave language absent: verify native multilingual detection for both tasks.
      check(rpc->Write(request), "translation open write failed");
      mediapb::TranscribeResponse response;
      check(rpc->Read(&response) && response.has_opened(), "translation handshake missing");
      check(response.opened().accepted().options().task()==task &&
            response.opened().producer().model_sha256()==model.model_hash(), "translation handshake metadata mismatch");
      request.Clear(); request.mutable_window()->set_window_number(1);
      request.mutable_window()->set_boundary(mediapb::INFERENCE_BOUNDARY_END);
      request.mutable_window()->set_pcm_f32le(pcm);
      check(rpc->Write(request), "translation window write failed");
      check(rpc->Read(&response) && response.has_hypothesis(), "translation hypothesis missing");
      std::string text;
      for (const auto& segment : response.hypothesis().segments()) {
        check(segment.task()==task && segment.source_language()=="es" && segment.track_id()==3,
              "translation source metadata incorrect");
        check(segment.output_language()==(task==mediapb::SPEECH_TASK_TRANSCRIBE ? "es" : "en"),
              "translation output language incorrect");
        check(!segment.final() && !segment.tokens().empty(), "translation native token evidence missing");
        text += segment.text();
      }
      std::cout << (task==mediapb::SPEECH_TASK_TRANSCRIBE ? "Spanish: " : "English: ") << text << std::endl;
      std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return c>='A' && c<='Z' ? static_cast<char>(c+'a'-'A') : static_cast<char>(c);
      });
      if (synthetic) {
        check(text.find("madrid")!=std::string::npos, "translation lost named location");
        check(text.find(task==mediapb::SPEECH_TASK_TRANSCRIBE ? "biblioteca" : "library")!=std::string::npos,
              "translation did not preserve the library meaning");
      } else if (task==mediapb::SPEECH_TASK_TRANSCRIBE) {
        check(text.find("transferencia")!=std::string::npos && text.find("cuenta")!=std::string::npos &&
              text.find("aplicaci")!=std::string::npos, "Spanish transcription lost application, transfer or account facts");
      } else {
        check(text.find("transfer")!=std::string::npos && text.find("account")!=std::string::npos &&
              text.find("app")!=std::string::npos, "English translation lost application, transfer or account facts");
      }
      if (task==mediapb::SPEECH_TASK_TRANSLATE_TO_ENGLISH)
        check(text.find(synthetic ? "biblioteca" : "transferencia")==std::string::npos,
              "Spanish text mislabeled as English translation");
      request.Clear(); request.mutable_finish()->set_window_count(1);
      check(rpc->Write(request), "translation finish write failed"); rpc->WritesDone();
      check(rpc->Read(&response) && response.has_closed() && response.closed().window_count()==1,
            "translation close accounting missing");
      check(!rpc->Read(&response) && rpc->Finish().ok(), "translation stream failed at close");
    }
    std::cout << "Spanish transcription and English translation passed\n";
  } catch (const std::exception& error) {std::cerr << error.what() << '\n';return 1;}
}
