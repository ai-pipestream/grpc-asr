// Native backend qualification only. This is not a service adapter.
#include "decoder.h"
#include "reconcile.h"
#include "parakeet_model.h"
#include "worker.h"
#include <grpcpp/grpcpp.h>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>

std::string read(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  const auto length = file.tellg();
  if (!file || length<=0 || length>INT64_C(2147483648)) throw std::runtime_error("probe asset missing or too large");
  std::string bytes(static_cast<size_t>(length), '\0');
  file.seekg(0);
  if (!file.read(bytes.data(), length)) throw std::runtime_error("probe asset truncated");
  return bytes;
}
void check(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
int main(int argc, char** argv) {
 try {
    check(argc==3, "usage: parakeet_tests MODEL JFK_AUDIO");
    auto weights=read(argv[1]); const auto source=read(argv[2]);
    media::ParakeetModel model(weights, "parakeet-test", 4);
    std::string().swap(weights);
    std::vector<float> samples; size_t offset = 0;
    media::mediapb::DecodeOptions decode; decode.set_emit_audio(true);
    media::decode_media([&](uint8_t* out, int capacity) {
      const auto count = std::min(source.size()-offset, static_cast<size_t>(capacity));
      std::memcpy(out, source.data()+offset, count); offset += count; return static_cast<int>(count);
    }, [&](media::mediapb::StreamEvent event) {
      if (!event.has_audio()) return;
      const auto& pcm = event.audio().pcm_f32le();
      for (size_t i=0; i<pcm.size(); i+=4) {
        uint32_t bits = 0;
        for (int b=0; b<4; ++b) bits |= static_cast<uint32_t>(static_cast<unsigned char>(pcm[i+b])) << (8*b);
        samples.push_back(std::bit_cast<float>(bits));
      }
    }, [] {return false;}, decode);
    check(!samples.empty() && samples.size()<=30*16000, "probe audio outside bounded window");

    media::SpeechWindow window{1,16000,16000,media::WindowBoundary::End,samples};
    media::mediapb::SpeechOptions options; options.set_model_id("parakeet-test");
    options.set_word_timestamps(true);
    unsigned callbacks=0; bool cancelled=false;
    try { model.infer(window,7,options,[&] { return ++callbacks>1; }); }
    catch(const media::Failure& e) { cancelled=e.code==grpc::StatusCode::CANCELLED; }
    check(cancelled && callbacks>1,"native adapter cancellation failed");
    unsigned exceptions=0; bool callback_error=false;
    try { model.infer(window,7,options,[&]() -> bool {
      if (++exceptions>1) throw std::runtime_error("callback failure");
      return false;
    }); } catch(const std::runtime_error& e) { callback_error=std::string(e.what())=="callback failure"; }
    check(callback_error,"exception escaped native callback boundary");
    window.samples[0]=std::numeric_limits<float>::quiet_NaN();
    bool rejected=false;
    try {model.infer(window,7,options,[]{return false;});}
    catch(const media::Failure& e) {rejected=e.code==grpc::StatusCode::INVALID_ARGUMENT;}
    check(rejected,"nonfinite PCM accepted"); window.samples=samples;
    // Short inputs must not divide by zero in feature normalization or turn
    // encoder-only padding into recognized source speech.
    for (size_t count : {size_t(1), size_t(160), size_t(320), size_t(8000)}) {
      media::SpeechWindow silence{1,0,0,media::WindowBoundary::End,std::vector<float>(count,0.0f)};
      const auto preparation=*model.encoder_context(count);
      check(preparation.analysis_samples()==std::max<size_t>(320,count) &&
            preparation.feature_frames()==preparation.analysis_samples()/160+1 &&
            preparation.encoder_frames()==400 && preparation.frame_hop_samples()==160,
            "incorrect short-input analysis accounting");
      check(model.infer(silence,7,options,[]{return false;}).empty(),"short silence produced speech");
      check(silence.samples.size()==count,"analysis padding changed source samples");
    }
    media::SpeechWorker worker(model);
    grpc::ServerBuilder builder; int port=0;
    builder.AddListeningPort("127.0.0.1:0",grpc::InsecureServerCredentials(),&port);
    builder.RegisterService(&worker);
    struct Guard { std::unique_ptr<grpc::Server> server;
      ~Guard(){if(server){server->Shutdown();server->Wait();}}
    } guard;
    const char* external=std::getenv("MEDIA_WORKER_TEST_TARGET");
    if (!external) {
      guard.server=builder.BuildAndStart();
      check(guard.server!=nullptr,"worker did not bind");
    }
    auto stub=media::mediapb::SpeechWorkerService::NewStub(grpc::CreateChannel(
        external ? external : "127.0.0.1:"+std::to_string(port),grpc::InsecureChannelCredentials()));
    grpc::ClientContext discover; discover.set_deadline(std::chrono::system_clock::now()+std::chrono::seconds(5));
    media::mediapb::GetModelInfoResponse info;
    check(stub->GetModelInfo(&discover,{},&info).ok(),"discovery failed");
    check(info.backend()==media::mediapb::SPEECH_BACKEND_KIND_PARAKEET && !info.multilingual_known() &&
          !info.accepts_language_hint() && !info.reports_language() && info.tasks_size()==1 &&
          info.tasks(0)==media::mediapb::SPEECH_TASK_TRANSCRIBE && info.producer().model_sha256()==model.model_hash() &&
          info.producer().library_version()==model.engine_version() &&
          info.has_minimum_encoder_context_ms() && info.minimum_encoder_context_ms()==4000,
          "incorrect Parakeet capabilities/provenance");
    for(int invalid=0;invalid<3;++invalid) {
      auto bad=options;
      if(invalid==0) bad.set_task(media::mediapb::SPEECH_TASK_TRANSLATE_TO_ENGLISH);
      if(invalid==1) bad.set_language("en");
      if(invalid==2) bad.set_model_id("wrong");
      grpc::ClientContext context; context.set_deadline(std::chrono::system_clock::now()+std::chrono::seconds(5));
      auto rpc=stub->Transcribe(&context); media::mediapb::TranscribeRequest request;
      request.mutable_open()->set_stream_id(media::uuid());
      *request.mutable_open()->mutable_options()=bad;
      check(rpc->Write(request),"invalid request write failed"); rpc->WritesDone();
      media::mediapb::TranscribeResponse response;
      check(!rpc->Read(&response) && rpc->Finish().error_code()==grpc::StatusCode::INVALID_ARGUMENT,
            "worker accepted unsupported Parakeet options");
    }
    grpc::ClientContext context; context.set_deadline(std::chrono::system_clock::now()+std::chrono::seconds(180));
    auto rpc=stub->Transcribe(&context); media::mediapb::TranscribeRequest request;
    request.mutable_open()->set_stream_id(media::uuid()); request.mutable_open()->set_track_id(7);
    *request.mutable_open()->mutable_options()=options;
    check(rpc->Write(request),"open failed"); media::mediapb::TranscribeResponse response;
    check(rpc->Read(&response) && response.has_opened(),"missing handshake");
    check(response.opened().producer().configuration_sha256().size()==32 &&
          response.opened().producer().build_id().size()==64,"missing handshake provenance");
    request.Clear(); auto* input=request.mutable_window();
    input->set_window_number(1); input->set_first_sample(16000); input->set_first_new_sample(16000);
    input->set_boundary(media::mediapb::INFERENCE_BOUNDARY_END);
    for(float sample:samples) {
      const auto bits=std::bit_cast<uint32_t>(sample);
      for(int b=0;b<4;++b) input->mutable_pcm_f32le()->push_back(static_cast<char>(bits>>(8*b)));
    }
    check(rpc->Write(request),"audio write failed");
    // Result must arrive before finish; successful inference also proves recovery after native aborts.
    check(rpc->Read(&response) && response.has_hypothesis(),"hypothesis buffered until finish");
    const auto hypothesis=response.hypothesis(); std::string text; unsigned tokens=0;
    check(hypothesis.sample_count()==samples.size() && hypothesis.has_encoder_context() &&
          hypothesis.encoder_context().SerializeAsString()==model.encoder_context(samples.size())->SerializeAsString(),
          "missing encoder context or changed source sample count");
    for(const auto& segment:hypothesis.segments()) {
      check(segment.track_id()==7 && segment.source_language().empty() && segment.output_language().empty() &&
            segment.task()==media::mediapb::SPEECH_TASK_TRANSCRIBE && !segment.final(),"invented language or wrong segment identity");
      text+=segment.text(); std::string rendered;
      for(const auto& token:segment.tokens()) {
        check(token.has_parakeet() && !token.has_whisper() && token.has_rendered_text() &&
              token.has_probability() && std::isfinite(token.probability()) &&
              token.probability()>=0 && token.probability()<=1 && token.has_start_us() &&
              token.start_us()==1000000+token.parakeet().start_centiseconds()*10000 &&
              token.end_us()==1000000+token.parakeet().end_centiseconds()*10000,
              "missing or incorrect native token metadata");
        rendered+=token.rendered_text(); ++tokens;
      }
      check(rendered==segment.text(),"native tokens do not reconstruct text");
    }
    check(tokens>0 && hypothesis.producer().model_sha256()==model.model_hash(),"missing output provenance");
    std::string normalized; bool space=false;
    for(unsigned char c:text) {
      if(c>='A' && c<='Z') c+='a'-'A';
      if(c>='a' && c<='z') {if(space && !normalized.empty()) normalized+=' '; normalized+=c;space=false;}
      else if(c==' ') space=true;
    }
    check(normalized=="and so my fellow americans ask not what your country can do for you ask what you can do for your country",
          "reference words differ");
    std::string reconciled;
    media::Reconciler reconciler([&](auto segment){if(segment.final()) reconciled+=segment.text();});
    reconciler.accept(hypothesis,0); reconciler.finish(); check(reconciled==text,"reconciliation lost text");
    request.Clear(); request.mutable_finish()->set_window_count(1);
    check(rpc->Write(request),"finish failed"); rpc->WritesDone();
    check(rpc->Read(&response) && response.has_closed() && response.closed().window_count()==1,"close accounting failed");
    check(!rpc->Read(&response) && rpc->Finish().ok(),"stream did not close successfully");
    std::cout<<"Parakeet adapter and gRPC qualification passed\n";
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
}
