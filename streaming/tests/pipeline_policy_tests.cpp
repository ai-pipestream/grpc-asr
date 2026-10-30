#include "pipeline.h"
#include "service.h"
#include <iostream>
#include <stdexcept>
#include <tuple>

int main() {
  try {
    media::SpeechPipelineFactory factory("127.0.0.1:1");
    media::Registry registry;
    media::Service service(registry,&factory);
    media::mediapb::GetServiceInfoRequest empty;
    media::mediapb::GetServiceInfoResponse info;
    if (!service.GetServiceInfo(nullptr,&empty,&info).ok() ||
        !info.has_minimum_window_advance_ms() || info.minimum_window_advance_ms()!=100)
      throw std::runtime_error("minimum advance missing from discovery");
    for (const auto& [window,overlap,expected] : {
        std::tuple{100u,0u,true}, {1000u,900u,true}, {30000u,29900u,true},
        {99u,0u,false}, {1000u,901u,false}, {30000u,29999u,false}}) {
      media::mediapb::OpenStream request;
      auto* speech=request.add_speech(); speech->set_model_id("policy-only");
      speech->mutable_windowing()->set_window_ms(window);
      speech->mutable_windowing()->set_overlap_ms(overlap);
      bool accepted=true;
      try {factory.resolve(request);}
      catch(const media::Failure& error) {
        if(error.code!=grpc::StatusCode::INVALID_ARGUMENT) throw;
        accepted=false;
      }
      if(accepted!=expected) throw std::runtime_error("window advance policy mismatch: "+std::to_string(window-overlap));
    }
    std::cout<<"6 configured window advance cases passed\n";
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
