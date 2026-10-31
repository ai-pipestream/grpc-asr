// Native backend qualification only. This is not a service adapter.
#include "decoder.h"
#include "reconcile.h"
#include "parakeet.h"
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
    check(argc==3, "usage: parakeet_probe MODEL JFK_AUDIO");
    auto weights = read(argv[1]);
    const auto source = read(argv[2]);
    const auto digest = media::sha256(weights);
    auto params = parakeet_context_default_params(); params.use_gpu = false;
    std::unique_ptr<parakeet_context, decltype(&parakeet_free)> model(
        parakeet_init_from_buffer_with_params_no_state(weights.data(), weights.size(), params), parakeet_free);
    check(model != nullptr, "native Parakeet buffer load failed");
    std::string().swap(weights);
    std::unique_ptr<parakeet_state, decltype(&parakeet_free_state)> state(parakeet_init_state(model.get()), parakeet_free_state);
    check(state != nullptr, "native Parakeet state allocation failed");
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
    auto infer = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
    infer.n_threads = 4; infer.no_context = true;
    std::atomic<bool> aborted{false};
    auto cancel = infer;
    cancel.abort_callback_user_data = &aborted;
    cancel.abort_callback = [](void* opaque) { static_cast<std::atomic<bool>*>(opaque)->store(true); return true; };
    check(parakeet_full_with_state(model.get(), state.get(), cancel, samples.data(), samples.size())!=0 && aborted,
          "native Parakeet ignored cancellation");
    // Reuse the same state after cancellation; no stale partial result may leak.
    check(parakeet_full_with_state(model.get(), state.get(), infer, samples.data(), samples.size())==0,
          "native Parakeet inference failed");
    media::mediapb::SpeechHypothesis hypothesis;
    hypothesis.set_window_number(1); hypothesis.set_sample_count(samples.size());
    std::string text;
    int tokens = 0;
    for (int s=0; s<parakeet_full_n_segments_from_state(state.get()); ++s) {
      const std::string native_text = parakeet_full_get_segment_text_from_state(state.get(), s);
      text += native_text;
      auto* segment = hypothesis.add_segments();
      segment->set_segment_id(media::uuid()); segment->set_revision(1);
      segment->set_text(native_text); segment->set_task(media::mediapb::SPEECH_TASK_TRANSCRIBE);
      std::string rendered;
      for (int t=0; t<parakeet_full_n_tokens_from_state(state.get(), s); ++t) {
        const auto token = parakeet_full_get_token_data_from_state(state.get(), s, t);
        check(token.t0>=0 && token.t1>=token.t0 && std::isfinite(token.p) && token.p>=0 && token.p<=1,
              "invalid native Parakeet token metadata");
        const char* raw = parakeet_full_get_token_text_from_state(model.get(), state.get(), s, t);
        const int size = parakeet_token_to_text(raw, t==0, nullptr, 0);
        check(size>=0 && size<65536, "unbounded native detokenized token");
        std::string contribution(size+1, '\0');
        check(parakeet_token_to_text(raw, t==0, contribution.data(), contribution.size())==size,
              "native detokenization size changed");
        contribution.resize(size); rendered += contribution;
        auto* typed = segment->add_tokens();
        typed->set_token_id(token.id); typed->set_text(raw); typed->set_rendered_text(contribution);
        typed->set_kind(media::mediapb::SPEECH_TOKEN_KIND_TEXT);
        typed->set_probability(token.p);
        typed->set_start_us(token.t0*10000); typed->set_end_us(token.t1*10000);
        ++tokens;
      }
      check(rendered==native_text, "native token rendering differs from segment text");
    }
    std::string reconciled;
    size_t preserved_tokens = 0;
    media::Reconciler reconciler([&](media::mediapb::TranscriptSegment segment) {
      if (!segment.final()) return;
      reconciled += segment.text();
      for (const auto& token : segment.tokens()) {
        check(token.has_rendered_text() && !token.text().empty(), "native token bytes lost in reconciliation");
        ++preserved_tokens;
      }
    });
    reconciler.accept(hypothesis, 0); reconciler.finish();
    check(reconciled==text && preserved_tokens==static_cast<size_t>(tokens),
          "native Parakeet reconciliation changed text or lost tokens");
    check(tokens>0 && digest.size()==32, "Parakeet probe lacks tokens or model hash");
    std::cout << text << '\n';
    std::string normalized; bool space = false;
    for (unsigned char c : text) {
      if (c>='A' && c<='Z') c += 'a'-'A';
      if (c>='a' && c<='z') { if (space && !normalized.empty()) normalized+=' '; normalized+=c; space=false; }
      else if (c==' ') space=true;
    }
    check(normalized=="and so my fellow americans ask not what your country can do for you ask what you can do for your country",
          "Parakeet reference words differ");
    std::cout << "Native Parakeet loading, cancellation recovery, reference transcription, token metadata and typed reconciliation passed\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
