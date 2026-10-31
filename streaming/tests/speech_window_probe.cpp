#include "speech.h"
#include "parakeet_model.h"
#include "reconcile.h"
#include "decoder.h"
#include <algorithm>
#include <bit>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>

std::string read(const char* path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("probe input missing");
  return {std::istreambuf_iterator<char>(file), {}};
}
int main(int argc, char** argv) {
  try {
    if (argc != 3 && argc != 4) throw std::runtime_error("usage: speech_window_probe MODEL AUDIO [whisper|parakeet]");
    const std::string backend=argc==4 ? argv[3] : "whisper";
    if(backend!="whisper" && backend!="parakeet") throw std::runtime_error("invalid probe backend");
    const auto weights = read(argv[1]), source = read(argv[2]);
    std::unique_ptr<media::SpeechBackend> model;
    if(backend=="parakeet") model=std::make_unique<media::ParakeetModel>(weights,"probe",4);
    else model=std::make_unique<media::SpeechModel>(weights,"probe",4);
    std::vector<float> samples;
    size_t offset = 0;
    media::mediapb::DecodeOptions decode; decode.set_emit_audio(true);
    media::decode_media([&](uint8_t* out, int cap) {
      const auto n = std::min(source.size() - offset, static_cast<size_t>(cap));
      std::memcpy(out, source.data() + offset, n); offset += n; return static_cast<int>(n);
    }, [&](media::mediapb::StreamEvent event) {
      if (!event.has_audio()) return;
      const auto& pcm = event.audio().pcm_f32le();
      for (size_t i = 0; i < pcm.size(); i += 4) {
        uint32_t bits = 0;
        for (int b = 0; b < 4; ++b) bits |= static_cast<uint32_t>(static_cast<unsigned char>(pcm[i+b])) << (8*b);
        samples.push_back(std::bit_cast<float>(bits));
      }
    }, [] { return false; }, decode);
    media::mediapb::SpeechOptions options; options.set_model_id("probe"); if(backend=="whisper") options.set_language("en"); options.set_word_timestamps(true);
    std::string finalized;
    media::Reconciler reconciler([&](media::mediapb::TranscriptSegment segment) {
      if (!segment.final()) return;
      if (segment.reconciliation() == media::mediapb::RECONCILIATION_STATUS_UNRESOLVED)
        throw std::runtime_error("reference overlap unresolved");
      finalized += segment.text();
    });
    uint64_t number = 0, previous_end = 0;
    for (size_t first = 0; first < samples.size(); first += 48000) {
      const auto end = std::min(first + 64000, samples.size());
      if (end <= previous_end) break;
      media::SpeechWindow window{++number, first, previous_end, end == samples.size() ? media::WindowBoundary::End : media::WindowBoundary::Fixed,
                                 {samples.begin() + first, samples.begin() + end}};
      previous_end = end;
      std::cout << "WINDOW " << number << ' ' << first << ' ' << end << '\n';
      media::mediapb::SpeechHypothesis hypothesis;
      hypothesis.set_window_number(number); hypothesis.set_first_sample(first);
      hypothesis.set_first_new_sample(window.first_new_sample); hypothesis.set_sample_count(end-first);
      if (backend=="parakeet" && std::getenv("MEDIA_PROBE_FRESH_STATE"))
        model=std::make_unique<media::ParakeetModel>(weights,"probe",4);
      for (const auto& segment : model->infer(window, 0, options, [] { return false; })) {
        *hypothesis.add_segments() = segment;
        std::cout << "SEGMENT " << segment.start_us() << ' ' << segment.end_us() << ' ' << std::quoted(segment.text()) << '\n';
        for (const auto& token : segment.tokens()) {
          if (token.kind() != media::mediapb::SPEECH_TOKEN_KIND_TEXT) continue;
          std::cout << "TOKEN " << token.token_id() << ' ' << token.start_us() << ' ' << token.end_us() << ' '
                    << std::quoted(token.text()) << '\n';
        }
      }
      reconciler.accept(hypothesis, end == samples.size() ? 0 : 16000);
    }
    reconciler.finish();
    std::string normalized;
    bool space = false;
    for (unsigned char c : finalized) {
      if (c >= 'A' && c <= 'Z') c += 'a'-'A';
      if (c >= 'a' && c <= 'z') { if(space && !normalized.empty()) normalized+=' '; normalized+=c; space=false; }
      else if(c==' ') space=true;
    }
    const std::string expected="and so my fellow americans ask not what your country can do for you ask what you can do for your country";
    std::cout << "FINAL " << normalized << '\n';
    if (normalized != expected) throw std::runtime_error("reference transcript omitted or repeated boundary words");
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
