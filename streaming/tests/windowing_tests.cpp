#include "windowing.h"
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace media;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void add(Windowing& windowing, int count, bool speech, int& position) {
  std::vector<float> pcm;
  for (int i = 0; i < count; ++i) pcm.push_back(static_cast<float>(position++));
  windowing.append(pcm, speech);
}
void coverage(const std::vector<SpeechWindow>& windows, uint64_t total) {
  uint64_t end = 0, number = 0;
  for (const auto& w : windows) {
    check(w.number == ++number, "noncontiguous window numbers");
    check(w.first_new_sample == end && w.first_sample <= end, "lost or repeated new audio");
    check(w.first_sample + w.samples.size() > end, "overlap-only window");
    for (size_t i = 0; i < w.samples.size(); ++i)
      check(w.samples[i] == static_cast<float>(w.first_sample + i), "audio samples changed");
    end = w.first_sample + w.samples.size();
  }
  check(end == total, "unflushed final audio");
}
int main() {
  try {
    for (int chunk = 1; chunk <= 13; ++chunk) {
      std::vector<SpeechWindow> output;
      Windowing w({10, 10, 3, 0, 0, false, false}, [&](auto x) { output.push_back(std::move(x)); });
      int pos = 0;
      while (pos < 45) add(w, std::min(chunk, 45 - pos), true, pos);
      w.finish(); coverage(output, 45);
      check(output.front().boundary == WindowBoundary::Fixed, "fixed cut missing");
    }
    {
      std::vector<SpeechWindow> output;
      Windowing w({10, 20, 3, 6, 2, true, false}, [&](auto x) { output.push_back(std::move(x)); });
      int pos = 0;
      add(w, 9, true, pos); add(w, 5, false, pos);
      check(output.empty(), "pause published before confirmation");
      add(w, 1, false, pos);
      check(output.size() == 1 && output[0].samples.size() == 11 && output[0].boundary == WindowBoundary::Pause,
            "pause margin cut wrong");
      add(w, 8, true, pos); w.finish(); coverage(output, 23);
      check(output[1].first_sample == 11, "normal pause repeated audio");
    }
    {
      std::vector<SpeechWindow> output;
      Windowing w({10, 20, 3, 6, 2, true, false}, [&](auto x) { output.push_back(std::move(x)); });
      int pos = 0; add(w, 45, true, pos); w.finish(); coverage(output, 45);
      check(output[0].boundary == WindowBoundary::Forced && output[1].first_sample == 17, "forced overlap absent");
    }
    {
      Windowing w({10, 20, 0, 6, 2, true, true}, [](auto) {});
      int pos = 0; bool failed = false;
      try { add(w, 20, true, pos); } catch (const std::length_error&) { failed = true; }
      check(failed && w.buffered_samples() == 20, "no-pause failure not bounded");
      failed = false;
      try { add(w, 1, true, pos); } catch (const std::logic_error&) { failed = true; }
      check(failed && w.buffered_samples() == 20, "failed stream accepted more audio");
    }
    {
      std::vector<SpeechWindow> output;
      Windowing w({10, 10, 3, 0, 0, false, false}, [&](auto x) { output.push_back(std::move(x)); });
      int pos = 0; add(w, 10, true, pos); w.discontinuity(); add(w, 4, true, pos); w.finish();
      coverage(output, 14); check(output.size() == 2 && output[1].first_sample == 10, "overlap crossed discontinuity");
    }
    {
      std::vector<SpeechWindow> output;
      Windowing w({10, 20, 3, 6, 2, true, false}, [&](auto x) { output.push_back(std::move(x)); });
      int pos = 0; add(w, 10000, false, pos); w.finish(); coverage(output, 10000);
      check(w.buffered_samples() == 0, "closed buffer retained");
      for (const auto& item : output) check(item.samples.size() <= 20, "silence grew without bound");
    }
    {
      std::vector<SpeechWindow> output;
      Windowing w({10, 20, 3, 6, 2, true, false}, [&](auto x) { output.push_back(std::move(x)); });
      int pos = 0;
      add(w, 9, true, pos); add(w, 5, false, pos); add(w, 6, true, pos);
      check(output.size() == 1 && output[0].boundary == WindowBoundary::Forced,
            "speech resuming during confirmation was cut as a pause");
      w.finish(); coverage(output, 20);
      check(output.size() == 1, "finish duplicated overlap-only tail");
    }
    std::cout << "7 speech window tests passed\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
