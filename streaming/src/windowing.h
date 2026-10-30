#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace media {
enum class WindowBoundary { Fixed, Pause, Forced, End, Discontinuity };
struct WindowPolicy {
  uint32_t target_samples = 0;
  uint32_t max_samples = 0;
  uint32_t overlap_samples = 0;
  uint32_t min_pause_samples = 0;
  uint32_t padding_samples = 0;
  bool speech_aware = false;
  bool fail_without_pause = false;
};
struct SpeechWindow {
  uint64_t number;
  uint64_t first_sample;
  uint64_t first_new_sample;
  WindowBoundary boundary;
  std::vector<float> samples;
};
// The caller supplies actual VAD classifications, not an energy approximation.
// Times and policy lengths are in mono PCM samples at the worker's input rate.
class Windowing {
 public:
  using Emit = std::function<void(SpeechWindow)>;
  Windowing(WindowPolicy policy, Emit emit);
  void append(std::span<const float> samples, bool speech);
  void finish();
  void discontinuity();
  size_t buffered_samples() const { return buffer_.size(); }
 private:
  void emit(size_t count, WindowBoundary boundary, size_t overlap);
  WindowPolicy policy_;
  Emit emit_;
  std::vector<float> buffer_;
  uint64_t first_ = 0, received_ = 0, committed_ = 0, number_ = 0;
  uint64_t silence_start_ = 0;
  bool in_silence_ = false, finished_ = false;
};
}  // namespace media
