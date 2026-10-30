#pragma once
#include <functional>
#include <memory>
#include <span>
#include <string>

namespace media {
// One detector per source track. Inputs are contiguous mono 16 kHz samples.
// Callbacks contain only real samples; model padding is never emitted.
class VoiceActivity {
 public:
  using Emit = std::function<void(std::span<const float>, float probability)>;
  VoiceActivity(std::span<const char> model, Emit emit);
  ~VoiceActivity();
  void append(std::span<const float> samples);
  void finish();
  void discontinuity();
  const std::string& model_hash() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace media
