#include "windowing.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace media {
Windowing::Windowing(WindowPolicy policy, Emit emit) : policy_(policy), emit_(std::move(emit)) {
  if (!emit_ || !policy.target_samples || policy.max_samples < policy.target_samples ||
      policy.max_samples > 16000 * 30 || policy.overlap_samples >= policy.target_samples)
    throw std::invalid_argument("invalid bounded speech window policy");
  if (policy.speech_aware &&
      (!policy.min_pause_samples || policy.min_pause_samples > policy.max_samples ||
       static_cast<uint64_t>(policy.padding_samples) * 2 >= policy.max_samples ||
       (!policy.fail_without_pause && !policy.overlap_samples)))
    throw std::invalid_argument("invalid pause boundary policy");
  buffer_.reserve(policy.max_samples);
}
void Windowing::emit(size_t count, WindowBoundary boundary, size_t overlap) {
  if (!count || first_ + count <= committed_) return;
  const auto end = first_ + count;
  SpeechWindow window{++number_, first_, committed_, boundary,
                      {buffer_.begin(), buffer_.begin() + count}};
  emit_(std::move(window));
  committed_ = end;
  const auto consume = count - overlap;
  buffer_.erase(buffer_.begin(), buffer_.begin() + consume);
  first_ += consume;
}
void Windowing::append(std::span<const float> samples, bool speech) {
  if (finished_) throw std::logic_error("audio after window close");
  for (float sample : samples) {
    if (!std::isfinite(sample)) throw std::invalid_argument("nonfinite PCM sample");
    if (speech) in_silence_ = false;
    else if (!in_silence_) { silence_start_ = received_; in_silence_ = true; }
    buffer_.push_back(sample);
    ++received_;
    if (!policy_.speech_aware) {
      if (buffer_.size() == policy_.target_samples)
        emit(buffer_.size(), WindowBoundary::Fixed, policy_.overlap_samples);
      continue;
    }
    if (in_silence_ && received_ - silence_start_ >= policy_.min_pause_samples) {
      const auto cut = std::max(first_ + policy_.target_samples,
                                silence_start_ + policy_.padding_samples);
      // Both margins have been observed before publishing the boundary.
      if (cut > first_ && cut + policy_.padding_samples <= received_) {
        emit(cut - first_, WindowBoundary::Pause, 0);
      }
    }
    if (buffer_.size() >= policy_.max_samples) {
      if (policy_.fail_without_pause) {
        finished_ = true;
        throw std::length_error("no speech pause before hard window limit");
      }
      emit(buffer_.size(), WindowBoundary::Forced, policy_.overlap_samples);
    }
  }
}
void Windowing::finish() {
  if (finished_) throw std::logic_error("window stream already closed");
  emit(buffer_.size(), WindowBoundary::End, 0);
  buffer_.clear();
  finished_ = true;
}
void Windowing::discontinuity() {
  if (finished_) throw std::logic_error("discontinuity after close");
  emit(buffer_.size(), WindowBoundary::Discontinuity, 0);
  buffer_.clear();
  first_ = received_;
  in_silence_ = false;
}
}  // namespace media
