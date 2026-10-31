#pragma once
#include "registry.h"
#include <functional>

namespace media {
// Per track/task, bounded reconciliation of window hypotheses. Raw hypotheses
// remain authoritative and must be published separately by the caller.
class Reconciler {
 public:
  using Emit = std::function<void(mediapb::TranscriptSegment)>;
  explicit Reconciler(Emit emit);
  void accept(const mediapb::SpeechHypothesis& hypothesis, uint32_t next_overlap_samples);
  void finish();
 private:
  void output(size_t count, bool final, mediapb::ReconciliationStatus status);
  Emit emit_;
  std::vector<mediapb::SpeechToken> pending_;
  mediapb::TranscriptSegment metadata_;
  std::string id_;
  uint64_t revision_ = 0, window_ = 0, end_sample_ = 0;
  int64_t end_time_ = 0;
  uint32_t track_ = 0;
  bool uncertain_ = false, closed_ = false;
  bool have_final_text_ = false, final_ends_space_ = false;
};
}  // namespace media
