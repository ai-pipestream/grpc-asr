#include "reconcile.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>

namespace media {
namespace {
constexpr size_t token_limit = 2048, text_limit = 128 * 1024;
constexpr int64_t tolerance = 350000;
int64_t micros(uint64_t sample) {
  return static_cast<int64_t>(sample / 16000 * 1000000 + sample % 16000 * 1000000 / 16000);
}
const std::string& rendered(const mediapb::SpeechToken& token) {
  return token.has_rendered_text() ? token.rendered_text() : token.text();
}
size_t token_bytes(const mediapb::SpeechToken& token) {
  return token.text().size() + (token.has_rendered_text() ? token.rendered_text().size() : 0);
}
std::string key(const std::string& text) {
  std::string result;
  for (unsigned char c : text) {
    if (c >= 'A' && c <= 'Z') result += static_cast<char>(c + ('a' - 'A'));
    else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c >= 128) result += static_cast<char>(c);
  }
  return result;
}
bool utf8(const std::string& text) {
  size_t i = 0;
  while (i < text.size()) {
    const auto lead = static_cast<unsigned char>(text[i++]);
    if (lead < 128) continue;
    int n; uint32_t value, minimum;
    if (lead >= 0xc2 && lead <= 0xdf) { n = 1; value = lead & 31; minimum = 0x80; }
    else if (lead >= 0xe0 && lead <= 0xef) { n = 2; value = lead & 15; minimum = 0x800; }
    else if (lead >= 0xf0 && lead <= 0xf4) { n = 3; value = lead & 7; minimum = 0x10000; }
    else return false;
    if (i + n > text.size()) return false;
    while (n--) {
      const auto c = static_cast<unsigned char>(text[i++]);
      if ((c & 0xc0) != 0x80) return false;
      value = (value << 6) | (c & 63);
    }
    if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) return false;
  }
  return true;
}
bool timed_match(const mediapb::SpeechToken& a, const mediapb::SpeechToken& b) {
  return std::max(a.start_us(), b.start_us()) - std::min(a.end_us(), b.end_us()) <= tolerance;
}
struct Run { size_t score = 0, count = 0; long double distance = 0; };
struct Match { size_t old_start = 0, new_start = 0; Run run; bool ambiguous = false; };
Match anchor(const std::vector<mediapb::SpeechToken>& old, const std::vector<mediapb::SpeechToken>& fresh, int64_t end) {
  std::vector<std::string> a, b;
  for (const auto& t : old) a.push_back(key(rendered(t)));
  for (const auto& t : fresh) b.push_back(key(rendered(t)));
  std::vector<Run> next(fresh.size() + 1), row(fresh.size() + 1);
  Match best;
  for (size_t i = old.size(); i-- > 0;) {
    for (size_t j = fresh.size(); j-- > 0;) {
      row[j] = {};
      if (a[i] != b[j] || (!a[i].empty() && !timed_match(old[i], fresh[j])) || fresh[j].start_us() > end + tolerance) continue;
      row[j] = next[j + 1]; ++row[j].count; row[j].score += a[i].size();
      row[j].distance += std::abs((static_cast<long double>(old[i].start_us()) + old[i].end_us()) -
                                  (static_cast<long double>(fresh[j].start_us()) + fresh[j].end_us())) / 2;
      const auto& candidate = row[j];
      if (candidate.score > best.run.score ||
          (candidate.score == best.run.score && candidate.count > best.run.count)) {
        best = {i, j, candidate, false};
      } else if (candidate.score == best.run.score && candidate.count == best.run.count && candidate.score) {
        const auto delta = candidate.distance / candidate.count - best.run.distance / best.run.count;
        if (delta < -30000) best = {i, j, candidate, false};
        else if (std::abs(delta) <= 30000) best.ambiguous = true;
      }
    }
    std::swap(row, next);
  }
  return best;
}
std::string joined_key(const std::vector<mediapb::SpeechToken>& tokens, size_t begin, size_t end) {
  std::string result;
  for (size_t i = begin; i < end; ++i) result += key(rendered(tokens[i]));
  return result;
}
}
Reconciler::Reconciler(Emit emit) : emit_(std::move(emit)) {
  if (!emit_) throw std::invalid_argument("reconciliation output required");
}
void Reconciler::output(size_t count, bool final, mediapb::ReconciliationStatus status) {
  if (!count) return;
  mediapb::TranscriptSegment segment = metadata_;
  segment.clear_tokens(); segment.clear_text(); segment.clear_average_log_probability();
  if (id_.empty()) { id_ = uuid(); revision_ = 0; }
  segment.set_segment_id(id_); segment.set_revision(++revision_); segment.set_final(final);
  segment.set_reconciliation(status);
  int64_t start = std::numeric_limits<int64_t>::max(), end = std::numeric_limits<int64_t>::min();
  double log_sum = 0; bool has_logs = true;
  size_t output_bytes = 0;
  for (size_t i = 0; i < count; ++i) {
    const auto& token = pending_[i];
    auto* emitted = segment.add_tokens(); *emitted = token;
    const auto& piece = rendered(token);
    const bool needs_separator = segment.text().empty() ? have_final_text_ && !final_ends_space_ :
        !std::isspace(static_cast<unsigned char>(segment.text().back()));
    if (token.has_parakeet() && token.parakeet().is_word_start() && needs_separator &&
        !piece.empty() && !std::isspace(static_cast<unsigned char>(piece.front())))
      emitted->set_rendered_text(" " + piece);
    output_bytes += token_bytes(*emitted);
    if (output_bytes > text_limit)
      throw Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "reconciled token text limit exceeded");
    *segment.mutable_text() += rendered(*emitted);
    start = std::min(start, token.start_us()); end = std::max(end, token.end_us());
    has_logs &= token.has_whisper();
    if (token.has_whisper()) log_sum += token.whisper().log_probability();
  }
  if (!utf8(segment.text())) throw Failure(grpc::StatusCode::DATA_LOSS, "reconciled text is not complete UTF-8");
  segment.set_start_us(start); segment.set_end_us(end);
  if (has_logs) segment.set_average_log_probability(log_sum / count);
  const bool has_text = !segment.text().empty();
  const bool ends_space = has_text && std::isspace(static_cast<unsigned char>(segment.text().back()));
  emit_(std::move(segment));
  if (final && has_text) { have_final_text_ = true; final_ends_space_ = ends_space; }
  if (final) { pending_.erase(pending_.begin(), pending_.begin() + count); id_.clear(); revision_ = 0; }
}
void Reconciler::accept(const mediapb::SpeechHypothesis& h, uint32_t overlap) {
  if (closed_) throw std::logic_error("hypothesis after reconciliation close");
  try {
  if (window_ && h.track_id() != track_)
    throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "reconciliation track changed");
  if (h.window_number() != window_ + 1 || !h.sample_count() || h.sample_count() > 480000 ||
      h.first_sample() > UINT64_C(100000000000000) || overlap >= h.sample_count() ||
      h.first_new_sample() < h.first_sample() || h.first_new_sample() >= h.first_sample() + h.sample_count() ||
      (window_ && h.first_new_sample() != end_sample_) || (!window_ && h.first_new_sample() != h.first_sample()))
    throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "invalid reconciliation window sequence");
  const int64_t start = h.has_presentation_time_us() ? h.presentation_time_us() : micros(h.first_sample());
  if (start < -(INT64_C(1) << 62) || start > (INT64_C(1) << 62))
    throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "reconciliation timestamp exceeds limit");
  const auto end = start + micros(h.sample_count());
  std::vector<mediapb::SpeechToken> fresh;
  mediapb::TranscriptSegment metadata;
  bool invalid_timing = false, has_metadata = false;
  size_t text_bytes = 0;
  for (const auto& segment : h.segments()) {
    if (segment.track_id() != h.track_id() ||
        (has_metadata && (segment.task() != metadata.task() ||
         segment.source_language() != metadata.source_language() ||
         segment.output_language() != metadata.output_language())))
      throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "inconsistent hypothesis segment identity");
    if (!has_metadata) { metadata = segment; has_metadata = true; }
    std::string text;
    for (const auto& token : segment.tokens()) {
      if (token.kind() == mediapb::SPEECH_TOKEN_KIND_CONTROL) continue;
      if (token.kind() != mediapb::SPEECH_TOKEN_KIND_TEXT || !token.has_start_us() || !token.has_end_us() || token.end_us() < token.start_us())
        throw Failure(grpc::StatusCode::FAILED_PRECONDITION, "reconciliation requires classified lexical tokens with timings");
      fresh.push_back(token); text += rendered(token); text_bytes += token_bytes(token);
      if (fresh.size() > token_limit || text_bytes > text_limit)
        throw Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "reconciliation context limit exceeded");
      if (token.start_us() < start || token.end_us() > end) {
        invalid_timing = true;
        fresh.back().set_start_us(std::clamp(token.start_us(), start, end));
        fresh.back().set_end_us(std::clamp(token.end_us(), start, end));
      }
    }
    if (text != segment.text() || !utf8(text)) throw Failure(grpc::StatusCode::DATA_LOSS, "lexical tokens do not reconstruct native text");
  }
  auto status = mediapb::RECONCILIATION_STATUS_DIRECT;
  if (!pending_.empty()) {
    const bool beyond_previous_window = !invalid_timing && !fresh.empty() &&
        std::all_of(fresh.begin(), fresh.end(), [&](const auto& token) {
          return token.start_us() > end_time_ + tolerance;
        });
    if (h.first_sample() >= h.first_new_sample() || beyond_previous_window || fresh.empty()) {
      output(pending_.size(), true, uncertain_ ? mediapb::RECONCILIATION_STATUS_UNRESOLVED : mediapb::RECONCILIATION_STATUS_DIRECT);
      uncertain_ = false;
    } else {
      const auto match = anchor(pending_, fresh, end_time_);
      bool safe = match.run.score >= 2 && !match.ambiguous && !fresh.empty() &&
          metadata.source_language() == metadata_.source_language() && metadata.output_language() == metadata_.output_language() && metadata.task() == metadata_.task();
      const size_t a_end = match.old_start + match.run.count, b_end = match.new_start + match.run.count;
      if (safe && match.old_start && match.new_start)
        safe = joined_key(pending_, 0, match.old_start) == joined_key(fresh, 0, match.new_start);
      if (safe && a_end < pending_.size()) {
        const auto old_tail = joined_key(pending_, a_end, pending_.size());
        const auto new_tail = joined_key(fresh, b_end, fresh.size());
        safe = !old_tail.empty() && new_tail.starts_with(old_tail) && pending_.back().end_us() >= end_time_ - 500000;
      }
      if (safe) {
        std::vector<mediapb::SpeechToken> merged;
        merged.insert(merged.end(), pending_.begin(), pending_.begin() + match.old_start);
        // If no old prefix exists, keep new words preceding the anchor too.
        merged.insert(merged.end(), fresh.begin() + (match.old_start ? match.new_start : 0), fresh.end());
        fresh = std::move(merged); status = mediapb::RECONCILIATION_STATUS_ALIGNED;
      } else {
        output(pending_.size(), true, mediapb::RECONCILIATION_STATUS_UNRESOLVED);
        status = mediapb::RECONCILIATION_STATUS_UNRESOLVED;
      }
    }
  }
  pending_ = std::move(fresh); metadata_ = metadata;
  if (pending_.size() > token_limit) throw Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "reconciliation context limit exceeded");
  if (invalid_timing || uncertain_) status = mediapb::RECONCILIATION_STATUS_UNRESOLVED;
  size_t merged_bytes = 0;
  for (const auto& token : pending_) merged_bytes += token_bytes(token);
  if (merged_bytes > text_limit) throw Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "reconciliation text limit exceeded");
  size_t stable = 0, safe_boundary = 0, word_start = 0; std::string prefix;
  const auto cutoff = end - micros(overlap);
  for (const auto& token : pending_) {
    if (overlap && token.end_us() > cutoff) break;
    if (token.has_parakeet() && token.parakeet().is_word_start()) word_start = stable;
    ++stable; prefix += rendered(token); if (utf8(prefix)) safe_boundary = stable;
  }
  // Keep the entire native word revisable when its later pieces overlap.
  // Otherwise the next window repeats the already-finalized word prefix.
  if (stable < pending_.size() && pending_[stable].has_parakeet() &&
      !pending_[stable].parakeet().is_word_start())
    safe_boundary = std::min(safe_boundary, word_start);
  output(safe_boundary, true, status);
  uncertain_ = !pending_.empty() && status == mediapb::RECONCILIATION_STATUS_UNRESOLVED;
  if (!pending_.empty()) output(pending_.size(), false, mediapb::RECONCILIATION_STATUS_PENDING);
  track_ = h.track_id();
  ++window_; end_sample_ = h.first_sample() + h.sample_count(); end_time_ = end;
  } catch (...) { closed_ = true; throw; }
}
void Reconciler::finish() {
  if (closed_) throw std::logic_error("reconciliation already closed");
  closed_ = true;
  output(pending_.size(), true, uncertain_ ? mediapb::RECONCILIATION_STATUS_UNRESOLVED : mediapb::RECONCILIATION_STATUS_DIRECT);
}
}  // namespace media
