#include "failure_report.h"
#include <atomic>
#include <cstdio>
#include <string>
namespace media {
namespace {
std::array<std::atomic<uint64_t>, failure_site_names.size()> counts{};
std::atomic<FailureObserver> observer{nullptr};
// Log text is a line: control bytes in an exception's message must not start
// a new one.
std::string one_line(std::string_view text, size_t limit) {
  std::string line(text.substr(0, limit));
  for (char& c : line)
    if (static_cast<unsigned char>(c) < 0x20 || c == 0x7F) c = ' ';
  return line;
}
}  // namespace
void report_unexpected_failure(FailureSite site, std::string_view stream_id,
                               const std::exception& error) noexcept {
  const auto index = static_cast<size_t>(site);
  if (index >= counts.size()) return;
  counts[index].fetch_add(1, std::memory_order_relaxed);
  try {
    const std::string line = "grpc-asr streaming: " + std::string(failure_site_names[index]) +
        " failed with an unexpected error (stream " +
        (stream_id.empty() ? std::string("-") : one_line(stream_id, 64)) + "): " +
        one_line(error.what(), 1024) + "\n";
    std::fputs(line.c_str(), stderr);
  } catch (...) {
    std::fputs("grpc-asr streaming: a handler failed with an unexpected error\n", stderr);
  }
  if (const FailureObserver notify = observer.load(std::memory_order_acquire)) notify(site);
}
uint64_t unexpected_failures(FailureSite site) noexcept {
  const auto index = static_cast<size_t>(site);
  return index < counts.size() ? counts[index].load(std::memory_order_relaxed) : 0;
}
void set_failure_observer(FailureObserver next) noexcept {
  observer.store(next, std::memory_order_release);
}
}  // namespace media
