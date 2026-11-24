#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string_view>
namespace media {
// The handlers that answer an exception which is not a Failure with INTERNAL
// and a fixed message. The message is fixed so that nothing internal reaches
// a client; the cause must then be said somewhere else, or an inference error
// (a ggml allocation failing on the GPU, say) is seen by nobody.
enum class FailureSite : size_t {
  Pull = 0,
  IngestProcessing,
  IngestFinalization,
  Subscribe,
  SpeechWorker,
};
inline constexpr std::array<std::string_view, 5> failure_site_names = {
    "pull", "ingest_processing", "ingest_finalization", "subscribe", "speech_worker"};
// Says the cause the client is not told: one stderr line with the handler,
// the stream it was serving ("-" before a stream exists) and the exception's
// text, and one count for the handler. Then calls the observer, if any.
void report_unexpected_failure(FailureSite site, std::string_view stream_id,
                               const std::exception& error) noexcept;
// How many unexpected failures a handler has reported in this process.
uint64_t unexpected_failures(FailureSite site) noexcept;
// Called after every report, so the process that hosts these services can
// count them in its own metrics. One observer; null removes it.
using FailureObserver = void (*)(FailureSite site) noexcept;
void set_failure_observer(FailureObserver observer) noexcept;
}  // namespace media
