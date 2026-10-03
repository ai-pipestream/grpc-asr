#pragma once

#include <stdexcept>

namespace asr {

// Thrown when a stream's stop token fires mid-work: the client cancelled,
// its deadline passed, or its upload failed. Media children, the model
// pool's wait, and the decoder all stop with it. The service maps it to
// CANCELLED; the client is already gone, so the status only feeds the
// metrics.
class Cancelled : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

}  // namespace asr
