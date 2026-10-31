#pragma once

#include "registry.h"

#include <cstdint>
#include <functional>
#include <string>

namespace media {
// The callback returns source bytes, zero at explicit EOF, or throws. Native C
// callbacks trap exceptions and rethrow them only after returning to C++.
using ReadMedia = std::function<int(uint8_t*, int)>;
using EmitMedia = std::function<void(mediapb::StreamEvent)>;

std::string decoder_version();
void decode_media(ReadMedia read, EmitMedia emit, std::function<bool()> cancelled,
                  const mediapb::DecodeOptions& options);
}  // namespace media
