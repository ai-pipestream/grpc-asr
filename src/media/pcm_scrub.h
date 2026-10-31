#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

namespace asr::media {

// Enforces the model's PCM contract on samples that came off the wire:
// finite values in [-1, 1]. A float container (WAV format tag 3, a float
// PCM track in a mov) may hold any bit pattern, and neither miniaudio nor
// ffmpeg's f32le pipe normalizes it. whisper given a single NaN sample
// spins for minutes over a few seconds of audio and emits segments with
// negative timestamps, so every producer scrubs before the window loop.
// NaN becomes silence; everything else clamps to the range. Returns the
// number of samples that were changed, for logging and tests.
inline size_t scrub_pcm(std::span<float> samples) {
    size_t changed = 0;
    for (float& sample : samples) {
        if (std::isnan(sample)) {
            sample = 0.0f;
            changed++;
        } else if (sample < -1.0f || sample > 1.0f) {
            sample = std::clamp(sample, -1.0f, 1.0f);
            changed++;
        }
    }
    return changed;
}

}  // namespace asr::media
