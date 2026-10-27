// The PCM scrub every decode path runs before handing samples to the model.

#include "media/pcm_scrub.h"

#include <cmath>
#include <limits>
#include <print>
#include <vector>

#include "fixture.h"

using asr::media::scrub_pcm;

namespace {

void verify_clean_samples_untouched() {
    std::vector<float> pcm = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 1e-30f};
    const std::vector<float> original = pcm;
    require(scrub_pcm(pcm) == 0, "in-range finite samples count as unchanged");
    require(pcm == original, "in-range finite samples keep their values");
}

void verify_nan_becomes_silence() {
    std::vector<float> pcm = {0.25f, std::nanf(""), -std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::signaling_NaN(), 0.25f};
    require(scrub_pcm(pcm) == 3, "every NaN pattern is reported");
    require(pcm[1] == 0.0f && pcm[2] == 0.0f && pcm[3] == 0.0f, "NaN decodes as silence");
    require(pcm[0] == 0.25f && pcm[4] == 0.25f, "neighbours are untouched");
}

void verify_out_of_range_clamps() {
    std::vector<float> pcm = {INFINITY, -INFINITY, 1e30f, -1e30f, 1.0001f, -1.0001f};
    require(scrub_pcm(pcm) == pcm.size(), "every out-of-range sample is reported");
    for (size_t i = 0; i < pcm.size(); i += 2) {
        require(pcm[i] == 1.0f, "sample " + std::to_string(i) + " clamps to 1");
        require(pcm[i + 1] == -1.0f, "sample " + std::to_string(i + 1) + " clamps to -1");
    }
}

void verify_empty_span() {
    std::vector<float> pcm;
    require(scrub_pcm(pcm) == 0, "an empty span is a no-op");
}

}  // namespace

int main() {
    try {
        verify_clean_samples_untouched();
        verify_nan_becomes_silence();
        verify_out_of_range_clamps();
        verify_empty_span();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("pcm-scrub-test passed");
    return 0;
}
