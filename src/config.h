#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace asr {

// Process configuration, entirely from GRPC_ASR_* environment variables.
// Malformed values throw at startup; nothing is silently defaulted away
// from what the operator wrote.
struct Config {
    std::string listen_address = "0.0.0.0:50055";
    // cuda | openvino | cpu. CUDA is the default and fails loud when this
    // build or host cannot provide it; cpu is an explicit choice, never a
    // fallback.
    std::string backend = "cuda";
    int cuda_device = 0;
    std::string models_dir = "/models";
    // Model names to serve ("tiny.en", "large-v3", ...). Empty means
    // discover every ggml-*.bin in models_dir. Each name must resolve to
    // an existing weight file or startup fails.
    std::vector<std::string> models;
    // Concurrent transcriptions per model: whisper states created at
    // startup per loaded context.
    size_t concurrency = 2;
    size_t max_media_bytes = 256ULL * 1024 * 1024;
    size_t max_duration_seconds = 14400;
    // PCM window fed to whisper_full per iteration; bounds resident PCM.
    size_t window_seconds = 480;
    // whisper decode threads per transcription.
    size_t threads = 4;
    // Server default when TranscribeOptions.keyframe_interval_seconds is 0.
    size_t keyframe_interval_seconds = 10;
    // Whether the folded Document locates every word, not just every
    // segment: one ProvenanceItem per word on the segment's text item.
    // On by default; the cost is one provenance entry per word (roughly
    // 150 per transcribed minute), which a many-hour transcript may not
    // want to carry in one message. Words only exist when the client asked
    // for TranscribeOptions.word_timestamps, so this narrows that further,
    // it never adds work of its own.
    bool document_word_provenance = true;
    // 0 disables the stdout metrics line.
    size_t metrics_interval_seconds = 60;
    // ffmpeg/ffprobe children with no output for this long are killed.
    size_t tool_inactivity_seconds = 120;
    std::string ffmpeg = "ffmpeg";
    std::string ffprobe = "ffprobe";
};

// Reads and validates the environment. Throws std::invalid_argument with
// the variable name and accepted range on any malformed value.
Config load_config_from_env();

}  // namespace asr
