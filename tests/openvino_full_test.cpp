// GRPC_ASR_BACKEND=openvino, startup contract and the full-graph mode.
//
// In a build without OpenVINO: asking for the backend fails loud, in both
// modes, before any model is needed.
//
// In an OpenVINO build: a device OpenVINO cannot open fails loud (the
// negative control for "no render node"), and full mode on an OpenVINO
// device transcribes the JFK sample with every Whisper compute node on that
// device, within a few words of the known sentence and of what the ggml CPU
// backend hears. The device is the OpenVINO CPU one unless
// GRPC_ASR_TEST_OPENVINO_DEVICE names another: the CPU device stands in for
// the GPU in image builds, where the graph translation and the placement
// check are the same code and no host has an Intel GPU. On a host with the
// card, GRPC_ASR_TEST_OPENVINO_DEVICE=GPU runs the same case on it, and
// scripts/test/gpu-placement-test.sh checks the shipped server.
//
// The full-mode case needs GRPC_ASR_TEST_MODEL, GRPC_ASR_TEST_SAMPLE and the
// chosen OpenVINO device, and skips with 77 without them.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.h"
#include "engine/model_pool.h"
#include "engine/openvino_devices.h"
#include "fixture.h"
#include "metrics.h"
#include "whisper.h"

using asr::Config;
using asr::engine::ModelPool;

namespace {

// The message a ModelPool for this config refuses to start with; empty when
// it starts.
std::string startup_failure(const Config& config) {
    try {
        ModelPool pool(config);
    } catch (const std::exception& error) {
        return error.what();
    }
    return "";
}

// 16 kHz mono 16-bit PCM WAV to whisper's float samples.
std::vector<float> pcm_from_wav(const std::string& wav) {
    const size_t data = wav.find("data");
    require(data != std::string::npos && data + 8 <= wav.size(), "sample is a WAV file");
    uint32_t bytes = 0;
    std::memcpy(&bytes, wav.data() + data + 4, sizeof(bytes));
    bytes = std::min<uint32_t>(bytes, static_cast<uint32_t>(wav.size() - data - 8));
    std::vector<float> samples(bytes / 2);
    for (size_t i = 0; i < samples.size(); i++) {
        int16_t value = 0;
        std::memcpy(&value, wav.data() + data + 8 + 2 * i, sizeof(value));
        samples[i] = static_cast<float>(value) / 32768.0f;
    }
    return samples;
}

std::string transcribe(ModelPool& pool, const std::vector<float>& samples, int beam_size) {
    ModelPool::Lease lease = pool.acquire("tiny.en");
    whisper_full_params params = whisper_full_default_params(
        beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
    params.n_threads = 4;
    params.language = "en";
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.print_special = false;
    if (beam_size > 1) {
        params.beam_search.beam_size = beam_size;
    }
    require(whisper_full_with_state(lease.context(), lease.state(), params, samples.data(),
                                    static_cast<int>(samples.size())) == 0,
            "transcription on the OpenVINO backend succeeds");
    std::string text;
    for (int i = 0; i < whisper_full_n_segments_from_state(lease.state()); i++) {
        text += whisper_full_get_segment_text_from_state(lease.state(), i);
    }
    std::ranges::transform(text, text.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

// Lowercase words with punctuation dropped, so "you, ask" and "you ask"
// compare equal.
std::vector<std::string> words_of(const std::string& text) {
    std::string plain;
    for (const unsigned char c : text) {
        plain += std::isalnum(c) != 0 ? static_cast<char>(std::tolower(c)) : ' ';
    }
    std::istringstream stream(plain);
    std::vector<std::string> words;
    for (std::string word; stream >> word;) {
        words.push_back(word);
    }
    return words;
}

// Word-level edit distance.
size_t word_distance(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    std::vector<size_t> row(b.size() + 1);
    for (size_t j = 0; j <= b.size(); j++) {
        row[j] = j;
    }
    for (size_t i = 1; i <= a.size(); i++) {
        size_t diagonal = row[0];
        row[0] = i;
        for (size_t j = 1; j <= b.size(); j++) {
            const size_t above = row[j];
            row[j] = std::min({row[j] + 1, row[j - 1] + 1,
                               diagonal + (a[i - 1] == b[j - 1] ? 0 : 1)});
            diagonal = above;
        }
    }
    return row[b.size()];
}

void verify_refused_without_openvino() {
    for (const char* mode : {"full", "encoder"}) {
        Config config;
        config.backend = "openvino";
        config.openvino_mode = mode;
        const std::string failure = startup_failure(config);
        require(failure.contains("no OpenVINO support") &&
                    failure.contains("-DGRPC_ASR_OPENVINO=ON"),
                std::string("a build without OpenVINO refuses mode ") + mode + ": " + failure);
    }
}

void verify_startup_refusals() {
    Config config;
    config.backend = "openvino";

    config.openvino_mode = "decoder";
    require(startup_failure(config).contains("GRPC_ASR_OPENVINO_MODE"),
            "an unknown mode is refused by name");

    // No model is read for any of these: the pool refuses before loading.
    config.openvino_mode = "full";
    config.warmup = false;
    require(startup_failure(config).contains("GRPC_ASR_WARMUP=0"),
            "full mode refuses to skip the warmup that carries the placement check");
    config.warmup = true;

    // The negative control: a device OpenVINO cannot open stops startup in
    // both modes and never runs anywhere else. GPU.7 is a valid device id no
    // test host has; a host without a render node reports plain GPU the
    // same way.
    for (const char* mode : {"full", "encoder"}) {
        config.openvino_mode = mode;
        config.openvino_device = "GPU.7";
        const std::string failure = startup_failure(config);
        require(failure.contains("OpenVINO device 'GPU.7'") &&
                    failure.contains("is not available") &&
                    failure.contains("nothing runs on another device instead"),
                std::string("an unavailable device fails startup in mode ") + mode + ": " +
                    failure);
    }
}

// The sentence in the JFK sample. The tiny model gets a word or two of it
// wrong on some backends ("as" for "ask"), which is the tolerance below.
constexpr const char* kJfkSentence =
    "and so my fellow americans ask not what your country can do for you ask what you can do "
    "for your country";
constexpr size_t kWordTolerance = 3;

void verify_full_graph(const std::string& model, const std::string& wav, const std::string& device) {
    const std::vector<float> samples = pcm_from_wav(wav);
    const std::vector<std::string> sentence = words_of(kJfkSentence);

    // ggml's OpenVINO backend fixes its device the first time ggml starts in
    // the process, and the CPU pool below is what starts it here.
    ::setenv("GGML_OPENVINO_DEVICE", device.c_str(), 1);

    Config config;
    config.models_dir = std::filesystem::path(model).parent_path();
    config.models = {"tiny.en"};

    // What the ggml CPU backend hears, taken before the OpenVINO pool exists:
    // that pool counts every Whisper graph in the process that runs off its
    // device.
    std::vector<std::string> cpu_words;
    {
        config.backend = "cpu";
        config.concurrency = 1;
        ModelPool cpu_pool(config);
        cpu_words = words_of(transcribe(cpu_pool, samples, 1));
    }
    require(word_distance(cpu_words, sentence) <= kWordTolerance,
            "the ggml CPU backend hears the JFK sentence");

    config.backend = "openvino";
    config.openvino_mode = "full";
    config.openvino_device = device;
    config.openvino_cache_dir =
        (std::filesystem::temp_directory_path() / "grpc-asr-openvino-full-test-cache").string();
    // Two states: the second one reuses the graphs the first compiled.
    config.concurrency = 2;

    ModelPool pool(config);
    require(pool.backend() == "openvino", "backend reported");
    require(pool.placement().starts_with("full graph on OpenVINO " + device + " ("),
            "placement names the mode and the device: " + pool.placement());
    std::println("placement: {}", pool.placement());
    require(pool.warmed_up(), "the warmup that carries the placement check ran");
    require(pool.off_device_nodes() == 0, "no node left the device during startup");

    // Greedy and beam search build different decoder graphs; beam search
    // also rebuilds the self-attention cache for its extra decoders.
    for (const int beam_size : {1, 5}) {
        const std::string text = transcribe(pool, samples, beam_size);
        const std::vector<std::string> heard = words_of(text);
        const size_t from_sentence = word_distance(heard, sentence);
        const size_t from_cpu = word_distance(heard, cpu_words);
        std::println("openvino {} full graph, beam {}: {} word(s) from the sentence, {} from "
                     "the ggml CPU transcript:{}",
                     device, beam_size, from_sentence, from_cpu, text);
        require(from_sentence <= kWordTolerance,
                "the JFK sample transcribes on OpenVINO " + device + " (beam " +
                    std::to_string(beam_size) + "), got:" + text);
        require(from_cpu <= 2 * kWordTolerance,
                "the OpenVINO transcript stays close to the ggml CPU one (beam " +
                    std::to_string(beam_size) + ")");
    }
    require(pool.off_device_nodes() == 0,
            "no Whisper compute node ran outside the OpenVINO device while transcribing (" +
                std::to_string(pool.off_device_nodes()) + " did)");
    require(asr::render_prometheus_metrics().contains(
                "grpc_asr_whisper_off_device_nodes_total 0\n"),
            "the off-device counter is exported and reads 0");
}

}  // namespace

int main() {
    try {
        if (!asr::engine::openvino_built()) {
            verify_refused_without_openvino();
            std::println("openvino-full-test passed (build without OpenVINO)");
            return 0;
        }
        verify_startup_refusals();

        const char* chosen = env_or_null("GRPC_ASR_TEST_OPENVINO_DEVICE");
        const std::string device = chosen != nullptr ? chosen : "CPU";
        const std::vector<asr::engine::OpenvinoDevice> devices = asr::engine::openvino_devices();
        const bool has_device = std::ranges::any_of(
            devices, [&](const asr::engine::OpenvinoDevice& seen) { return seen.id == device; });
        if (!has_device) {
            return missing_model_input("OpenVINO has no " + device +
                                       " device here; the full-graph case runs on it");
        }
        const char* model = env_or_null("GRPC_ASR_TEST_MODEL");
        const char* sample = env_or_null("GRPC_ASR_TEST_SAMPLE");
        if (model == nullptr || slurp(model).empty()) {
            return missing_model_input("no model weights at $GRPC_ASR_TEST_MODEL");
        }
        const std::string wav = sample == nullptr ? "" : slurp(sample);
        if (wav.empty()) {
            return missing_model_input("no fixture wav at $GRPC_ASR_TEST_SAMPLE");
        }
        verify_full_graph(model, wav, device);
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("openvino-full-test passed");
    return 0;
}
