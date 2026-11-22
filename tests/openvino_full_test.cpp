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
// The case runs twice: on the model file as it is (f16), and on a q5_0
// quantization of it that the test writes with whisper.cpp's own quantizer.
// q5_0 is what most published whisper.cpp quantizations are stored as, and
// ggml's OpenVINO backend runs it through cmake/PatchGgmlOpenvino.cmake
// (openvino-q5-0-test checks the weight values themselves).
//
// Each run also decodes once with a quality check that cannot pass, so
// whisper.cpp falls back to a higher temperature and five decoders and
// replaces the self-attention cache with a larger one between two decoder
// graphs. ggml's OpenVINO backend used to reuse the graph it had compiled
// for the smaller cache and fail the decode (the second edit of
// cmake/PatchGgmlOpenvino.cmake).
//
// Before those, the test boots twice in child processes: a pool over two
// models, sharing one kernel cache directory, as a restarted server does.
// The second boot starts from the compiled graphs the first one left and
// must hear the same thing from both models. (On the Arc B70 a boot from
// OpenVINO's weightless cache did not: the third edit of the patch.)
//
// The full-mode case needs GRPC_ASR_TEST_MODEL, GRPC_ASR_TEST_SAMPLE and the
// chosen OpenVINO device, and skips with 77 without them.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <print>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "common-ggml.h"
#include "config.h"
#include "engine/model_pool.h"
#include "engine/openvino_devices.h"
#include "fixture.h"
#include "ggml.h"
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

std::string transcribe(ModelPool& pool, const std::string& name,
                       const std::vector<float>& samples, int beam_size) {
    ModelPool::Lease lease = pool.acquire(name);
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

// A greedy decode whose result can never pass whisper.cpp's quality check
// (no average log-probability reaches 1, and no no-speech probability reaches
// 2 to excuse it), so it is decoded again at temperature 1 with five decoders. On a state that has only decoded with one
// decoder so far, whisper.cpp frees the self-attention cache at that point and
// allocates one seven times the size. The text is sampled and not compared.
void decode_with_fallback(ModelPool& pool, const std::string& name,
                          const std::vector<float>& samples) {
    ModelPool::Lease lease = pool.acquire(name);
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = 4;
    params.language = "en";
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.logprob_thold = 1.0f;
    params.no_speech_thold = 2.0f;
    params.temperature_inc = 1.0f;
    params.greedy.best_of = 5;
    const int rc = whisper_full_with_state(lease.context(), lease.state(), params, samples.data(),
                                           static_cast<int>(samples.size()));
    require(rc == 0, "a decode that falls back to five decoders succeeds for model '" + name +
                         "' (whisper_full returned " + std::to_string(rc) + ")");
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

// Writes a quantization of the whisper model at `source` to `target`, the way
// whisper.cpp's quantize tool does: the header, mel filters and vocabulary are
// copied, then whisper.cpp's ggml_common_quantize_0 rewrites the 2-D weights.
void quantize_model(const std::string& source, const std::filesystem::path& target,
                    ggml_ftype ftype) {
    std::ifstream in(source, std::ios::binary);
    std::ofstream out(target, std::ios::binary | std::ios::trunc);
    require(in.good() && out.good(), "the model opens and " + target.string() + " can be written");
    auto copy = [&](size_t bytes) {
        std::string buffer(bytes, '\0');
        in.read(buffer.data(), static_cast<std::streamsize>(bytes));
        out.write(buffer.data(), static_cast<std::streamsize>(bytes));
    };
    auto copy_i32 = [&] {
        int32_t value = 0;
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
        return value;
    };
    require(copy_i32() == GGML_FILE_MAGIC, "the model is a ggml file");
    // n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
    // n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, then the
    // file type, which is the one field that changes.
    for (int field = 0; field < 10; field++) {
        copy_i32();
    }
    int32_t source_ftype = 0;
    in.read(reinterpret_cast<char*>(&source_ftype), sizeof(source_ftype));
    require(source_ftype == GGML_FTYPE_MOSTLY_F16 || source_ftype == GGML_FTYPE_ALL_F32,
            "the model to quantize is an unquantized one");
    const int32_t target_ftype =
        GGML_QNT_VERSION * GGML_QNT_VERSION_FACTOR + static_cast<int32_t>(ftype);
    out.write(reinterpret_cast<const char*>(&target_ftype), sizeof(target_ftype));
    const int32_t n_mel = copy_i32();
    const int32_t n_fft = copy_i32();
    copy(static_cast<size_t>(n_mel) * static_cast<size_t>(n_fft) * sizeof(float));
    const int32_t n_vocab = copy_i32();
    for (int32_t token = 0; token < n_vocab; token++) {
        copy(static_cast<size_t>(copy_i32()));
    }
    require(in.good(), "the model header reads");
    // The tensors whisper.cpp's quantize tool leaves alone.
    const std::vector<std::string> unquantized = {
        "encoder.conv1.bias", "encoder.conv2.bias", "encoder.positional_embedding",
        "decoder.positional_embedding"};
    require(ggml_common_quantize_0(in, out, ftype, {".*"}, unquantized),
            "whisper.cpp quantizes the model");
    out.close();
    require(out.good(), "the quantized model is written");
}

// Full mode on `device` for the model `name` (ggml-<name>.bin in
// `models_dir`), stored as `ftype`.
void verify_full_graph(const std::string& models_dir, const std::string& name, ggml_ftype ftype,
                       const std::string& wav, const std::string& device) {
    const std::vector<float> samples = pcm_from_wav(wav);
    const std::vector<std::string> sentence = words_of(kJfkSentence);
    const std::string stored = ggml_type_name(ggml_ftype_to_ggml_type(ftype));

    // ggml's OpenVINO backend fixes its device the first time ggml starts in
    // the process, and the first CPU pool below is what starts it here.
    ::setenv("GGML_OPENVINO_DEVICE", device.c_str(), 1);

    Config config;
    config.models_dir = models_dir;
    config.models = {name};

    // What the ggml CPU backend hears from the same file, taken before the
    // OpenVINO pool exists: that pool counts every Whisper graph in the
    // process that runs off its device.
    std::vector<std::string> cpu_words;
    {
        config.backend = "cpu";
        config.concurrency = 1;
        ModelPool cpu_pool(config);
        {
            ModelPool::Lease lease = cpu_pool.acquire(name);
            require(whisper_model_ftype(lease.context()) == ftype,
                    "the model file stores its weights as " + stored);
        }
        cpu_words = words_of(transcribe(cpu_pool, name, samples, 1));
    }
    require(word_distance(cpu_words, sentence) <= kWordTolerance,
            "the ggml CPU backend hears the JFK sentence from the " + stored + " weights");

    config.backend = "openvino";
    config.openvino_mode = "full";
    config.openvino_device = device;
    // One kernel cache per weight type: the graphs are the same, the weights
    // are not.
    config.openvino_cache_dir = (std::filesystem::temp_directory_path() /
                                 ("grpc-asr-openvino-full-test-cache-" + stored))
                                    .string();
    // Two states: the second one reuses the graphs the first compiled.
    config.concurrency = 2;

    ModelPool pool(config);
    require(pool.backend() == "openvino", "backend reported");
    require(pool.placement().starts_with("full graph on OpenVINO " + device + " ("),
            "placement names the mode and the device: " + pool.placement());
    std::println("placement: {}", pool.placement());
    require(pool.warmed_up(), "the warmup that carries the placement check ran");
    require(pool.off_device_nodes() == 0, "no node left the device during startup");

    // On a fresh state: the self-attention cache grows between two decoder
    // graphs of one transcription.
    decode_with_fallback(pool, name, samples);
    require(pool.off_device_nodes() == 0, "the fallback decode stayed on the device");

    // Greedy and beam search build different decoder graphs; beam search
    // also rebuilds the self-attention cache for its extra decoders.
    for (const int beam_size : {1, 5}) {
        const std::string text = transcribe(pool, name, samples, beam_size);
        const std::vector<std::string> heard = words_of(text);
        const size_t from_sentence = word_distance(heard, sentence);
        const size_t from_cpu = word_distance(heard, cpu_words);
        std::println("openvino {} full graph, {} weights, beam {}: {} word(s) from the sentence, "
                     "{} from the ggml CPU transcript:{}",
                     device, stored, beam_size, from_sentence, from_cpu, text);
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

// The environment variables that turn this binary into one boot of
// verify_two_boots: where to write what it heard, and the kernel cache
// directory to use.
constexpr const char* kBootReportVariable = "GRPC_ASR_TEST_OPENVINO_BOOT_REPORT";
constexpr const char* kBootCacheVariable = "GRPC_ASR_TEST_OPENVINO_BOOT_CACHE";

// "tiny.en" for ".../ggml-tiny.en.bin".
std::string model_name_of(const std::string& path) {
    const std::string stem = std::filesystem::path(path).stem().string();
    require(stem.starts_with("ggml-"), "model files are named ggml-<name>.bin: " + path);
    return stem.substr(5);
}

// One boot, in a process of its own: a full-mode pool over both models (load,
// warmup and placement check, as the server starts), then for each model the
// fallback decode and a greedy transcription. Writes "<model>\t<text>" lines
// to `report`.
int boot_once(const std::string& report) {
    const char* model = env_or_null("GRPC_ASR_TEST_MODEL");
    const char* second_model = env_or_null("GRPC_ASR_TEST_MULTILINGUAL_MODEL");
    const char* sample = env_or_null("GRPC_ASR_TEST_SAMPLE");
    const char* device = env_or_null("GRPC_ASR_TEST_OPENVINO_DEVICE");
    const char* cache = env_or_null(kBootCacheVariable);
    require(model != nullptr && second_model != nullptr && sample != nullptr && cache != nullptr,
            "the boot process is given its models, sample and cache directory");
    const std::vector<float> samples = pcm_from_wav(slurp(sample));

    Config config;
    config.models_dir = std::filesystem::path(model).parent_path().string();
    config.models = {model_name_of(model), model_name_of(second_model)};
    config.backend = "openvino";
    config.openvino_mode = "full";
    config.openvino_device = device != nullptr ? device : "CPU";
    config.openvino_cache_dir = cache;
    config.concurrency = 1;

    ModelPool pool(config);
    require(pool.warmed_up() && pool.off_device_nodes() == 0, "the boot warmed up on the device");
    std::ofstream out(report, std::ios::trunc);
    for (const std::string& name : config.models) {
        decode_with_fallback(pool, name, samples);
        out << name << '\t' << transcribe(pool, name, samples, 1) << '\n';
    }
    require(pool.off_device_nodes() == 0, "no node left the device in this boot");
    out.close();
    require(out.good(), "the boot report is written");
    return 0;
}

// Runs this binary again as one boot and returns the lines it reported.
std::vector<std::string> run_boot(int number, const std::filesystem::path& work) {
    const std::string report = (work / ("boot-" + std::to_string(number) + ".txt")).string();
    std::vector<std::string> variables;
    for (char** variable = environ; *variable != nullptr; variable++) {
        variables.emplace_back(*variable);
    }
    variables.push_back(std::string(kBootReportVariable) + "=" + report);
    variables.push_back(std::string(kBootCacheVariable) + "=" + (work / "cache").string());
    std::vector<char*> envp;
    for (std::string& variable : variables) {
        envp.push_back(variable.data());
    }
    envp.push_back(nullptr);
    char self[] = "/proc/self/exe";
    char* argv[] = {self, nullptr};
    pid_t child = 0;
    require(posix_spawn(&child, self, nullptr, nullptr, argv, envp.data()) == 0,
            "boot " + std::to_string(number) + " starts");
    int status = 0;
    require(waitpid(child, &status, 0) == child, "boot " + std::to_string(number) + " is awaited");
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "boot " + std::to_string(number) + " over the shared kernel cache succeeds (wait " +
                "status " + std::to_string(status) + ")");
    std::vector<std::string> lines;
    std::istringstream stream(slurp(report));
    for (std::string line; std::getline(stream, line);) {
        lines.push_back(line);
    }
    return lines;
}

// Two boots over one kernel cache directory, with two models in the pool.
void verify_two_boots() {
    const std::filesystem::path work =
        std::filesystem::temp_directory_path() / "grpc-asr-openvino-full-test-boots";
    std::filesystem::remove_all(work);
    std::filesystem::create_directories(work / "cache");
    const std::vector<std::string> first = run_boot(1, work);
    require(!std::filesystem::is_empty(work / "cache"),
            "the first boot left compiled kernels in the cache directory");
    const std::vector<std::string> second = run_boot(2, work);
    require(first.size() == 2 && second.size() == 2, "each boot reports both models");
    const std::vector<std::string> sentence = words_of(kJfkSentence);
    for (size_t i = 0; i < first.size(); i++) {
        std::println("boot 1: {}", first[i]);
        std::println("boot 2: {}", second[i]);
        const std::vector<std::string> heard_first = words_of(first[i].substr(first[i].find('\t')));
        const std::vector<std::string> heard_second =
            words_of(second[i].substr(second[i].find('\t')));
        require(first[i].substr(0, first[i].find('\t')) ==
                    second[i].substr(0, second[i].find('\t')),
                "both boots report the models in the same order");
        require(word_distance(heard_first, sentence) <= kWordTolerance &&
                    word_distance(heard_second, sentence) <= kWordTolerance,
                "both boots hear the JFK sentence");
        require(word_distance(heard_first, heard_second) <= kWordTolerance,
                "the boot from the warm kernel cache hears what the first boot heard");
    }
    std::filesystem::remove_all(work);
}

}  // namespace

int main() {
    try {
        if (const char* report = env_or_null(kBootReportVariable)) {
            return boot_once(report);
        }
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
        const char* second_model = env_or_null("GRPC_ASR_TEST_MULTILINGUAL_MODEL");
        if (second_model == nullptr || !std::filesystem::exists(second_model)) {
            return missing_model_input(
                "no second model at $GRPC_ASR_TEST_MULTILINGUAL_MODEL for the two-boot case");
        }
        verify_two_boots();

        const std::filesystem::path model_path(model);
        verify_full_graph(model_path.parent_path().string(), "tiny.en", GGML_FTYPE_MOSTLY_F16,
                          wav, device);

        // The same model as q5_0, in a directory of its own.
        const std::filesystem::path quantized_dir =
            std::filesystem::temp_directory_path() / "grpc-asr-openvino-full-test-q5_0";
        std::filesystem::create_directories(quantized_dir);
        quantize_model(model, quantized_dir / "ggml-tiny.en-q5_0.bin", GGML_FTYPE_MOSTLY_Q5_0);
        verify_full_graph(quantized_dir.string(), "tiny.en-q5_0", GGML_FTYPE_MOSTLY_Q5_0, wav,
                          device);
        std::filesystem::remove_all(quantized_dir);
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("openvino-full-test passed");
    return 0;
}
