#include "engine/model_pool.h"

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <openssl/sha.h>
#include <mutex>
#include <print>
#include <ranges>
#include <stdexcept>
#include <string_view>

#include "cancellation.h"
#include "ggml-backend.h"
#include "metrics.h"
#include "whisper.h"

namespace asr::engine {

namespace {

std::string available_backends() {
    std::string names;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        if (!names.empty()) {
            names += ", ";
        }
        names += ggml_backend_reg_name(ggml_backend_reg_get(i));
    }
    return names;
}

// Resolves the weight file for a model name: <models_dir>/ggml-<name>.bin.
std::filesystem::path model_path(const std::string& models_dir, const std::string& name) {
    return std::filesystem::path(models_dir) / ("ggml-" + name + ".bin");
}

// Suffix of the OpenVINO encoder IR payload that sits next to converted
// weights (ggml-<name>-encoder-openvino.bin plus its .xml). Not a model.
constexpr std::string_view kOpenvinoEncoderSuffix = "-encoder-openvino.bin";

}  // namespace

std::vector<std::string> discover_models(const std::string& models_dir) {
    std::error_code listing_error;
    std::filesystem::directory_iterator dir(models_dir, listing_error);
    if (listing_error) {
        throw std::invalid_argument("GRPC_ASR_MODELS_DIR=" + models_dir +
                                    " cannot be listed: " + listing_error.message());
    }
    std::vector<std::string> names;
    for (const auto& file : dir) {
        const std::string filename = file.path().filename().string();
        if (!filename.starts_with("ggml-") || file.path().extension() != ".bin") {
            continue;
        }
        if (filename.ends_with(kOpenvinoEncoderSuffix)) {
            continue;
        }
        names.push_back(filename.substr(5, filename.size() - 5 - 4));
    }
    if (names.empty()) {
        throw std::invalid_argument("no ggml-*.bin model files in " + models_dir +
                                    " and GRPC_ASR_MODELS is unset");
    }
    return names;
}

struct ModelPool::Entry {
    std::string name;
    size_t registered_states = 0;
    std::string hash;
    whisper_context* ctx = nullptr;
    std::mutex mutex;
    // _any for the stop_token-aware waits in acquire.
    std::condition_variable_any available;
    std::deque<whisper_state*> free_states;
    std::vector<whisper_state*> all_states;

    ~Entry() {
        if (registered_states > 0) {
            metrics_remove_model_states(name, registered_states);
        }
        for (whisper_state* state : all_states) {
            whisper_free_state(state);
        }
        if (ctx != nullptr) {
            whisper_free(ctx);
        }
    }
};

ModelPool::ModelPool(const Config& config) : backend_(config.backend), warmed_up_(config.warmup) {
    whisper_context_params cparams = whisper_context_default_params();
    if (backend_ == "cuda") {
        if (ggml_backend_reg_by_name("CUDA") == nullptr) {
            throw std::invalid_argument(
                "GRPC_ASR_BACKEND=cuda: this build has no CUDA backend (available: " +
                available_backends() + "); build with -DGRPC_ASR_CUDA=ON on a CUDA host");
        }
        cparams.use_gpu = true;
        cparams.gpu_device = config.cuda_device;
    } else {
        // cpu and openvino both run the ggml graph on CPU; openvino
        // additionally offloads the encoder and is verified per model
        // below.
        cparams.use_gpu = false;
    }

    std::vector<std::string> names = config.models;
    if (names.empty()) {
        // Discover every ggml-*.bin so a mounted model directory works
        // with zero configuration.
        names = discover_models(config.models_dir);
    }

    constexpr size_t kWarmupSamples = 16000;
    const std::vector<float> warmup_silence(config.warmup ? kWarmupSamples : 0, 0.0f);

    for (const std::string& name : names) {
        std::filesystem::path path = model_path(config.models_dir, name);
        if (!std::filesystem::exists(path)) {
            throw std::invalid_argument("model '" + name + "' has no weight file at " +
                                        path.string());
        }
        auto entry = std::make_unique<Entry>();
        entry->name = name;
        std::ifstream weights(path, std::ios::binary | std::ios::ate);
        const auto size = weights.tellg();
        if (!weights || size <= 0 || size > INT64_C(8589934592))
            throw std::runtime_error("invalid model file size");
        std::vector<char> bytes(static_cast<size_t>(size));
        weights.seekg(0);
        if (!weights.read(bytes.data(), size)) throw std::runtime_error("cannot read model weights");
        unsigned char digest[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size(), digest);
        entry->hash.assign(reinterpret_cast<const char*>(digest), sizeof(digest));
        entry->ctx = whisper_init_from_buffer_with_params_no_state(bytes.data(), bytes.size(), cparams);
        std::vector<char>().swap(bytes);
        if (entry->ctx == nullptr) {
            throw std::runtime_error("loading model '" + name + "' from " + path.string() +
                                     " failed");
        }
        for (size_t i = 0; i < config.concurrency; i++) {
            whisper_state* state = whisper_init_state(entry->ctx);
            if (state == nullptr) {
                throw std::runtime_error("allocating state " + std::to_string(i) +
                                         " for model '" + name + "' failed");
            }
            // Push before OpenVINO/warmup so ~Entry frees the state on throw.
            entry->all_states.push_back(state);
            if (backend_ == "openvino") {
                // Buffer-loaded weights do not retain a filesystem path. Pass
                // the matching encoder XML explicitly; cache directory comes
                // from GRPC_ASR_OPENVINO_CACHE_DIR.
                // Returns non-zero both when the converted encoder is
                // missing, when the device is unavailable, and when this
                // binary was built without the OpenVINO encoder; either way
                // the operator asked for something this process cannot do.
                if (whisper_ctx_init_openvino_encoder_with_state(
                        entry->ctx, state,
                        (path.parent_path() / (path.stem().string() + "-encoder-openvino.xml"))
                            .c_str(),
                        config.openvino_device.c_str(), config.openvino_cache_dir.c_str()) != 0) {
                    throw std::runtime_error(
                        "GRPC_ASR_BACKEND=openvino: OpenVINO encoder init failed for model '" +
                        name + "' on device '" + config.openvino_device +
                        "' (GRPC_ASR_OPENVINO_DEVICE); use an image built with "
                        "-DGRPC_ASR_OPENVINO=ON, a converted encoder model next to the "
                        "weights, and a device this host has");
                }
            }
            if (config.warmup) {
                whisper_full_params wparams =
                    whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                wparams.n_threads = static_cast<int>(config.threads);
                wparams.no_context = true;
                wparams.single_segment = true;
                wparams.print_special = false;
                wparams.print_progress = false;
                wparams.print_realtime = false;
                wparams.print_timestamps = false;
                wparams.language = "en";
                const auto start = std::chrono::steady_clock::now();
                const int rc = whisper_full_with_state(
                    entry->ctx, state, wparams, warmup_silence.data(),
                    static_cast<int>(warmup_silence.size()));
                const auto elapsed_ms = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count());
                if (rc != 0) {
                    throw std::runtime_error("warmup failed for model '" + name + "' state " +
                                             std::to_string(i) + " (rc=" + std::to_string(rc) +
                                             ")");
                }
                warmup_ms_ += elapsed_ms;
                std::println("grpc-asr warmup: {} state {} {}ms", name, i, elapsed_ms);
            }
            entry->free_states.push_back(state);
        }
        metrics_add_model_states(name, config.concurrency);
        entry->registered_states = config.concurrency;
        entries_[name] = std::move(entry);
        std::println("grpc-asr model loaded: {} ({} state(s), backend {})", name,
                     config.concurrency, backend_);
    }
    if (config.warmup) {
        metrics_record_warmup_ms(warmup_ms_);
    }
    std::println("grpc-asr backend: {} (GRPC_ASR_BACKEND={})", backend_, backend_);
}

ModelPool::~ModelPool() = default;

std::string ModelPool::model_hash(const std::string& model) const {
    return entries_.at(model)->hash;
}

ModelPool::Lease::~Lease() {
    if (pool_ != nullptr) {
        pool_->release(model_, state_);
    }
}

bool ModelPool::has_model(const std::string& model) const {
    return entries_.contains(model);
}

ModelPool::Lease ModelPool::acquire(const std::string& model, std::stop_token stop,
                                    std::chrono::steady_clock::time_point deadline) {
    auto found = entries_.find(model);
    if (found == entries_.end()) {
        throw std::invalid_argument("model '" + model + "' is not loaded");
    }
    Entry& entry = *found->second;
    const auto wait_start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(entry.mutex);
    const auto state_free = [&] { return !entry.free_states.empty(); };

    struct QueueDepthGuard {
        std::string_view model_name;
        bool queued = false;
        explicit QueueDepthGuard(std::string_view name, bool needs_wait)
            : model_name(name), queued(needs_wait) {
            if (queued) {
                metrics_inc_queue_depth(model_name);
            }
        }
        ~QueueDepthGuard() {
            if (queued) {
                metrics_dec_queue_depth(model_name);
            }
        }
    };

    bool got_state = state_free();
    if (!got_state) {
        QueueDepthGuard depth_guard(model, true);
        // A cancelled or timed-out stream leaves the queue at once instead of
        // holding its RPC thread and its whole upload until a state frees.
        got_state = deadline == std::chrono::steady_clock::time_point::max()
                        ? entry.available.wait(lock, stop, state_free)
                        : entry.available.wait_until(lock, stop, deadline, state_free);
    } else if (stop.stop_requested()) {
        got_state = false;
    }

    const auto wait_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                             wait_start)
            .count());
    metrics_observe_queue_wait(model, wait_ns);

    if (!got_state) {
        if (stop.stop_requested()) {
            throw Cancelled("the stream went away while waiting for a '" + model + "' state");
        }
        throw QueueTimeout("every '" + model + "' state stayed busy");
    }
    whisper_state* state = entry.free_states.front();
    entry.free_states.pop_front();
    metrics_inc_model_busy(model);
    return Lease(this, model, entry.ctx, state);
}

std::vector<std::string> ModelPool::model_names() const {
    return std::views::keys(entries_) | std::ranges::to<std::vector>();
}

void ModelPool::release(const std::string& model, whisper_state* state) {
    Entry& entry = *entries_.at(model);
    {
        std::lock_guard<std::mutex> lock(entry.mutex);
        entry.free_states.push_back(state);
    }
    metrics_dec_model_busy(model);
    entry.available.notify_one();
}

}  // namespace asr::engine
