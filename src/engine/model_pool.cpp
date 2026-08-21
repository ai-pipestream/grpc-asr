#include "engine/model_pool.h"

#include <condition_variable>
#include <deque>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>

#include "ggml-backend.h"
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

}  // namespace

struct ModelPool::Entry {
    whisper_context* ctx = nullptr;
    std::mutex mutex;
    std::condition_variable available;
    std::deque<whisper_state*> free_states;
    std::vector<whisper_state*> all_states;

    ~Entry() {
        for (whisper_state* state : all_states) {
            whisper_free_state(state);
        }
        if (ctx != nullptr) {
            whisper_free(ctx);
        }
    }
};

ModelPool::ModelPool(const Config& config) : backend_(config.backend) {
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
        std::error_code listing_error;
        std::filesystem::directory_iterator dir(config.models_dir, listing_error);
        if (listing_error) {
            throw std::invalid_argument("GRPC_ASR_MODELS_DIR=" + config.models_dir +
                                        " cannot be listed: " + listing_error.message());
        }
        for (const auto& file : dir) {
            std::string stem = file.path().filename().string();
            if (stem.rfind("ggml-", 0) == 0 && file.path().extension() == ".bin") {
                names.push_back(stem.substr(5, stem.size() - 5 - 4));
            }
        }
        if (names.empty()) {
            throw std::invalid_argument("no ggml-*.bin model files in " + config.models_dir +
                                        " and GRPC_ASR_MODELS is unset");
        }
    }

    for (const std::string& name : names) {
        std::filesystem::path path = model_path(config.models_dir, name);
        if (!std::filesystem::exists(path)) {
            throw std::invalid_argument("model '" + name + "' has no weight file at " +
                                        path.string());
        }
        auto entry = std::make_unique<Entry>();
        entry->ctx = whisper_init_from_file_with_params_no_state(path.c_str(), cparams);
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
            if (backend_ == "openvino") {
                // nullptr model_path: whisper then derives the encoder IR as
                // <stem>-encoder-openvino.xml next to the ggml weights.
                // Passing the ggml path here makes OpenVINO read the ggml
                // bin as its IR, which can never work. The compile cache
                // cannot live next to the weights either (the models mount
                // is read-only), so point it at tmpfs /tmp.
                // Returns non-zero both when the converted encoder is
                // missing, when the device is unavailable, and when this
                // binary was built without the OpenVINO encoder; either way
                // the operator asked for something this process cannot do.
                static const std::string openvino_cache_dir = "/tmp/grpc-asr-openvino-cache";
                if (whisper_ctx_init_openvino_encoder_with_state(entry->ctx, state, nullptr,
                                                                 "GPU", openvino_cache_dir.c_str()) != 0) {
                    throw std::runtime_error(
                        "GRPC_ASR_BACKEND=openvino: OpenVINO encoder init failed for model '" +
                        name +
                        "'; use an image built with -DGRPC_ASR_OPENVINO=ON and a converted "
                        "encoder model next to the weights");
                }
            }
            entry->free_states.push_back(state);
            entry->all_states.push_back(state);
        }
        entries_[name] = std::move(entry);
        std::cout << "grpc-asr model loaded: " << name << " (" << config.concurrency
                  << " state(s), backend " << backend_ << ")" << std::endl;
    }
    std::cout << "grpc-asr backend: " << backend_ << " (GRPC_ASR_BACKEND=" << backend_ << ")"
              << std::endl;
}

ModelPool::~ModelPool() = default;

ModelPool::Lease::~Lease() {
    if (pool_ != nullptr) {
        pool_->release(model_, state_);
    }
}

bool ModelPool::has_model(const std::string& model) const {
    return entries_.count(model) != 0;
}

ModelPool::Lease ModelPool::acquire(const std::string& model) {
    auto found = entries_.find(model);
    if (found == entries_.end()) {
        throw std::invalid_argument("model '" + model + "' is not loaded");
    }
    Entry& entry = *found->second;
    std::unique_lock<std::mutex> lock(entry.mutex);
    entry.available.wait(lock, [&] { return !entry.free_states.empty(); });
    whisper_state* state = entry.free_states.front();
    entry.free_states.pop_front();
    return Lease(this, model, entry.ctx, state);
}

std::vector<std::string> ModelPool::model_names() const {
    std::vector<std::string> names;
    names.reserve(entries_.size());
    for (const auto& [name, entry] : entries_) {
        (void)entry;
        names.push_back(name);
    }
    return names;
}

void ModelPool::release(const std::string& model, whisper_state* state) {
    Entry& entry = *entries_.at(model);
    {
        std::lock_guard<std::mutex> lock(entry.mutex);
        entry.free_states.push_back(state);
    }
    entry.available.notify_one();
}

}  // namespace asr::engine
