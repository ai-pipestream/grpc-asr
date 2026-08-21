#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "config.h"

struct whisper_context;
struct whisper_state;

namespace asr::engine {

// Lists the model names a models directory serves with zero configuration:
// every ggml-*.bin weight file, minus the OpenVINO encoder IR payloads
// (ggml-<name>-encoder-openvino.bin) that live next to converted weights.
// Throws std::invalid_argument when the directory cannot be listed or holds
// no weight files.
std::vector<std::string> discover_models(const std::string& models_dir);

// Loads every configured whisper model once at startup (weights are shared
// per model via one whisper_context) and hands out per-transcription
// whisper_states from a bounded free-list. Any load or backend failure
// throws, so a misconfigured process never comes up half-working.
class ModelPool {
  public:
    // Verifies the requested backend actually exists in this build/host
    // (fail loud, never fall back), then loads the models. Prints one line
    // per decision, gRParse-style.
    explicit ModelPool(const Config& config);
    ~ModelPool();

    ModelPool(const ModelPool&) = delete;
    ModelPool& operator=(const ModelPool&) = delete;

    // A checked-out (context, state) pair. Returns the state to the pool
    // on destruction. The context is shared and must be treated as
    // read-only model weights.
    class Lease {
      public:
        Lease(ModelPool* pool, const std::string& model, whisper_context* ctx,
              whisper_state* state)
            : pool_(pool), model_(model), ctx_(ctx), state_(state) {}
        ~Lease();
        Lease(Lease&& other) noexcept
            : pool_(other.pool_), model_(other.model_), ctx_(other.ctx_), state_(other.state_) {
            other.pool_ = nullptr;
        }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease& operator=(Lease&&) = delete;

        whisper_context* context() const { return ctx_; }
        whisper_state* state() const { return state_; }

      private:
        ModelPool* pool_;
        std::string model_;
        whisper_context* ctx_;
        whisper_state* state_;
    };

    // True when the model name was loaded at startup.
    bool has_model(const std::string& model) const;

    // Blocks until a state for the model is free. The caller must have
    // checked has_model first; unknown names throw.
    Lease acquire(const std::string& model);

    // Loaded model names, for GetServiceInfo.
    std::vector<std::string> model_names() const;

    // The verified backend name: "cuda", "openvino", or "cpu".
    const std::string& backend() const { return backend_; }

  private:
    friend class Lease;
    void release(const std::string& model, whisper_state* state);

    struct Entry;
    std::string backend_;
    std::map<std::string, std::unique_ptr<Entry>> entries_;
};

}  // namespace asr::engine
