#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "ggml-backend.h"

struct ggml_cgraph;

namespace asr::engine {

// Where the nodes of one kind of Whisper graph ran, summed over every time
// that graph was computed while a PlacementRecorder was installed.
struct StagePlacement {
    // Graph computes seen for this stage.
    size_t graphs = 0;
    // Nodes that compute something. Views, reshapes, permutes and transposes
    // only relabel memory and are not counted.
    size_t compute_nodes = 0;
    // Compute nodes the scheduler gave to the required device.
    size_t on_device = 0;
    // Largest number of scheduler splits in one compute; 1 means the whole
    // graph went to one backend in one piece.
    int max_splits = 0;
    // Compute nodes that ran elsewhere, keyed "<OP> on <backend>".
    std::map<std::string, size_t> elsewhere;
};

// The graphs whisper.cpp computes for one transcription, in order.
inline constexpr std::string_view kWhisperStages[] = {"conv", "encoder", "cross", "decoder"};

// Adds one computed graph to `stage`: every compute node is looked up in the
// scheduler and counted on `required` or under the backend that took it.
// Must run after the graph was computed and before the scheduler is reset.
void record_graph_placement(StagePlacement& stage, ggml_backend_dev_t required,
                            ggml_backend_sched_t sched, ggml_cgraph* graph);

// Collects the scheduler's real node placement for every Whisper graph
// computed in this process while the object is alive. ggml's scheduler hands
// ops the selected device does not support to the CPU backend without a
// word; this is how that gets noticed. One recorder at a time: it is the
// observer of the patched whisper.cpp, installed before the first graph
// runs. Safe to feed from several transcription threads.
class PlacementRecorder {
  public:
    explicit PlacementRecorder(ggml_backend_dev_t required);
    ~PlacementRecorder();

    PlacementRecorder(const PlacementRecorder&) = delete;
    PlacementRecorder& operator=(const PlacementRecorder&) = delete;

    // Per-stage counts since construction or the last take().
    std::map<std::string, StagePlacement, std::less<>> stages() const;

    // Empty when every Whisper stage was computed at least once since the
    // last take() and each of its compute nodes ran on the required device.
    // Otherwise one line per problem, naming the stage, the op and the
    // backend that took it.
    std::vector<std::string> violations() const;

    // "conv 6/6, encoder 81/81, cross 16/16, decoder 204/204" style receipt.
    std::string summary() const;

    // Forgets the counts so the next model starts from zero.
    void take();

    // After this, a compute node that leaves the device is no longer a
    // startup question: it is counted (metrics and off_device_nodes) and
    // each distinct stage/op/backend is logged once.
    void arm();

    // Compute nodes that ran off the device since arm().
    uint64_t off_device_nodes() const;

  private:
    static void observe(void* user, const char* stage, ggml_backend_sched_t sched,
                        ggml_cgraph* graph);

    ggml_backend_dev_t required_;
    mutable std::mutex mutex_;
    std::map<std::string, StagePlacement, std::less<>> stages_;
    bool armed_ = false;
    uint64_t off_device_nodes_ = 0;
    std::set<std::string> reported_;
};

// The ggml device the OpenVINO backend runs on, and where whisper.cpp finds
// it.
struct OpenvinoGgmlDevice {
    ggml_backend_dev_t device = nullptr;
    // Index among the GPU-type ggml devices of every backend, which is how
    // whisper_context_params.gpu_device counts.
    int gpu_index = -1;
    // ggml's own text for it, with the OpenVINO device id and its full name.
    std::string description;
};

// Finds the device ggml's OpenVINO backend selected at its one-time
// initialization and checks it is `requested` (an OpenVINO device id such as
// "GPU"). Throws std::invalid_argument when this build has no OpenVINO ggml
// backend, or when the backend came up on a different device.
OpenvinoGgmlDevice find_openvino_ggml_device(const std::string& requested);

// True when ggml's OpenVINO backend can run whisper weights of this
// whisper_model_ftype value. Q5_0, the usual whisper.cpp quantization, is
// among them; Q2_K and Q3_K are not.
bool openvino_supports_ftype(int ftype);

// "f16", "q5_0", ... for messages.
std::string ftype_name(int ftype);

}  // namespace asr::engine
