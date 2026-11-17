#include "engine/placement.h"

#include <algorithm>
#include <print>
#include <stdexcept>

#include "ggml.h"
#include "metrics.h"

struct whisper_state;

// Defined in the pinned whisper.cpp by cmake/PatchWhisperPlacement.cmake.
extern "C" {
typedef void (*asr_whisper_placement_fn)(void* user, const char* stage, ggml_backend_sched_t sched,
                                         ggml_cgraph* graph);
void asr_whisper_set_placement_observer(asr_whisper_placement_fn fn, void* user);
}

namespace asr::engine {

namespace {

// Ops that hand on their input's memory under a new shape; no backend runs
// anything for them.
bool is_layout_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

std::string backend_names() {
    std::string names;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        if (!names.empty()) {
            names += ", ";
        }
        names += ggml_backend_reg_name(ggml_backend_reg_get(i));
    }
    return names;
}

}  // namespace

void record_graph_placement(StagePlacement& stage, ggml_backend_dev_t required,
                            ggml_backend_sched_t sched, ggml_cgraph* graph) {
    stage.graphs++;
    stage.max_splits = std::max(stage.max_splits, ggml_backend_sched_get_n_splits(sched));
    const int node_count = ggml_graph_n_nodes(graph);
    for (int i = 0; i < node_count; i++) {
        ggml_tensor* node = ggml_graph_node(graph, i);
        if (is_layout_op(node->op)) {
            continue;
        }
        stage.compute_nodes++;
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, node);
        ggml_backend_dev_t device = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
        if (required != nullptr && device == required) {
            stage.on_device++;
            continue;
        }
        stage.elsewhere[std::string(ggml_op_desc(node)) + " on " +
                        (backend != nullptr ? ggml_backend_name(backend) : "no backend")]++;
    }
}

PlacementRecorder::PlacementRecorder(ggml_backend_dev_t required) : required_(required) {
    asr_whisper_set_placement_observer(&PlacementRecorder::observe, this);
}

PlacementRecorder::~PlacementRecorder() {
    asr_whisper_set_placement_observer(nullptr, nullptr);
}

void PlacementRecorder::observe(void* user, const char* stage, ggml_backend_sched_t sched,
                                ggml_cgraph* graph) {
    auto* self = static_cast<PlacementRecorder*>(user);
    StagePlacement seen;
    record_graph_placement(seen, self->required_, sched, graph);

    std::lock_guard<std::mutex> lock(self->mutex_);
    StagePlacement& total = self->stages_[stage];
    total.graphs += seen.graphs;
    total.compute_nodes += seen.compute_nodes;
    total.on_device += seen.on_device;
    total.max_splits = std::max(total.max_splits, seen.max_splits);
    for (const auto& [op_on_backend, count] : seen.elsewhere) {
        total.elsewhere[op_on_backend] += count;
        if (!self->armed_) {
            continue;
        }
        self->off_device_nodes_ += count;
        metrics_add_off_device_nodes(count);
        if (self->reported_.insert(std::string(stage) + ": " + op_on_backend).second) {
            std::println(stderr,
                         "grpc-asr placement: {} graph ran {} outside the selected device; "
                         "grpc_asr_whisper_off_device_nodes_total counts every such node",
                         stage, op_on_backend);
        }
    }
}

std::map<std::string, StagePlacement, std::less<>> PlacementRecorder::stages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stages_;
}

std::vector<std::string> PlacementRecorder::violations() const {
    const auto seen = stages();
    std::vector<std::string> problems;
    for (const std::string_view name : kWhisperStages) {
        const auto found = seen.find(name);
        if (found == seen.end() || found->second.compute_nodes == 0) {
            problems.push_back(std::string(name) + ": the graph was never computed");
            continue;
        }
        for (const auto& [op_on_backend, count] : found->second.elsewhere) {
            problems.push_back(std::string(name) + ": " + std::to_string(count) + " x " +
                               op_on_backend);
        }
    }
    return problems;
}

std::string PlacementRecorder::summary() const {
    const auto seen = stages();
    std::string text;
    for (const std::string_view name : kWhisperStages) {
        const auto found = seen.find(name);
        const StagePlacement stage = found == seen.end() ? StagePlacement{} : found->second;
        if (!text.empty()) {
            text += ", ";
        }
        text += std::string(name) + " " + std::to_string(stage.on_device) + "/" +
                std::to_string(stage.compute_nodes);
    }
    return text;
}

void PlacementRecorder::take() {
    std::lock_guard<std::mutex> lock(mutex_);
    stages_.clear();
}

void PlacementRecorder::arm() {
    std::lock_guard<std::mutex> lock(mutex_);
    armed_ = true;
}

uint64_t PlacementRecorder::off_device_nodes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return off_device_nodes_;
}

OpenvinoGgmlDevice find_openvino_ggml_device(const std::string& requested) {
    ggml_backend_reg_t registry = ggml_backend_reg_by_name("OPENVINO");
    if (registry == nullptr) {
        throw std::invalid_argument(
            "GRPC_ASR_BACKEND=openvino: this build has no ggml OpenVINO backend (available: " +
            backend_names() + "); build with -DGRPC_ASR_OPENVINO=ON");
    }
    // ggml registers one device per OpenVINO device and marks the one its
    // backend runs on as GPU-type; the others are IGPU-type and listed only.
    // whisper.cpp counts both kinds when it resolves gpu_device.
    OpenvinoGgmlDevice found;
    int gpu_index = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        if (ggml_backend_dev_backend_reg(device) == registry &&
            type == GGML_BACKEND_DEVICE_TYPE_GPU) {
            found.device = device;
            found.gpu_index = gpu_index;
            found.description = ggml_backend_dev_description(device);
            break;
        }
        gpu_index++;
    }
    if (found.device == nullptr) {
        throw std::invalid_argument(
            "GRPC_ASR_BACKEND=openvino: ggml's OpenVINO backend selected no device");
    }
    // The backend reads GGML_OPENVINO_DEVICE once per process. If something
    // initialized ggml before the pool set it, the backend is on whatever it
    // picked then; refuse instead of running there.
    const std::string expected = "GGML_OPENVINO_DEVICE=" + requested + " (selected)";
    if (!found.description.starts_with(expected)) {
        throw std::invalid_argument(
            "GRPC_ASR_BACKEND=openvino: ggml's OpenVINO backend is on '" + found.description +
            "', not on the requested device " + requested +
            " (GRPC_ASR_OPENVINO_DEVICE); the device is fixed the first time ggml starts in a "
            "process");
    }
    return found;
}

bool openvino_supports_ftype(int ftype) {
    // The weight types ggml's OpenVINO backend accepts
    // (ggml-openvino.cpp, supported_types), as whisper file types. Q5_0 is
    // there through cmake/PatchGgmlOpenvino.cmake.
    switch (ftype) {
        case GGML_FTYPE_ALL_F32:
        case GGML_FTYPE_MOSTLY_F16:
        case GGML_FTYPE_MOSTLY_Q4_0:
        case GGML_FTYPE_MOSTLY_Q4_1:
        case GGML_FTYPE_MOSTLY_Q8_0:
        case GGML_FTYPE_MOSTLY_Q5_0:
        case GGML_FTYPE_MOSTLY_Q5_1:
        case GGML_FTYPE_MOSTLY_Q4_K:
        case GGML_FTYPE_MOSTLY_Q5_K:
        case GGML_FTYPE_MOSTLY_Q6_K:
            return true;
        default:
            return false;
    }
}

std::string ftype_name(int ftype) {
    // ggml_ftype_to_ggml_type asserts on a file type it has no weight type
    // for, so only the ones whisper.cpp writes are passed to it.
    switch (ftype) {
        case GGML_FTYPE_ALL_F32:
        case GGML_FTYPE_MOSTLY_F16:
        case GGML_FTYPE_MOSTLY_Q4_0:
        case GGML_FTYPE_MOSTLY_Q4_1:
        case GGML_FTYPE_MOSTLY_Q8_0:
        case GGML_FTYPE_MOSTLY_Q5_0:
        case GGML_FTYPE_MOSTLY_Q5_1:
        case GGML_FTYPE_MOSTLY_Q2_K:
        case GGML_FTYPE_MOSTLY_Q3_K:
        case GGML_FTYPE_MOSTLY_Q4_K:
        case GGML_FTYPE_MOSTLY_Q5_K:
        case GGML_FTYPE_MOSTLY_Q6_K:
            return ggml_type_name(ggml_ftype_to_ggml_type(static_cast<ggml_ftype>(ftype)));
        default:
            return "file type " + std::to_string(ftype);
    }
}

}  // namespace asr::engine
