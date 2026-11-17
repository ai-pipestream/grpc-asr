// Placement accounting: what counts as a compute node, how a node that ran
// off the required device is named, and that the recorder really sees all
// four Whisper graphs through the patched whisper.cpp. Runs on the CPU
// backend, so it holds in every build; the OpenVINO device itself is
// openvino-full-test's job.
//
// The graph cases need no model. The Whisper case needs GRPC_ASR_TEST_MODEL
// and skips with 77 when it is absent.

#include "engine/placement.h"

#include <filesystem>
#include <print>
#include <stdexcept>
#include <string>

#include "config.h"
#include "engine/model_pool.h"
#include "fixture.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "metrics.h"

using asr::engine::PlacementRecorder;
using asr::engine::StagePlacement;

namespace {

// A graph with three compute nodes (MUL_MAT, ADD, GELU) and two layout-only
// ones (RESHAPE, TRANSPOSE), scheduled on the CPU backend alone.
void verify_graph_accounting() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    require(cpu != nullptr, "CPU backend starts");
    ggml_backend_dev_t cpu_device = ggml_backend_get_device(cpu);
    ggml_backend_sched_t sched = ggml_backend_sched_new(&cpu, nullptr, 1, 64, false, true);

    ggml_init_params params = {
        .mem_size = 64 * ggml_tensor_overhead() + ggml_graph_overhead(),
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_tensor* bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_set_input(weight);
    ggml_set_input(input);
    ggml_set_input(bias);
    ggml_tensor* product = ggml_mul_mat(ctx, weight, input);
    ggml_tensor* sum = ggml_add(ctx, product, bias);
    ggml_tensor* activated = ggml_gelu(ctx, sum);
    ggml_tensor* reshaped = ggml_reshape_2d(ctx, activated, 2, 4);
    ggml_tensor* result = ggml_transpose(ctx, reshaped);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    require(ggml_backend_sched_alloc_graph(sched, graph), "graph allocates");
    for (ggml_tensor* tensor : {weight, input, bias}) {
        ggml_backend_tensor_memset(tensor, 0, 0, ggml_nbytes(tensor));
    }
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
            "graph computes");

    StagePlacement on_cpu;
    asr::engine::record_graph_placement(on_cpu, cpu_device, sched, graph);
    require(on_cpu.graphs == 1, "one graph recorded");
    require(on_cpu.compute_nodes == 3,
            "MUL_MAT, ADD and GELU are compute nodes, RESHAPE and TRANSPOSE are not (got " +
                std::to_string(on_cpu.compute_nodes) + ")");
    require(on_cpu.on_device == 3 && on_cpu.elsewhere.empty(),
            "every compute node is on the device that ran it");
    require(on_cpu.max_splits == 1, "one backend means one split");

    // The same graph judged against a device that ran none of it: every
    // compute node is named with the backend that really took it.
    StagePlacement strays;
    asr::engine::record_graph_placement(strays, nullptr, sched, graph);
    require(strays.on_device == 0 && strays.compute_nodes == 3, "nothing ran on a null device");
    require(strays.elsewhere.size() == 3, "three distinct ops are named");
    for (const char* op : {"MUL_MAT on CPU", "ADD on CPU", "GELU on CPU"}) {
        require(strays.elsewhere.contains(op) && strays.elsewhere.at(op) == 1,
                std::string("a stray node is named by op and backend: ") + op);
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(cpu);
}

void verify_ftypes() {
    using asr::engine::ftype_name;
    using asr::engine::openvino_supports_ftype;
    for (const ggml_ftype ok :
         {GGML_FTYPE_ALL_F32, GGML_FTYPE_MOSTLY_F16, GGML_FTYPE_MOSTLY_Q8_0,
          GGML_FTYPE_MOSTLY_Q5_1, GGML_FTYPE_MOSTLY_Q4_0, GGML_FTYPE_MOSTLY_Q4_1,
          GGML_FTYPE_MOSTLY_Q4_K, GGML_FTYPE_MOSTLY_Q5_K, GGML_FTYPE_MOSTLY_Q6_K}) {
        require(openvino_supports_ftype(ok), "OpenVINO runs " + ftype_name(ok) + " weights");
    }
    for (const ggml_ftype refused :
         {GGML_FTYPE_MOSTLY_Q5_0, GGML_FTYPE_MOSTLY_Q2_K, GGML_FTYPE_MOSTLY_Q3_K,
          GGML_FTYPE_UNKNOWN}) {
        require(!openvino_supports_ftype(refused),
                "OpenVINO refuses " + ftype_name(refused) + " weights");
    }
    require(ftype_name(GGML_FTYPE_MOSTLY_Q5_0) == "q5_0", "file types are named like ggml does");
    require(ftype_name(GGML_FTYPE_MOSTLY_F16) == "f16", "f16 is named");
}

// A real warmup through the patched whisper.cpp, on CPU: the recorder must
// see conv, encoder, cross and decoder, with every compute node on the CPU
// device, and must name them all when judged against a device that ran none.
void verify_whisper_stages(const std::string& model) {
    asr::Config config;
    config.backend = "cpu";
    config.concurrency = 1;
    config.models_dir = std::filesystem::path(model).parent_path();
    config.models = {"tiny.en"};

    ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    require(cpu_device != nullptr, "ggml has a CPU device");
    {
        PlacementRecorder recorder(cpu_device);
        asr::engine::ModelPool pool(config);
        const auto stages = recorder.stages();
        for (const std::string_view stage : asr::engine::kWhisperStages) {
            const auto found = stages.find(stage);
            require(found != stages.end() && found->second.graphs >= 1 &&
                        found->second.compute_nodes > 0,
                    "warmup computed the " + std::string(stage) + " graph");
            require(found->second.on_device == found->second.compute_nodes,
                    "every " + std::string(stage) + " node ran on the CPU device");
        }
        require(stages.size() == 4, "only Whisper's four graphs report a stage");
        require(recorder.violations().empty(), "a CPU run has no node off the CPU device");
        std::println("cpu placement: {}", recorder.summary());

        recorder.take();
        require(recorder.violations().size() == 4,
                "after take() every stage counts as never computed");
    }
    {
        // Judged against a device nothing ran on: startup-style violations
        // name the ops, and once armed the same strays are counted.
        PlacementRecorder recorder(nullptr);
        recorder.arm();
        asr::engine::ModelPool pool(config);
        const std::vector<std::string> strays = recorder.violations();
        require(!strays.empty(), "nodes off the device are violations");
        bool names_decoder_op = false;
        for (const std::string& line : strays) {
            names_decoder_op |= line.starts_with("decoder: ") && line.contains(" on CPU");
        }
        require(names_decoder_op, "a decoder op off the device is named with its backend");
        require(recorder.off_device_nodes() > 0, "an armed recorder counts stray nodes");
        require(asr::render_prometheus_metrics().contains(
                    "grpc_asr_whisper_off_device_nodes_total " +
                    std::to_string(recorder.off_device_nodes())),
                "stray nodes reach the Prometheus counter");
    }
}

}  // namespace

int main() {
    try {
        verify_graph_accounting();
        verify_ftypes();
        const char* model = env_or_null("GRPC_ASR_TEST_MODEL");
        if (model == nullptr || slurp(model).empty()) {
            return missing_model_input("no model weights at $GRPC_ASR_TEST_MODEL");
        }
        verify_whisper_stages(model);
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("placement-test passed");
    return 0;
}
