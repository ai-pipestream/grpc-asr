// Q5_0 weights on ggml's OpenVINO backend (cmake/PatchGgmlOpenvino.cmake).
//
// A Q5_0 block is an fp16 scale d, four bytes holding the fifth bit of each
// of its 32 values, and sixteen bytes of low nibbles; a weight is
// d * (q - 16). The test writes blocks byte by byte, with every q at every
// position, and checks two things:
//
// - ggml's CPU dequantizer reads those bytes as d * (q - 16). This holds in
//   every build and pins the layout the patch unpacks.
// - In an OpenVINO build, the same tensor read back through the OpenVINO
//   backend gives the CPU dequantizer's values. GET_ROWS of every row copies
//   the dequantized weights: bit for bit, or within the rounding of the f16
//   multiply the backend dequantizes with. MUL_MAT with the identity is the
//   op Whisper's layers use; a device computes it in its inference precision
//   (f32, f16 on a GPU, bf16 on a CPU with AVX512-BF16), and the comparison
//   allows for the coarsest of those. The scheduler must have run the op on
//   the OpenVINO device: before the patch it gave Q5_0 ops to the CPU
//   backend, and reading CPU results back would prove nothing.
//
// Weights are loaded both ways ggml callers do it: marked as weights after
// the load (whisper.cpp) and before it (the backend then unpacks at load).
//
// The device is the OpenVINO CPU one unless GRPC_ASR_TEST_OPENVINO_DEVICE
// names another, as in openvino-full-test. No model is needed.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/openvino_devices.h"
#include "engine/placement.h"
#include "fixture.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

namespace {

constexpr int kBlockValues = 32;
constexpr size_t kBlockBytes = 2 + 4 + 16;
// Three blocks to a row, so a row is not a power of two of them.
constexpr int kColumns = 3 * kBlockValues;
constexpr int kRows = 40;

struct Q5_0Fixture {
    // The tensor as it is stored in a model file.
    std::vector<uint8_t> bytes;
    // d * (q - 16) for every weight, from the definition of the format.
    std::vector<float> expected;
};

// A fixed pseudo-random sequence (no <random>: its distributions differ
// between standard libraries).
struct Sequence {
    uint32_t state = 0x9E3779B9U;
    uint32_t next() {
        state = state * 1664525U + 1013904223U;
        return state >> 8;
    }
};

Q5_0Fixture make_fixture() {
    // Scales as a quantizer writes them (max / -16): both signs, small and
    // large, and zero for a block of all-zero weights.
    constexpr float scales[] = {0.0123f, -0.0457f, 0.25f,   -1.0f,     0.000731f, -0.00219f,
                                0.0f,    3.0625f,  -0.731f, 0.000097f, 0.5f,      -0.0333f};
    Q5_0Fixture fixture;
    Sequence sequence;
    const int blocks = kColumns / kBlockValues * kRows;
    fixture.bytes.resize(static_cast<size_t>(blocks) * kBlockBytes);
    fixture.expected.resize(static_cast<size_t>(blocks) * kBlockValues);
    for (int block = 0; block < blocks; block++) {
        const ggml_fp16_t scale = ggml_fp32_to_fp16(scales[block % std::size(scales)]);
        const float d = ggml_fp16_to_fp32(scale);
        uint32_t high_bits = 0;
        uint8_t nibbles[kBlockValues / 2] = {};
        for (int j = 0; j < kBlockValues; j++) {
            // The first 32 blocks put each q at each position once; the rest
            // are pseudo-random.
            const uint32_t q = block < kBlockValues
                                   ? static_cast<uint32_t>(block + j) % 32U
                                   : sequence.next() % 32U;
            // Value j < 16 is the low nibble of byte j, value j >= 16 the
            // high nibble of byte j - 16; bit j of the 32-bit word is its
            // fifth bit.
            high_bits |= (q >> 4) << j;
            const auto nibble = static_cast<uint8_t>(q & 0x0FU);
            if (j < kBlockValues / 2) {
                nibbles[j] = static_cast<uint8_t>(nibbles[j] | nibble);
            } else {
                nibbles[j - kBlockValues / 2] =
                    static_cast<uint8_t>(nibbles[j - kBlockValues / 2] | (nibble << 4));
            }
            fixture.expected[static_cast<size_t>(block) * kBlockValues + static_cast<size_t>(j)] =
                d * (static_cast<float>(q) - 16.0f);
        }
        uint8_t* out = fixture.bytes.data() + static_cast<size_t>(block) * kBlockBytes;
        std::memcpy(out, &scale, 2);
        std::memcpy(out + 2, &high_bits, 4);
        std::memcpy(out + 6, nibbles, sizeof(nibbles));
    }
    return fixture;
}

// What ggml's CPU backend makes of the fixture.
std::vector<float> cpu_dequantized(const Q5_0Fixture& fixture) {
    require(ggml_row_size(GGML_TYPE_Q5_0, kColumns) * kRows == fixture.bytes.size(),
            "the fixture is the size ggml gives a Q5_0 tensor");
    std::vector<float> values(fixture.expected.size());
    ggml_get_type_traits(GGML_TYPE_Q5_0)
        ->to_float(fixture.bytes.data(), values.data(), static_cast<int64_t>(values.size()));
    return values;
}

void verify_cpu_reference(const Q5_0Fixture& fixture, const std::vector<float>& cpu) {
    for (size_t i = 0; i < cpu.size(); i++) {
        require(cpu[i] == fixture.expected[i],
                "ggml's CPU dequantizer reads weight " + std::to_string(i) + " as d * (q - 16): " +
                    std::to_string(cpu[i]) + " vs " + std::to_string(fixture.expected[i]));
    }
    // The largest scale is 3.0625: q = 0 is -49 there and q = 31 is 45.9.
    const auto [low, high] = std::ranges::minmax_element(cpu);
    require(*low == -49.0f && *high > 45.9f, "the fixture reaches both ends of the 5-bit range");
}

// The fixture tensor read back through `op` on the OpenVINO device, in the
// memory order of the tensor itself (row after row).
std::vector<float> openvino_dequantized(ggml_backend_dev_t device, const Q5_0Fixture& fixture,
                                        ggml_op op, bool weights_marked_before_load) {
    ggml_backend_t openvino = ggml_backend_dev_init(device, nullptr);
    require(openvino != nullptr, "the OpenVINO backend starts");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    require(cpu != nullptr, "the CPU backend starts");
    ggml_backend_t backends[] = {openvino, cpu};

    // The weight lives in its own context and buffer, as a model's do.
    ggml_init_params weight_params = {
        .mem_size = 4 * ggml_tensor_overhead(),
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    ggml_context* weight_ctx = ggml_init(weight_params);
    ggml_tensor* weight = ggml_new_tensor_2d(weight_ctx, GGML_TYPE_Q5_0, kColumns, kRows);
    ggml_set_name(weight, "q5_0.weight");
    ggml_backend_buffer_t weight_buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, ggml_backend_dev_buffer_type(device));
    require(weight_buffer != nullptr, "the OpenVINO buffer for the weight allocates");
    if (weights_marked_before_load) {
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    require(ggml_nbytes(weight) == fixture.bytes.size(), "the weight is the fixture's size");
    ggml_backend_tensor_set(weight, fixture.bytes.data(), 0, fixture.bytes.size());
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_init_params graph_params = {
        .mem_size = 16 * ggml_tensor_overhead() + ggml_graph_overhead(),
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    ggml_context* ctx = ggml_init(graph_params);
    ggml_tensor* input = nullptr;
    ggml_tensor* result = nullptr;
    if (op == GGML_OP_GET_ROWS) {
        input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kRows);
        result = ggml_get_rows(ctx, weight, input);
    } else {
        input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kColumns, kColumns);
        result = ggml_mul_mat(ctx, weight, input);
    }
    ggml_set_name(input, "q5_0.input");
    ggml_set_name(result, "q5_0.result");
    ggml_set_input(input);
    ggml_set_output(result);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 64, false, true);
    require(ggml_backend_sched_alloc_graph(sched, graph), "the graph allocates");
    if (op == GGML_OP_GET_ROWS) {
        std::vector<int32_t> rows(kRows);
        for (int row = 0; row < kRows; row++) {
            rows[static_cast<size_t>(row)] = row;
        }
        ggml_backend_tensor_set(input, rows.data(), 0, ggml_nbytes(input));
    } else {
        std::vector<float> identity(static_cast<size_t>(kColumns) * kColumns, 0.0f);
        for (int i = 0; i < kColumns; i++) {
            identity[static_cast<size_t>(i) * kColumns + static_cast<size_t>(i)] = 1.0f;
        }
        ggml_backend_tensor_set(input, identity.data(), 0, ggml_nbytes(input));
    }
    require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
            std::string("the ") + ggml_op_name(op) + " graph computes");

    // Read before the scheduler is freed: where the one compute node ran.
    asr::engine::StagePlacement placement;
    asr::engine::record_graph_placement(placement, device, sched, graph);
    std::string elsewhere;
    for (const auto& [name, count] : placement.elsewhere) {
        elsewhere += " " + name;
    }
    require(placement.compute_nodes == 1 && placement.on_device == 1,
            std::string("the ") + ggml_op_name(op) +
                " over Q5_0 weights ran on the OpenVINO device, not:" + elsewhere);

    std::vector<float> computed(static_cast<size_t>(ggml_nelements(result)));
    require(computed.size() == fixture.expected.size(), "the result has one value per weight");
    ggml_backend_tensor_get(result, computed.data(), 0, ggml_nbytes(result));

    std::vector<float> values(computed.size());
    if (op == GGML_OP_GET_ROWS) {
        values = computed;
    } else {
        // weight x identity is [rows, columns] with the row index running
        // fastest; put it back in the tensor's own order.
        for (int row = 0; row < kRows; row++) {
            for (int column = 0; column < kColumns; column++) {
                values[static_cast<size_t>(row) * kColumns + static_cast<size_t>(column)] =
                    computed[static_cast<size_t>(column) * kRows + static_cast<size_t>(row)];
            }
        }
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(weight_ctx);
    ggml_backend_free(cpu);
    ggml_backend_free(openvino);
    return values;
}

// How far a value read back through `op` may be from the CPU dequantizer's,
// as a power of two of the value.
//
// GET_ROWS: the backend multiplies q - 16 by the scale in f16. The exact
// product needs at most 16 bits and f16 keeps 11, so the rounded result is
// within half a unit in the last place, 2^-11 of the value. 2^-10 leaves room
// for a device that rounds the other way.
//
// MUL_MAT: the device multiplies in its inference precision. The coarsest is
// bf16, 8 bits, on CPUs with AVX512-BF16: 2^-8 of the value for one rounding,
// 2^-7 allowed. A weight unpacked with a wrong bit is off by a sixteenth of
// the scale or more, far outside either bound.
int rounding_exponent(ggml_op op) { return op == GGML_OP_GET_ROWS ? -10 : -7; }

bool within_rounding(float value, float reference, int exponent) {
    // One f16 subnormal step covers the smallest values.
    const float bound = std::ldexp(std::fabs(reference), exponent) + std::ldexp(1.0f, -24);
    return std::fabs(value - reference) <= bound;
}

void verify_openvino(ggml_backend_dev_t device, const std::string& device_id,
                     const Q5_0Fixture& fixture, const std::vector<float>& cpu) {
    for (const ggml_op op : {GGML_OP_GET_ROWS, GGML_OP_MUL_MAT}) {
        for (const bool marked_before_load : {false, true}) {
            const std::vector<float> values =
                openvino_dequantized(device, fixture, op, marked_before_load);
            size_t exact = 0;
            float worst = 0.0f;
            for (size_t i = 0; i < values.size(); i++) {
                require(within_rounding(values[i], cpu[i], rounding_exponent(op)),
                        std::string(ggml_op_name(op)) + ": weight " + std::to_string(i) +
                            " is " + std::to_string(values[i]) + " on OpenVINO " + device_id +
                            " and " + std::to_string(cpu[i]) + " on CPU");
                exact += values[i] == cpu[i] ? 1 : 0;
                if (cpu[i] != 0.0f) {
                    worst = std::max(worst, std::fabs(values[i] - cpu[i]) / std::fabs(cpu[i]));
                }
            }
            std::println("openvino {} {} (weights marked {} load): {} of {} values equal the CPU "
                         "dequantizer's, largest relative difference {:.3g}",
                         device_id, ggml_op_name(op), marked_before_load ? "before" : "after",
                         exact, values.size(), worst);
        }
    }
}

}  // namespace

int main() {
    try {
        // ggml fills its f16 tables on the first context.
        ggml_free(ggml_init({.mem_size = 0, .mem_buffer = nullptr, .no_alloc = false}));

        const Q5_0Fixture fixture = make_fixture();
        const std::vector<float> cpu = cpu_dequantized(fixture);
        verify_cpu_reference(fixture, cpu);

        if (!asr::engine::openvino_built()) {
            std::println("openvino-q5-0-test passed (build without OpenVINO: CPU reference only)");
            return 0;
        }
        const char* chosen = env_or_null("GRPC_ASR_TEST_OPENVINO_DEVICE");
        const std::string device_id = chosen != nullptr ? chosen : "CPU";
        const std::vector<asr::engine::OpenvinoDevice> devices = asr::engine::openvino_devices();
        const bool has_device = std::ranges::any_of(
            devices, [&](const asr::engine::OpenvinoDevice& seen) { return seen.id == device_id; });
        if (!has_device) {
            return missing_model_input("OpenVINO has no " + device_id +
                                       " device here; the Q5_0 weights are read back on it");
        }
        // ggml's OpenVINO backend fixes its device the first time ggml lists
        // its backends, which find_openvino_ggml_device does.
        ::setenv("GGML_OPENVINO_DEVICE", device_id.c_str(), 1);
        const asr::engine::OpenvinoGgmlDevice openvino =
            asr::engine::find_openvino_ggml_device(device_id);
        std::println("ggml device: {}", openvino.description);
        verify_openvino(openvino.device, device_id, fixture, cpu);
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("openvino-q5-0-test passed");
    return 0;
}
