#include "ggml.h"
#include "ggml-cpu.h"
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

int main() {
  try {
    unsigned cases = 0;
    for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
      for (bool channels_first : {false, true}) {
        for (int threads : {1, 3}) {
          std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
              ggml_init({2 * 1024 * 1024, nullptr, false}), ggml_free);
          if (!ctx) throw std::runtime_error("cannot allocate test context");
          constexpr int channels = 3;
          const float a = type == GGML_TYPE_F32 ? 1.0f + 0x1p-23f : 1.0f + 0x1p-10f;
          const float b = 1.0f - 0x1p-23f;
          auto* input = channels_first ? ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, channels, 2, 1, 1) :
              ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 2, 1, channels, 1);
          auto* kernel = channels_first ? ggml_new_tensor_4d(ctx.get(), type, channels, 1, 2, 1) :
              ggml_new_tensor_4d(ctx.get(), type, 2, 1, 1, channels);
          for (int c = 0; c < channels; ++c) {
            for (int x = 0; x < 2; ++x) {
              const int offset = channels_first ? x * channels + c : c * 2 + x;
              static_cast<float*>(input->data)[offset] = x == 0 ? 1.0f : b;
              const float weight = x == 0 ? -1.0f : a;
              if (type == GGML_TYPE_F32) static_cast<float*>(kernel->data)[offset] = weight;
              else static_cast<ggml_fp16_t*>(kernel->data)[offset] = ggml_fp32_to_fp16(weight);
            }
          }
          if (channels_first) {
            input = ggml_permute(ctx.get(), input, 2, 0, 1, 3);
            kernel = ggml_permute(ctx.get(), kernel, 3, 2, 0, 1);
          }
          auto* output = ggml_conv_2d_dw_direct(ctx.get(), kernel, input, 1, 1, 0, 0, 1, 1);
          auto* graph = ggml_new_graph(ctx.get());
          ggml_build_forward_expand(graph, output);
          if (ggml_graph_compute_with_ctx(ctx.get(), graph, threads) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("depthwise graph failed");
          const float expected = std::fma(a, b, -1.0f);
          for (int c = 0; c < channels; ++c) {
            float actual;
            std::memcpy(&actual, static_cast<const char*>(output->data) + c * output->nb[2], sizeof actual);
            if (std::bit_cast<uint32_t>(actual) != std::bit_cast<uint32_t>(expected)) {
              std::cerr << "type=" << ggml_type_name(type) << " channels_first=" << channels_first
                        << " threads=" << threads << " actual=" << std::hexfloat << actual
                        << " expected=" << expected << '\n';
              throw std::runtime_error("depthwise accumulation changed rounding");
            }
          }
          ++cases;
        }
      }
    }
    std::cout << cases << " depthwise accumulation cases passed\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
