// Native model qualification. This does not replace the streaming RPC gate.
#include "whisper.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<float> wav(const char* path) {
  std::ifstream file(path, std::ios::binary);
  std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), {});
  auto u16 = [&](size_t n) { return unsigned(bytes.at(n)) | (unsigned(bytes.at(n+1)) << 8); };
  auto u32 = [&](size_t n) { return u16(n) | (u16(n+2) << 16); };
  if (bytes.size() < 44 || std::string(reinterpret_cast<char*>(bytes.data()), 4) != "RIFF")
    throw std::runtime_error("missing WAV fixture");
  bool format = false;
  for (size_t pos = 12; pos + 8 <= bytes.size();) {
    const auto size = u32(pos + 4);
    if (size > bytes.size() - pos - 8) throw std::runtime_error("truncated fixture");
    const std::string id(reinterpret_cast<char*>(bytes.data() + pos), 4);
    if (id == "fmt ") format = size >= 16 && u16(pos+8) == 1 && u16(pos+10) == 1 &&
        u32(pos+12) == 16000 && u16(pos+22) == 16;
    if (id == "data") {
      if (!format || size % 2) throw std::runtime_error("fixture must be mono PCM16/16000");
      std::vector<float> audio(size/2);
      for (size_t i = 0; i < audio.size(); ++i) audio[i] = static_cast<int16_t>(u16(pos+8+2*i)) / 32768.0f;
      return audio;
    }
    pos += 8 + size + (size & 1);
  }
  throw std::runtime_error("fixture has no audio");
}
int main(int argc, char** argv) {
  if (argc != 4) { std::cerr << "usage: sycl_whisper_probe MODEL WAV cpu|sycl\n"; return 64; }
  try {
    const std::string backend = argv[3];
    if (backend != "cpu" && backend != "sycl") throw std::runtime_error("explicit backend required");
    auto cp = whisper_context_default_params(); cp.use_gpu = backend == "sycl";
    if (cp.use_gpu) {
      int ordinal = 0; bool found = false;
      for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto* device = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        if (std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device))) == "SYCL") {
          cp.gpu_device = ordinal; found = true;
          std::cout << "device=" << ggml_backend_dev_description(device) << '\n'; break;
        }
        ++ordinal;
      }
      if (!found) throw std::runtime_error("SYCL GPU unavailable");
    }
    auto audio = wav(argv[2]);
    std::unique_ptr<whisper_context, decltype(&whisper_free)> ctx(
        whisper_init_from_file_with_params_no_state(argv[1], cp), whisper_free);
    if (!ctx) throw std::runtime_error("model load failed");
    std::unique_ptr<whisper_state, decltype(&whisper_free_state)> state(whisper_init_state(ctx.get()), whisper_free_state);
    if (!state) throw std::runtime_error("state initialization failed");
    auto params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = 2; params.language = "en"; params.greedy.best_of = 1;
    params.print_progress = params.print_realtime = params.print_timestamps = params.print_special = false;
    params.no_context = true;
    bool cancelled = false;
    params.abort_callback = [](void* data) { return *static_cast<bool*>(data); };
    params.abort_callback_user_data = &cancelled;
    for (int repeat = 0; repeat < 7; ++repeat) {
      const auto start = std::chrono::steady_clock::now();
      if (whisper_full_with_state(ctx.get(), state.get(), params, audio.data(), static_cast<int>(audio.size())))
        throw std::runtime_error("transcription failed");
      const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
      std::string text; int64_t last = 0;
      for (int i = 0; i < whisper_full_n_segments_from_state(state.get()); ++i) {
        const auto begin = whisper_full_get_segment_t0_from_state(state.get(), i);
        const auto end = whisper_full_get_segment_t1_from_state(state.get(), i);
        if (begin < last || end <= begin || end > 1200) throw std::runtime_error("invalid timestamp order/bounds");
        last = end; text += whisper_full_get_segment_text_from_state(state.get(), i);
      }
      if (text.find("country can do for you") == std::string::npos || text.find("you can do for your country") == std::string::npos)
        throw std::runtime_error("JFK transcript assertion failed: " + text);
      std::cout << "repeat=" << repeat << " backend=" << backend << " elapsed_us=" << elapsed << " text=" << text << '\n' << std::flush;
      if (repeat == 5) {
        cancelled = true;
        if (!whisper_full_with_state(ctx.get(), state.get(), params, audio.data(), static_cast<int>(audio.size())))
          throw std::runtime_error("cancellation was ignored");
        cancelled = false;
      }
    }
    std::cout << "PASS real native transcript/timestamps/repeats/cancel/recovery; RPC separate\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
