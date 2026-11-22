#pragma once

#include <string>
#include <vector>

namespace asr::engine {

// One device the OpenVINO runtime can open.
struct OpenvinoDevice {
    // OpenVINO's id: "GPU", "GPU.1", "CPU", "NPU".
    std::string id;
    // The hardware's own name, e.g. "Intel(R) Arc(TM) ... Graphics (dGPU)".
    std::string full_name;
};

// True when this binary links the OpenVINO runtime (-DGRPC_ASR_OPENVINO=ON).
bool openvino_built();

// The devices the OpenVINO runtime can open in this process right now.
// Empty in a build without OpenVINO, and empty when no plugin finds
// hardware: the GPU plugin reports nothing without a render node. Asks
// OpenVINO directly and never touches ggml, whose OpenVINO backend aborts
// the process when told to use a device that is not there. Throws
// std::runtime_error when the runtime itself cannot enumerate.
std::vector<OpenvinoDevice> openvino_devices();

}  // namespace asr::engine
