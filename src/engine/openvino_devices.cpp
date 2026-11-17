#include "engine/openvino_devices.h"

#ifdef GRPC_ASR_OPENVINO_BUILD
#include <exception>
#include <stdexcept>

#include <openvino/runtime/core.hpp>
#include <openvino/runtime/properties.hpp>
#endif

namespace asr::engine {

#ifdef GRPC_ASR_OPENVINO_BUILD

bool openvino_built() {
    return true;
}

std::vector<OpenvinoDevice> openvino_devices() {
    std::vector<OpenvinoDevice> devices;
    try {
        ov::Core core;
        for (const std::string& id : core.get_available_devices()) {
            OpenvinoDevice device{.id = id, .full_name = id};
            try {
                device.full_name = core.get_property(id, ov::device::full_name);
            } catch (const std::exception&) {
                // The id alone still identifies the device.
            }
            devices.push_back(std::move(device));
        }
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("OpenVINO could not list its devices: ") +
                                 error.what());
    }
    return devices;
}

#else

bool openvino_built() {
    return false;
}

std::vector<OpenvinoDevice> openvino_devices() {
    return {};
}

#endif

}  // namespace asr::engine
