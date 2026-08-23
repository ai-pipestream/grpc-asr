// Model discovery over a hand-authored models directory: ggml weight files
// become model names, OpenVINO encoder IR payloads next to them must not.

#include "engine/model_pool.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

#include "fixture.h"

namespace {

void touch(const std::filesystem::path& path) {
    std::ofstream file(path);
    file << "x";
}

}  // namespace

int main() {
    try {
        const std::filesystem::path dir =
            std::filesystem::temp_directory_path() / "grpc-asr-discovery-test";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);

        touch(dir / "ggml-tiny.en.bin");
        touch(dir / "ggml-base.bin");
        // Converted OpenVINO encoder for tiny.en: must not become a model.
        touch(dir / "ggml-tiny.en-encoder-openvino.bin");
        touch(dir / "ggml-tiny.en-encoder-openvino.xml");
        // Not weight files at all.
        touch(dir / "README.md");
        touch(dir / "ggml-tiny.en.txt");

        const std::vector<std::string> names = asr::engine::discover_models(dir.string());
        require(names.size() == 2, "discovers exactly the two weight files");
        require(std::ranges::contains(names, "tiny.en"), "tiny.en discovered");
        require(std::ranges::contains(names, "base"), "base discovered");
        require(!std::ranges::contains(names, "tiny.en-encoder-openvino"),
                "encoder IR payload is not a model");

        std::filesystem::remove_all(dir);

        const std::filesystem::path empty_dir =
            std::filesystem::temp_directory_path() / "grpc-asr-discovery-test-empty";
        std::filesystem::remove_all(empty_dir);
        std::filesystem::create_directories(empty_dir);
        bool threw = false;
        try {
            (void)asr::engine::discover_models(empty_dir.string());
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        std::filesystem::remove_all(empty_dir);
        require(threw, "empty models dir fails loud");
    } catch (const std::exception& e) {
        std::println(stderr, "{}", e.what());
        return 1;
    }
    std::println("model-discovery-test OK");
    return 0;
}
