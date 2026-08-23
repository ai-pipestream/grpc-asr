cmake_minimum_required(VERSION 3.20)
project(grpc-asr VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# gRPC's CMake flips BUILD_SHARED_LIBS-adjacent defaults between releases;
# pin static explicitly so a green build cannot produce an unbootable image.
set(BUILD_SHARED_LIBS OFF)

option(GRPC_ASR_WERROR "Treat warnings as errors for first-party targets" OFF)
option(GRPC_ASR_CUDA "Build whisper.cpp with the CUDA (GGML) backend" OFF)
option(GRPC_ASR_OPENVINO "Build whisper.cpp with the OpenVINO encoder" OFF)
set(GRPC_ASR_SANITIZE "" CACHE STRING "Comma list for -fsanitize on first-party targets")

# Warnings apply to first-party targets only; fetched dependencies use -I
# (not -isystem) includes and would poison a global -Werror.
add_library(asr_flags INTERFACE)
target_compile_options(asr_flags INTERFACE -Wall -Wextra -Wnon-virtual-dtor)
if(GRPC_ASR_WERROR)
  target_compile_options(asr_flags INTERFACE -Werror -Wno-error=deprecated-declarations)
endif()
if(GRPC_ASR_SANITIZE)
  target_compile_options(asr_flags INTERFACE -fsanitize=${GRPC_ASR_SANITIZE} -fno-omit-frame-pointer -g)
  target_link_options(asr_flags INTERFACE -fsanitize=${GRPC_ASR_SANITIZE})
endif()

find_package(Threads REQUIRED)

include(FetchContent)

set(gRPC_INSTALL OFF CACHE BOOL "" FORCE)
set(gRPC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(grpc
  GIT_REPOSITORY https://github.com/grpc/grpc.git
  GIT_TAG v1.83.0
  GIT_SHALLOW TRUE
  GIT_SUBMODULES_RECURSE TRUE)
FetchContent_MakeAvailable(grpc)

# whisper.cpp is the ASR system of record. Examples/tests/server are the
# upstream demo surface, not this service.
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
if(GRPC_ASR_CUDA)
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
  # The CUDA-linked tests need the driver library (libcuda.so.1) at load
  # time. A container build has the toolkit but no driver, so the loader
  # kills them before main() can exit 77 to skip; gate at configure time
  # instead. Deliberately exclude the toolkit stub directory: the stub
  # links but never runs.
  find_library(GRPC_ASR_CUDA_DRIVER
    NAMES cuda libcuda.so.1
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib64 /lib64 /usr/lib/wsl/lib)
endif()
if(GRPC_ASR_OPENVINO)
  set(WHISPER_OPENVINO ON CACHE BOOL "" FORCE)
  # The OpenVINO-linked tests need the OpenVINO runtime (libopenvino.so)
  # at load time: they reference whisper symbols, so the linker pulls the
  # OpenVINO encoder objects and the loader resolves libopenvino before
  # main() can exit 77 to skip. A container build installs the toolkit, but
  # a build tree configured against headers alone would fail the in-build
  # ctest gate at load time; gate at configure time instead, same as the
  # CUDA driver probe above.
  find_library(GRPC_ASR_OPENVINO_RUNTIME
    NAMES openvino libopenvino.so
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib64
      /opt/intel/openvino/runtime/lib/intel64)
endif()
FetchContent_Declare(whisper
  GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
  GIT_TAG v1.9.3
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(whisper)

# ---------------------------------------------------------------------------
# Generated proto code. protoc and grpc_cpp_plugin are the fetched targets,
# so no host protoc is required (buf handles lint only).
# ---------------------------------------------------------------------------
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(ASR_PROTO "${PROTO_DIR}/ai/pipestream/asr/v1/asr_service.proto")
# Vendored verbatim from gRParse (the canonical copy); message-only, but one
# protoc run generates both files so the grpc plugin emits its (empty) stubs
# for it too rather than needing a second command.
set(DOC_PROTO "${PROTO_DIR}/ai/pipestream/document/v1/document.proto")

add_custom_command(
  OUTPUT
    "${GENERATED_DIR}/ai/pipestream/asr/v1/asr_service.pb.cc"
    "${GENERATED_DIR}/ai/pipestream/asr/v1/asr_service.pb.h"
    "${GENERATED_DIR}/ai/pipestream/asr/v1/asr_service.grpc.pb.cc"
    "${GENERATED_DIR}/ai/pipestream/asr/v1/asr_service.grpc.pb.h"
    "${GENERATED_DIR}/ai/pipestream/document/v1/document.pb.cc"
    "${GENERATED_DIR}/ai/pipestream/document/v1/document.pb.h"
    "${GENERATED_DIR}/ai/pipestream/document/v1/document.grpc.pb.cc"
    "${GENERATED_DIR}/ai/pipestream/document/v1/document.grpc.pb.h"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_DIR}"
  COMMAND $<TARGET_FILE:protoc>
  ARGS --proto_path=${PROTO_DIR}
       --proto_path=${protobuf_SOURCE_DIR}/src
       --cpp_out=${GENERATED_DIR}
       --grpc_out=${GENERATED_DIR}
       --plugin=protoc-gen-grpc=$<TARGET_FILE:grpc_cpp_plugin>
       ${ASR_PROTO} ${DOC_PROTO}
  DEPENDS ${ASR_PROTO} ${DOC_PROTO} protoc grpc_cpp_plugin
  VERBATIM)

add_library(asr_proto STATIC
  "${GENERATED_DIR}/ai/pipestream/asr/v1/asr_service.pb.cc"
  "${GENERATED_DIR}/ai/pipestream/asr/v1/asr_service.grpc.pb.cc"
  "${GENERATED_DIR}/ai/pipestream/document/v1/document.pb.cc"
  "${GENERATED_DIR}/ai/pipestream/document/v1/document.grpc.pb.cc")
target_include_directories(asr_proto PUBLIC "${GENERATED_DIR}")
target_link_libraries(asr_proto PUBLIC grpc++)

# ---------------------------------------------------------------------------
# First-party code
# ---------------------------------------------------------------------------
add_library(asr_core STATIC
  src/config.cpp
  src/media/sniff.cpp
  src/media/audio_decoder.cpp
  src/media/video_demux.cpp
  src/engine/model_pool.cpp
  src/engine/transcriber.cpp
  src/document/document_fold.cpp
  src/service/asr_service.cpp)
target_include_directories(asr_core PUBLIC src)
target_include_directories(asr_core SYSTEM PRIVATE third_party/miniaudio)
target_compile_definitions(asr_core PRIVATE GRPC_ASR_VERSION="${PROJECT_VERSION}")
target_link_libraries(asr_core
  PUBLIC asr_proto whisper Threads::Threads
  PRIVATE asr_flags)

add_executable(grpc-asr-server src/main.cpp)
target_link_libraries(grpc-asr-server PRIVATE asr_core asr_flags grpc++_reflection)

install(TARGETS grpc-asr-server RUNTIME DESTINATION bin)

# ---------------------------------------------------------------------------
# Tests: plain int main() executables under CTest, fleet style. Tests that
# need model weights or the ffmpeg binary skip with 77 when absent so the
# suite stays green on a bare checkout and asserts for real when provisioned.
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  set(GRPC_ASR_TESTS
    sniff-test
    audio-decoder-test
    document-fold-test
    model-discovery-test
    transcriber-test
    video-demux-test
    asr-service-test)
  foreach(test_name IN LISTS GRPC_ASR_TESTS)
    string(REPLACE "-" "_" test_source "${test_name}")
    add_executable(${test_name} "tests/${test_source}.cpp")
    target_link_libraries(${test_name} PRIVATE asr_core asr_flags)
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
  add_custom_target(grpc-asr-tests DEPENDS ${GRPC_ASR_TESTS})
  # 300 is generous on native hardware (asr-service-test finishes in ~3s);
  # QEMU-emulated arm64 image builds override it, whisper inference there
  # runs two orders of magnitude slower.
  set(GRPC_ASR_TEST_TIMEOUT 300 CACHE STRING
    "Per-test ctest timeout in seconds for the asr suite")
  set_tests_properties(${GRPC_ASR_TESTS} PROPERTIES
    TIMEOUT ${GRPC_ASR_TEST_TIMEOUT} LABELS asr)
  set_tests_properties(transcriber-test video-demux-test asr-service-test PROPERTIES
    SKIP_RETURN_CODE 77
    ENVIRONMENT "GRPC_ASR_TEST_MODEL=${CMAKE_CURRENT_SOURCE_DIR}/models/ggml-tiny.en.bin;GRPC_ASR_TEST_SAMPLE=${whisper_SOURCE_DIR}/samples/jfk.wav;GRPC_ASR_SERVER_BINARY=$<TARGET_FILE:grpc-asr-server>")
  # transcriber-test and asr-service-test pull in the ggml CUDA backend, so
  # they carry a load-time dependency on the driver library. video-demux-test
  # never references whisper symbols and links clean without it.
  # model-discovery-test links model_pool.o, which references the whisper and
  # ggml backend symbols, so it shares the same load-time exposure.
  if(GRPC_ASR_CUDA AND NOT GRPC_ASR_CUDA_DRIVER)
    set_tests_properties(transcriber-test asr-service-test model-discovery-test PROPERTIES DISABLED TRUE)
    message(STATUS "grpc-asr: CUDA driver library (libcuda.so.1) not found; "
      "disabling transcriber-test, asr-service-test, and model-discovery-test. GPU hosts with the driver installed still run them.")
  endif()
  # Same shape for OpenVINO: the whisper-linked tests carry a load-time
  # dependency on libopenvino.so through the encoder objects. Hosts with the
  # OpenVINO runtime installed run them exactly as before.
  if(GRPC_ASR_OPENVINO AND NOT GRPC_ASR_OPENVINO_RUNTIME)
    set_tests_properties(transcriber-test asr-service-test model-discovery-test PROPERTIES DISABLED TRUE)
    message(STATUS "grpc-asr: OpenVINO runtime library (libopenvino.so) not found; "
      "disabling transcriber-test, asr-service-test, and model-discovery-test. Hosts with the OpenVINO runtime installed still run them.")
  endif()
endif()
