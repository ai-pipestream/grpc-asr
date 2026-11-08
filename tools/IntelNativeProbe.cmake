# Include with CMAKE_PROJECT_TOP_LEVEL_INCLUDES when configuring the pinned
# standalone Whisper source. Uses the same controlled patch chain as service.
get_filename_component(ASR_PROBE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
function(asr_add_native_probe)
  include("${ASR_PROBE_ROOT}/cmake/ObserveWhisperPlacement.cmake")
  add_executable(sycl-whisper-probe "${ASR_PROBE_ROOT}/tools/sycl_whisper_probe.cpp")
  target_compile_features(sycl-whisper-probe PRIVATE cxx_std_23)
  target_link_libraries(sycl-whisper-probe PRIVATE whisper)
endfunction()
cmake_language(DEFER CALL asr_add_native_probe)
