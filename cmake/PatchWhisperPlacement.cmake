# whisper's public API does not expose the backends of an initialized state.
# This read-only accessor detects GPU initialization falling back to CPU.
# It is deliberately not a claim that every graph operation ran on the GPU.
set(_placement_source "${whisper_SOURCE_DIR}/src/whisper.cpp")
file(READ "${_placement_source}" _placement_text)
if(NOT _placement_text MATCHES "asr_whisper_state_has_device")
  string(FIND "${_placement_text}" "state->backends = whisper_backend_init(ctx->params);" _placement_anchor)
  if(_placement_anchor EQUAL -1)
    message(FATAL_ERROR "Whisper placement accessor needs review for changed upstream source")
  endif()
  file(APPEND "${_placement_source}" [=[

extern "C" bool asr_whisper_state_has_device(whisper_state * state, ggml_backend_dev_t device) {
    for (auto backend : state->backends) {
        if (ggml_backend_get_device(backend) == device) return true;
    }
    return false;
}
]=])
endif()
