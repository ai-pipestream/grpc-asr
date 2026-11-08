# One generated translation unit for the pinned Whisper source. No source-tree
# writes. VAD repair, fail-closed GPU init, execution tracing and the Metal
# state accessor share this controlled chain.
set(_source "${whisper_SOURCE_DIR}/src/whisper.cpp")
file(SHA256 "${_source}" _input_hash)
if(NOT _input_hash STREQUAL "891168a838301bd2baa5e8b87422a854dafeeb7da914d40a2e4c6bf33ff448ed" AND
   NOT _input_hash STREQUAL "0836362aa9cc6d5b43c40a89a0eefc7aaf88a2c4141b984d41d67af9a521b4b5")
  message(FATAL_ERROR "Whisper placement chain input hash changed; review the pinned source")
endif()
file(READ "${_source}" _text)
set(_vad_old [=[WHISPER_LOG_ERROR("%s: failed to compute VAD graph\n", __func__);
            break;]=])
set(_vad_new [=[WHISPER_LOG_ERROR("%s: failed to compute VAD graph\n", __func__);
            ggml_backend_sched_reset(sched);
            return false; // pipestream: propagate VAD compute failure]=])
string(REPLACE "${_vad_old}" "${_vad_new}" _text "${_text}")
set(_gpu_before "    ggml_backend_t backend_gpu = whisper_backend_init_gpu(params);\n")
set(_gpu_after "${_gpu_before}\n    // grpc-asr: an explicitly requested GPU must initialize.\n    if (params.use_gpu && !backend_gpu) {\n        return {};\n    }\n")
string(REPLACE "${_gpu_before}" "${_gpu_after}" _text "${_text}")
  set(_signature "bool   sched_reset = true) {")
  string(FIND "${_text}" "${_signature}" _position)
  if(_position EQUAL -1)
    message(FATAL_ERROR "whisper placement trace signature needs review")
  endif()
  string(REPLACE "${_signature}" "bool   sched_reset = true, const char * stage = \"other\") {" _text "${_text}")
  set(_call "ggml_graph_compute_helper(sched, gf, n_threads)")
  string(LENGTH "${_call}" _length)
  foreach(_stage conv encoder cross decoder)
    string(FIND "${_text}" "${_call}" _position)
    if(_position EQUAL -1)
      message(FATAL_ERROR "whisper placement trace call sites need review")
    endif()
    string(SUBSTRING "${_text}" 0 ${_position} _before)
    math(EXPR _end "${_position} + ${_length}")
    string(SUBSTRING "${_text}" ${_end} -1 _after)
    set(_text "${_before}ggml_graph_compute_helper(sched, gf, n_threads, true, \"${_stage}\")${_after}")
  endforeach()
  set(_anchor "    if (!t || sched_reset) {")
  string(FIND "${_text}" "${_anchor}" _position)
  if(_position EQUAL -1)
    message(FATAL_ERROR "whisper placement trace execution site needs review")
  endif()
  set(_trace [=[    // grpc-asr: scheduler placement receipt
    if (t && getenv("GRPC_ASR_TRACE_PLACEMENT")) {
        int cpu = 0, gpu = 0, other = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW ||
                node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE ||
                node->op == GGML_OP_PERMUTE) continue;
            auto * backend = ggml_backend_sched_get_tensor_backend(sched, node);
            auto * device = backend ? ggml_backend_get_device(backend) : nullptr;
            auto type = device ? ggml_backend_dev_type(device) : GGML_BACKEND_DEVICE_TYPE_ACCEL;
            if (type == GGML_BACKEND_DEVICE_TYPE_CPU) ++cpu;
            else if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) ++gpu;
            else ++other;
        }
        WHISPER_LOG_INFO("grpc-asr placement stage=%s cpu_ops=%d gpu_ops=%d other_ops=%d\n", stage, cpu, gpu, other);
    }

]=])
  string(REPLACE "${_anchor}" "${_trace}${_anchor}" _text "${_text}")

string(APPEND _text [=[

extern "C" bool asr_whisper_state_has_device(whisper_state * state, ggml_backend_dev_t device) {
    for (auto backend : state->backends) {
        if (ggml_backend_get_device(backend) == device) return true;
    }
    return false;
}
]=])
string(SHA256 _output_hash "${_text}")
if(NOT _output_hash STREQUAL "5bb518beb0defd5098badee6c7c6e576f15e98a1ce7a8e233408f52ac51fccc6")
  message(FATAL_ERROR "Whisper placement chain output hash mismatch: ${_output_hash}")
endif()
set(_generated "${CMAKE_CURRENT_BINARY_DIR}/observed-whisper.cpp")
file(WRITE "${_generated}" "${_text}")
get_target_property(_sources whisper SOURCES)
list(FILTER _sources EXCLUDE REGEX "(^|/)whisper\\.cpp$")
set_property(TARGET whisper PROPERTY SOURCES "${_sources};${_generated}")
