# Qualification-only tracing reads actual scheduler assignments after successful
# execution. Buffer residency or requested configuration is not execution proof.
set(_source "${whisper_SOURCE_DIR}/src/whisper.cpp")
file(READ "${_source}" _text)
string(FIND "${_text}" "// grpc-asr: scheduler placement receipt" _already)
if(_already EQUAL -1)
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
  file(WRITE "${_source}" "${_text}")
endif()
