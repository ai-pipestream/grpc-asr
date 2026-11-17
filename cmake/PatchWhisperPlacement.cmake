# Three edits to the pinned whisper.cpp, all needed to run the whole Whisper
# graph on one explicitly selected ggml device and prove that it did (#30),
# without breaking the graphs that run on CPU in the same binary.
# Applied in place like the streaming/cmake patches: idempotent, and a
# changed upstream source stops the configure instead of building unpatched.
#
# 1. Tensor names. whisper.cpp leaves its weights and KV caches unnamed, so
#    ggml names each one leaf_<n> after its position in whichever graph
#    reaches it first. Two tensors in one graph can then share a name. ggml's
#    OpenVINO backend looks graph inputs up by name and then reads the wrong
#    tensor: the decoder graph fails to translate. Every weight gets its model
#    file name and every KV cache a fixed name.
#
# 2. Placement receipt. ggml's scheduler gives an op the selected device does
#    not support to the CPU backend without a word. The scheduler's choice is
#    only readable between graph compute and scheduler reset, inside
#    whisper.cpp, so the compute helper hands (stage, scheduler, graph) to
#    an observer there. src/engine/placement.cpp installs one for the life of
#    a full-OpenVINO model pool: it fails startup on any node that left the
#    device during warmup and counts any that leave it later. The observer is
#    set before the first graph runs and cleared after the last, so the two
#    plain statics need no lock.
#    asr_whisper_state_has_device reports whether a state really opened the
#    selected device, since whisper.cpp falls back to CPU when a GPU backend
#    fails to initialize.
#
# 3. Cross-attention key scale. Upstream removes the scale of the
#    cross-attention keys with #ifndef GGML_USE_OPENVINO, because the
#    OpenVINO backend applies it itself. That is a build-time switch: a
#    binary that merely contains the backend also loses the scale in every
#    graph it runs on CPU (GRPC_ASR_BACKEND=cpu, and the decoder of
#    GRPC_ASR_OPENVINO_MODE=encoder), and those transcripts degrade. The
#    scale is decided per state instead: dropped only when the state's first
#    backend is the OpenVINO one.
set(placement_source "${whisper_SOURCE_DIR}/src/whisper.cpp")
file(READ "${placement_source}" placement_text)
set(placement_marker "grpc-asr placement patch v1")

# Replaces the one occurrence of `old`; anything but exactly one is a pin
# change that needs a person.
function(asr_placement_replace old new what)
  string(FIND "${placement_text}" "${old}" first)
  string(FIND "${placement_text}" "${old}" last REVERSE)
  if(first EQUAL -1 OR NOT first EQUAL last)
    message(FATAL_ERROR "Pinned Whisper placement patch no longer matches: ${what}")
  endif()
  string(REPLACE "${old}" "${new}" placement_text "${placement_text}")
  set(placement_text "${placement_text}" PARENT_SCOPE)
endfunction()

string(FIND "${placement_text}" "${placement_marker}" placement_done)
if(placement_done EQUAL -1)
  string(FIND "${placement_text}" "grpc-asr placement patch" placement_stale)
  if(NOT placement_stale EQUAL -1)
    message(FATAL_ERROR "The fetched whisper.cpp carries an older placement patch; "
      "delete ${whisper_SOURCE_DIR} and configure again")
  endif()

  # --- 1. tensor names -----------------------------------------------------
  asr_placement_replace(
    [=[        model.tensors[format(ASR_TENSOR_NAMES.at(system).at(type), layer)] = tensor;
]=]
    [=[        model.tensors[format(ASR_TENSOR_NAMES.at(system).at(type), layer)] = tensor;
        ggml_set_name(tensor, format(ASR_TENSOR_NAMES.at(system).at(type), layer).c_str()); // grpc-asr: unique names
]=]
    "weight names")
  foreach(cache self cross pad)
    if(cache STREQUAL "self")
      set(label "kv self size  = %7.2f MB")
    elseif(cache STREQUAL "cross")
      set(label "kv cross size = %7.2f MB")
    else()
      set(label "kv pad  size  = %7.2f MB")
    endif()
    set(anchor "        WHISPER_LOG_INFO(\"%s: ${label}\\n\", __func__, memory_size / 1e6);\n")
    asr_placement_replace("${anchor}"
      "${anchor}        ggml_set_name(state->kv_${cache}.k, \"kv_${cache}_k\"); // grpc-asr: unique names\n        ggml_set_name(state->kv_${cache}.v, \"kv_${cache}_v\");\n"
      "kv_${cache} names")
  endforeach()
  # Beam search with more decoders than the state was sized for rebuilds the
  # self-attention cache; the new tensors need the names again.
  asr_placement_replace(
    [=[                    state->kv_self_n_dec = n_decoders_cur;
]=]
    [=[                    state->kv_self_n_dec = n_decoders_cur;
                    ggml_set_name(state->kv_self.k, "kv_self_k"); // grpc-asr: unique names
                    ggml_set_name(state->kv_self.v, "kv_self_v");
]=]
    "rebuilt kv_self names")

  # --- 2. placement receipt ------------------------------------------------
  asr_placement_replace(
    [=[//
// ggml helpers
//
]=]
    "//\n// ggml helpers\n//\n\n// ${placement_marker} (cmake/PatchWhisperPlacement.cmake)\ntypedef void (*asr_whisper_placement_fn)(void * user, const char * stage, ggml_backend_sched_t sched, struct ggml_cgraph * graph);\nstatic asr_whisper_placement_fn g_asr_placement_fn = nullptr;\nstatic void * g_asr_placement_user = nullptr;\n"
    "observer storage")
  asr_placement_replace(
    [=[                      bool   sched_reset = true) {]=]
    [=[                      bool   sched_reset = true,
                const char * asr_stage = nullptr) {]=]
    "compute helper signature")
  asr_placement_replace(
    [=[    if (!t || sched_reset) {]=]
    [=[    // grpc-asr: the scheduler's node placement is gone after the reset below
    if (t && asr_stage != nullptr && g_asr_placement_fn != nullptr) {
        g_asr_placement_fn(g_asr_placement_user, asr_stage, sched, graph);
    }

    if (!t || sched_reset) {]=]
    "observer call")
  # The four Whisper graphs call the helper with identical text, in this order.
  set(call "ggml_graph_compute_helper(sched, gf, n_threads)")
  string(LENGTH "${call}" call_length)
  foreach(stage conv encoder cross decoder)
    string(FIND "${placement_text}" "${call}" position)
    if(position EQUAL -1)
      message(FATAL_ERROR "Pinned Whisper placement patch no longer matches: ${stage} call site")
    endif()
    string(SUBSTRING "${placement_text}" 0 ${position} before)
    math(EXPR after_position "${position} + ${call_length}")
    string(SUBSTRING "${placement_text}" ${after_position} -1 after)
    set(placement_text
      "${before}ggml_graph_compute_helper(sched, gf, n_threads, true, \"${stage}\")${after}")
  endforeach()
  string(FIND "${placement_text}" "${call}" position)
  if(NOT position EQUAL -1)
    message(FATAL_ERROR "Pinned Whisper placement patch no longer matches: a fifth graph call site")
  endif()

  string(APPEND placement_text [=[

// grpc-asr: see cmake/PatchWhisperPlacement.cmake
extern "C" void asr_whisper_set_placement_observer(asr_whisper_placement_fn fn, void * user) {
    g_asr_placement_fn = fn;
    g_asr_placement_user = user;
}

extern "C" bool asr_whisper_state_has_device(struct whisper_state * state, ggml_backend_dev_t device) {
    for (auto backend : state->backends) {
        if (ggml_backend_get_device(backend) == device) {
            return true;
        }
    }
    return false;
}
]=])
  file(WRITE "${placement_source}" "${placement_text}")
endif()

# --- 3. cross-attention key scale ------------------------------------------
# Separate from the block above so a source tree that already has edits 1
# and 2 picks this one up.
set(scale_old [=[#ifndef GGML_USE_OPENVINO // scaling is handled internally in OpenVINO backend
        Kcross = ggml_scale(ctx0, Kcross, Kscale);
#endif]=])
set(scale_new [=[        // grpc-asr: per state, not per build (cmake/PatchWhisperPlacement.cmake).
        // ggml's OpenVINO backend applies this scale itself; every other backend needs it.
        if (wstate.backends.empty() || strcmp(ggml_backend_name(wstate.backends[0]), "OPENVINO") != 0) {
            Kcross = ggml_scale(ctx0, Kcross, Kscale);
        }]=])
string(FIND "${placement_text}" "${scale_old}" scale_position)
if(NOT scale_position EQUAL -1)
  string(REPLACE "${scale_old}" "${scale_new}" placement_text "${placement_text}")
  file(WRITE "${placement_source}" "${placement_text}")
else()
  string(FIND "${placement_text}" "${scale_new}" scale_position)
  if(scale_position EQUAL -1)
    message(FATAL_ERROR "Pinned Whisper placement patch no longer matches: cross-attention key scale")
  endif()
endif()
string(FIND "${placement_text}" "GGML_USE_OPENVINO" scale_position)
if(NOT scale_position EQUAL -1)
  message(FATAL_ERROR "The pinned whisper.cpp has another GGML_USE_OPENVINO build-time switch; "
    "review it: such a switch also changes graphs that run on CPU")
endif()
