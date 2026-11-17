# Three edits to ggml's OpenVINO backend, in the pinned whisper.cpp's ggml.
# Each is also kept as a diff for the backend's maintainers, with a write-up,
# under docs/upstream/. Keep the two forms in step.
# Applied in place like the other patches: idempotent, and a changed upstream
# source stops the configure instead of building unpatched.
#
# 1. Q5_0 weights (docs/upstream/ggml-openvino-q5_0.patch). The backend accepts
#    Q4_0, Q4_1, Q5_1, Q8_0 and the K types as weights, and not Q5_0: a tensor
#    of that type is handed to the CPU backend, and whisper.cpp aborts when a
#    state is built over it. Q5_0 is what most published whisper.cpp
#    quantizations are stored as.
#    A Q5_0 block is an fp16 scale d, 32 high bits and 32 low nibbles, and a
#    weight is d * (q - 16) with q in [0, 31]. It is symmetric, like Q8_0, and
#    takes the layout the backend already has for Q8_0: signed i8 weights
#    (q - 16), one f16 scale per group of 32, no zero point. Only the unpacking
#    of the 5-bit values is new.
#
# 2. A compiled graph outliving the KV buffer it was compiled for
#    (docs/upstream/ggml-openvino-flat-kv-reuse.patch). whisper.cpp keeps its
#    KV caches as flat buffers, which the backend compiles as inputs of a
#    fixed size, with the offsets of every layer's view fixed in the graph.
#    When a decode needs more decoders than the state has room for (beam
#    search, or the temperature fallback of a greedy decode, which samples
#    with five), whisper.cpp frees kv_self and allocates a larger one. Each
#    backend instance remembers the last graph it ran under a key of node
#    count and end names, which the decoder graph before and after the
#    reallocation share; when the two run back to back, as they do in a
#    temperature fallback, the backend reused the request compiled for the
#    small buffer, OpenVINO refused the larger tensor ("Can't set the input
#    tensor with index: 4, because the model input (shape=[1,1,1,786432])
#    and the tensor (shape=(1.1.1.5505024)) are incompatible") and the decode
#    failed. Seen on the Arc B70 as a warmup that failed on one boot and not
#    on another: silence decodes near the fallback threshold. The sizes of
#    the flat KV buffers are now part of what must match before a compiled
#    graph is reused; a change sends the graph down the cold path, whose key
#    already includes every tensor shape and the identity of the weights.
#
# 3. Cache blobs that carry their weights
#    (docs/upstream/ggml-openvino-cache-mode.patch). With a cache directory
#    the backend asks OpenVINO for CacheMode::OPTIMIZE_SIZE, the weightless
#    cache: a blob holds the compiled kernels, and the weights are taken from
#    the model again when the blob is loaded. On the Arc B70 a server started
#    from such a cache did not behave like the one that wrote it: the first
#    state of each model decoded the warmup silence differently (far enough
#    to take the temperature fallback, which is how edit 2 was found), and
#    compile_model crashed in the GPU plugin while it copied weights from
#    host memory (SIGSEGV in memcpy under libigdrcl) for some sets of models.
#    A server without a cache directory, and one with OPTIMIZE_SPEED, boot the
#    same way every time. The backend now asks for OPTIMIZE_SPEED: a blob
#    holds its weights, costs about as much disk as the graph's weights, and
#    loads several times faster than the weightless one did.
set(ggml_ov_directory "${whisper_SOURCE_DIR}/ggml/src/ggml-openvino")
set(q5_0_marker "GGML_TYPE_Q5_0")

# Replaces the one occurrence of `old` in ${ggml_ov_text}; anything but exactly
# one is a pin change that needs a person.
function(asr_ggml_ov_replace old new what)
  string(FIND "${ggml_ov_text}" "${old}" first)
  string(FIND "${ggml_ov_text}" "${old}" last REVERSE)
  if(first EQUAL -1 OR NOT first EQUAL last)
    message(FATAL_ERROR "Pinned ggml OpenVINO patch no longer matches: ${what}")
  endif()
  string(REPLACE "${old}" "${new}" ggml_ov_text "${ggml_ov_text}")
  set(ggml_ov_text "${ggml_ov_text}" PARENT_SCOPE)
endfunction()

# --- 1. Q5_0: ggml-quants.cpp unpacks the blocks and routes the type ----------
file(READ "${ggml_ov_directory}/ggml-quants.cpp" ggml_ov_text)
string(FIND "${ggml_ov_text}" "${q5_0_marker}" ggml_ov_done)
if(ggml_ov_done EQUAL -1)
  asr_ggml_ov_replace(
    [=[// Extracts (weight, scales, zp) from Q8_0 tensors.
]=]
    [=[// Extracts (weight, scales) from Q5_0 tensors.
// Data layout is: |16 bit scale|32 bit qh (5th bits)|32 x 4bit low nibbles|.
// Reconstructed quant q in [0,31]: q = (low nibble) | (qh_bit << 4). Dequant: (q - 16) * d.
// Q5_0 is symmetric, so the weights are stored as signed i8 (q - 16, in [-16, 15]) with no
// zero point: the same i8 weights * f16 scale layout as symmetric Q8_0, group size 32.
void extract_q5_0_data(const ggml_tensor * tensor, ov::Tensor & weights_arr, ov::Tensor & scales_arr) {
    const uint64_t bytes_per_block = 22;  // 2 scale + 4 qh + 16 (32x0.5) weights
    const int qk = 32;

    GGML_ASSERT(tensor->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(weights_arr.get_element_type() == ov::element::i8);

    const auto * data = static_cast<const uint8_t *>(tensor->data);
    auto * weights = static_cast<int8_t *>(weights_arr.data());  // i8 weights, one byte per weight
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
        const uint8_t * block = data + i * bytes_per_block;
        uint16_t d;
        uint32_t qh;
        memcpy(&d, block, sizeof(d));
        memcpy(&qh, block + 2, sizeof(qh));
        scales[i] = ov::float16::from_bits(d);

        const uint8_t * qs = block + 6;
        int8_t * dst = weights + i * qk;
        for (int j = 0; j < qk / 2; ++j) {
            const uint8_t lo = qs[j] & 0x0F;
            const uint8_t hi = qs[j] >> 4;
            const uint8_t bit_lo = (qh >> j) & 1;
            const uint8_t bit_hi = (qh >> (j + qk / 2)) & 1;
            dst[j] = static_cast<int8_t>((lo | (bit_lo << 4)) - 16);           // first 16 weights
            dst[j + qk / 2] = static_cast<int8_t>((hi | (bit_hi << 4)) - 16);  // last 16 weights
        }
    });
}

// Extracts (weight, scales, zp) from Q8_0 tensors.
]=]
    "extractor")
  asr_ggml_ov_replace(
    [=[    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q5_K:
        is_u4 = false;
]=]
    [=[    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q5_K:
        is_u4 = false;
]=]
    "block size switch")
  asr_ggml_ov_replace(
    [=[    case GGML_TYPE_Q5_1:
        extract_q5_1_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
]=]
    [=[    case GGML_TYPE_Q5_0:
        extract_q5_0_data(&temp_tensor, weights, scales);
        break;
    case GGML_TYPE_Q5_1:
        extract_q5_1_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
]=]
    "extraction switch")
  file(WRITE "${ggml_ov_directory}/ggml-quants.cpp" "${ggml_ov_text}")
endif()

# --- 1. Q5_0: ggml-openvino-extra.cpp, buffer layout --------------------------
file(READ "${ggml_ov_directory}/ggml-openvino-extra.cpp" ggml_ov_text)
string(FIND "${ggml_ov_text}" "${q5_0_marker}" ggml_ov_done)
if(ggml_ov_done EQUAL -1)
  asr_ggml_ov_replace(
    [=[    case GGML_TYPE_Q8_0:
        layout.is_symmetric = true;
        break;
]=]
    [=[    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q5_0:
        // i8 weights, symmetric (scale only). Q5_0's 5-bit values are stored as q - 16.
        layout.is_symmetric = true;
        break;
]=]
    "extracted layout")
  # 3D expert weights in a grouped 8-bit layout are requantized to 4 bits on
  # GPU (a plugin bug the backend works around); Q5_0 has that layout now.
  asr_ggml_ov_replace(
    [=[    if (tensor->ne[2] > 1 && (tensor->type == GGML_TYPE_Q5_1 || tensor->type == GGML_TYPE_Q8_0)) {]=]
    [=[    if (tensor->ne[2] > 1 &&
        (tensor->type == GGML_TYPE_Q5_0 || tensor->type == GGML_TYPE_Q5_1 || tensor->type == GGML_TYPE_Q8_0)) {]=]
    "expert weight requantization")
  asr_ggml_ov_replace(
    [=[stored as Q5_1/Q8_0 are the expert-side]=]
    [=[stored as Q5_0/Q5_1/Q8_0 are the expert-side]=]
    "expert weight comment")
  file(WRITE "${ggml_ov_directory}/ggml-openvino-extra.cpp" "${ggml_ov_text}")
endif()

# --- 1. Q5_0: ggml-openvino.cpp, the type is one the device runs --------------
file(READ "${ggml_ov_directory}/ggml-openvino.cpp" ggml_ov_text)
string(FIND "${ggml_ov_text}" "${q5_0_marker}" ggml_ov_done)
if(ggml_ov_done EQUAL -1)
  asr_ggml_ov_replace(
    [=[        GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_1, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K,
        GGML_TYPE_MXFP4};]=]
    [=[        GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0,
        GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};]=]
    "supported types")
  file(WRITE "${ggml_ov_directory}/ggml-openvino.cpp" "${ggml_ov_text}")
endif()

# --- 1. Q5_0: ggml-decoder.cpp, the type is a weight --------------------------
file(READ "${ggml_ov_directory}/ggml-decoder.cpp" ggml_ov_text)
string(FIND "${ggml_ov_text}" "${q5_0_marker}" ggml_ov_done)
if(ggml_ov_done EQUAL -1)
  asr_ggml_ov_replace(
    [=[                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};]=]
    [=[                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
                                                     GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};]=]
    "weight types")
  file(WRITE "${ggml_ov_directory}/ggml-decoder.cpp" "${ggml_ov_text}")
endif()

# --- 2. flat KV buffer sizes decide reuse -------------------------------------
set(flat_kv_marker "flat_kv_sizes")
file(READ "${ggml_ov_directory}/ggml-decoder.h" ggml_ov_text)
string(FIND "${ggml_ov_text}" "${flat_kv_marker}" ggml_ov_done)
if(ggml_ov_done EQUAL -1)
  asr_ggml_ov_replace(
    [=[    size_t kv_buffer_ctx_id = 0;
]=]
    [=[    size_t kv_buffer_ctx_id = 0;
    // Element counts of the flat KV buffers ([N, 1, 1, 1], whisper.cpp), in the order the
    // attention layers use them. A flat buffer is compiled with a static shape and the layers'
    // views into it have fixed offsets (see get_graph_input_shape), and whisper.cpp replaces
    // kv_self with a larger buffer when a decode needs more decoders. A compiled model is only
    // reusable while these are unchanged.
    std::vector<int64_t> flat_kv_sizes;
]=]
    "model params field")
  asr_ggml_ov_replace(
    [=[        return same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool can_reuse_statically(const ModelParams & other) const {
        return same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }]=]
    [=[        return same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback && flat_kv_sizes == other.flat_kv_sizes;
    }

    bool can_reuse_statically(const ModelParams & other) const {
        return same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback && flat_kv_sizes == other.flat_kv_sizes;
    }]=]
    "reuse rules")
  file(WRITE "${ggml_ov_directory}/ggml-decoder.h" "${ggml_ov_text}")
endif()

file(READ "${ggml_ov_directory}/ggml-decoder.cpp" ggml_ov_text)
string(FIND "${ggml_ov_text}" "${flat_kv_marker}" ggml_ov_done)
if(ggml_ov_done EQUAL -1)
  asr_ggml_ov_replace(
    [=[                auto * cache_k_view = node->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
]=]
    [=[                auto * cache_k_view = node->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                model_params.flat_kv_sizes.push_back(ggml_nelements(cache_k_view->view_src));
]=]
    "flat KV sizes")
  file(WRITE "${ggml_ov_directory}/ggml-decoder.cpp" "${ggml_ov_text}")
endif()

# --- 3. cache blobs carry their weights ---------------------------------------
# NPU keeps its own setting (NPUW_CACHE_DIR, a few lines above this one).
set(cache_mode_marker "grpc-asr: cache blobs carry their weights")
file(READ "${ggml_ov_directory}/ggml-openvino-extra.cpp" ggml_ov_text)
string(FIND "${ggml_ov_text}" "${cache_mode_marker}" ggml_ov_done)
if(ggml_ov_done EQUAL -1)
  asr_ggml_ov_replace(
    [=[    } else if (cache_dir && strlen(cache_dir) > 0) {
        compile_config.insert(ov::cache_dir(cache_dir));
        compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    }]=]
    "    } else if (cache_dir && strlen(cache_dir) > 0) {
        compile_config.insert(ov::cache_dir(cache_dir));
        // ${cache_mode_marker} (cmake/PatchGgmlOpenvino.cmake)
        compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SPEED));
    }"
    "cache mode")
  file(WRITE "${ggml_ov_directory}/ggml-openvino-extra.cpp" "${ggml_ov_text}")
endif()
