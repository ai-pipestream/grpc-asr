# openvino: recompile when whisper.cpp replaces a flat KV buffer

Patch: `ggml-openvino-flat-kv-reuse.patch`.

## What happens

A whisper.cpp decode fails on the backend with

```
GGML OpenVINO backend ov::Exception: ...
Can't set the input tensor with index: 4, because the model input
(shape=[1,1,1,786432]) and the tensor (shape=(1.1.1.5505024)) are incompatible
whisper_full_with_state: failed to decode
```

(the CPU plugin's wording; the GPU plugin reports the same two shapes). The
two sizes are whisper's self-attention cache for one decoder and for seven.

whisper.cpp keeps its KV caches as flat `[N, 1, 1, 1]` buffers.
`get_graph_input_shape` gives such a buffer a static shape, and the offsets
of the layers' views into it are fixed in the graph. A state starts with
room for one decoder. When a decode needs more, `whisper_full_with_state`
frees `kv_self` and allocates a larger one ("recreating KV cache"): for beam
search, and for the temperature fallback of a greedy decode, which samples
with `best_of` decoders.

Each backend instance remembers the last graph it computed under
`graph_key`: node count, leaf count, first and last node name. The decoder
graph built over the new buffer has the same key as the one built over the
old buffer, and `can_reuse_dynamically()` compares rope parameters and
recurrent slots only. If the two graphs are computed back to back, the
request compiled for the small buffer is reused and OpenVINO refuses the
larger tensor.

Beam search does not show it: the cache grows before the first decoder graph
of the transcription, right after the conv, encoder and cross graphs, which
have other keys and evict the entry. A greedy decode that falls back to a
higher temperature does: decoder graph, larger cache, decoder graph. It
needs a state that has only ever decoded with one decoder, so it is the
first fallback on a state that fails.

## Reproducing it

On any device, on a fresh state: greedy sampling with a quality check that
cannot pass, so the first result is rejected and decoded again with five
decoders.

```c
whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
p.logprob_thold = 1.0f;   // no average log-probability reaches 1
p.no_speech_thold = 2.0f; // and no no-speech probability excuses it
p.temperature_inc = 1.0f; // temperatures 0 and 1
p.greedy.best_of = 5;
whisper_full_with_state(ctx, state, p, samples, n_samples); // returns -8
```

This fails on the OpenVINO CPU device with the message above, every time.
It was first seen on an Arc B70 as a server whose warmup (one second of
silence, greedy) failed on some boots and not on others: silence sits near
whisper's fallback thresholds.

## The change

`ModelParams` gets `flat_kv_sizes`: the element counts of the flat KV
buffers the attention layers read, in graph order, filled in
`compute_llm_params` where the whisper attention patterns (cases 4 and 5)
are recognized. `can_reuse_dynamically()` and `can_reuse_statically()`
compare it. When a buffer changes size the graph takes the cold path, whose
`compiled_graph_key` already covers every tensor shape and the identity of
the weight buffers, so the graph for the larger buffer is compiled once and
shared between contexts like any other. Graphs without flat KV buffers have
an empty list and behave as before.

## How it was checked

The decode above is part of grpc-asr's `openvino-full-test`: it failed
before the patch on the OpenVINO CPU device and passes after it, on the CPU
device and on the Arc B70, for f16 and q5_0 weights; the greedy and beam-5
transcriptions that follow on the same pool still match the CPU backend's.
