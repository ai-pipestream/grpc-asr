# openvino: store weights in the plugin cache on CPU and GPU

Patch: `ggml-openvino-cache-mode.patch`. This one is a report as much as a
patch: the change is one line and the maintainers may prefer another remedy.

## What was seen

With `GGML_OPENVINO_CACHE_DIR` set, the backend passes `ov::cache_dir` and
`ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE)` to `compile_model` on CPU and
GPU. That is OpenVINO's weightless cache: a blob holds the compiled kernels,
and the weights are read from the model again when the blob is loaded. The
models here are built in memory by the frontend, with constants that share
ggml's weight buffers.

On an Arc B70, with whisper.cpp graphs (two models in one process, two
states each; f16 weights), a process that starts from the cache does not
behave like the one that filled it:

- **Different results on first use.** The warmup decodes one second of
  silence, greedy, on each state. In the process that compiles the graphs,
  the decoder graphs of the smaller model come to 1296 compute nodes per
  state. In a process that loads them from the cache they come to 190080 on
  the first state and 1296 on the second: the first state's decode produces
  something else, fails whisper's quality check and goes through every
  fallback temperature. The second state uses the same compiled encoder,
  cross and decoder models as the first. The larger model does the same.
- **A crash.** In the same process, `compile_model` for the conv graph of
  the second model died with SIGSEGV in `memcpy`, called from the NEO
  runtime (`libigdrcl.so`) under the GPU plugin, reading a host buffer near
  the top of the heap:

  ```
  #0  libc.so.6 (memmove, AVX-512)
  #1  libigdrcl.so
  #4  libopenvino_intel_gpu_plugin.so
  #17 ov::Core::compile_model(model, RemoteContext, config)
  #18 naive_compute(ggml_cgraph*, ...)
  #19 ov_graph_compute_dynamic
  #24 whisper_encode_internal
  ```

  It depends on the models and their order. From the cache, the smaller
  model alone crashed (one boot of one) and the smaller then the larger
  crashed (seven boots of seven); the larger alone and the larger then the
  smaller did not (one boot each). A boot that fills the cache never
  crashed.
- The cache directory gains new small blobs, the size of the conv graphs',
  on boots that should only have loaded them, so some graph's key is not
  the same from one process to the next.

Without a cache directory (two boots) and with `OPTIMIZE_SPEED` (four boots,
three of them from the cache), every state runs the same 1296 decoder nodes
and nothing crashes, on the same machine with the same models. With
`OPTIMIZE_SPEED` the boot from the cache is also the fast one: 0.35 s of
warmup for the first state of the smaller model, against 1.6 s when
compiling and 8.5 s from the weightless cache (its fallback decode
included).

Whether the fault is in the plugin's weightless import or in how the
frontend's shared-memory constants reach it was not established: the plugin
frames have no symbols in the release archive.

## The change

`OPTIMIZE_SPEED` in place of `OPTIMIZE_SIZE` in the CPU and GPU branch. NPU
keeps its own setting. The blobs then hold their weights and take about as
much disk as the weights of each compiled graph (291 MB for two models whose
files are 226 MB), which is what
`GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR` already does for its own blobs.

An environment switch that keeps `OPTIMIZE_SIZE` as the default would serve
grpc-asr as well, if the weightless cache is known to be sound for the
llama.cpp graphs.
