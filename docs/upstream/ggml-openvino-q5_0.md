# openvino: support Q5_0 weights

Patch: `ggml-openvino-q5_0.patch`.

## What is missing

`supported_types` in `ggml-openvino.cpp` has Q4_0, Q4_1, Q5_1, Q8_0 and the K
types, and no Q5_0, and neither do the extraction switch in `ggml-quants.cpp`,
the layout switch in `ggml-openvino-extra.cpp` or the weight types in
`ggml-decoder.cpp`. An op over a Q5_0 tensor is therefore left to the CPU
backend. Q5_0 is what most published whisper.cpp quantizations are stored
as, and whisper.cpp aborts when it builds a state over weights the selected
backend does not take, so those files cannot be used with the backend at
all.

## The change

A Q5_0 block is 22 bytes: an fp16 scale `d`, 32 bits holding the fifth bit of
each value, and 16 bytes of low nibbles. A weight is `d * (q - 16)` with `q`
in [0, 31]. That is a symmetric format, so it fits the layout the backend
already builds for Q8_0: signed i8 weights (`q - 16`, in [-16, 15]), one f16
scale per group of 32, no zero point, `Convert -> Multiply` as the
decompression subgraph. The patch adds:

- `extract_q5_0_data` in `ggml-quants.cpp`, which unpacks the 5-bit values to
  i8 (the same bit handling as `extract_q5_1_data`, minus the zero point);
- Q5_0 next to Q8_0 in `ggml_openvino_get_extracted_layout` (`is_symmetric`,
  group 32) and in both switches of `extract_quantized_weights`;
- Q5_0 in `supported_types` and in `create_weight_node`'s weight types;
- Q5_0 next to Q5_1 and Q8_0 where 3D expert weights in a grouped 8-bit
  layout are requantized to 4 bits on GPU, since it now has that layout;
- one line in `docs/backend/OPENVINO.md`.

The weights take one byte each on the device, as Q5_1 and Q8_0 do.

## How it was checked

- **Values.** A Q5_0 tensor written byte by byte, with every `q` at every
  position of a block, both signs of scale, and a zero scale, is read back
  through the backend and compared with ggml's CPU dequantizer
  (`ggml_get_type_traits(GGML_TYPE_Q5_0)->to_float`). The scheduler's
  placement is checked too, so the op really ran on the OpenVINO device.
  - GET_ROWS of every row: 3840 of 3840 values equal bit for bit, on the CPU
    device and on the Arc B70, with the weights marked as weights before the
    load (extraction at `set_tensor`) and after it (whisper.cpp's order).
  - MUL_MAT with the identity: equal at the device's inference precision.
    Arc B70 (f16): largest relative difference 3.9e-4, inside one f16
    rounding of the product. Ryzen 9950X (the CPU plugin picks bf16 there):
    3.5e-3, inside one bf16 rounding.
- **A model.** A Whisper model quantized to q5_0 with whisper.cpp's own
  `ggml_common_quantize_0` runs with all four graphs on the device (every
  compute node checked through the scheduler) and transcribes the JFK sample
  to the same words as the ggml CPU backend does from the same file, greedy
  and with beam 5, on the CPU device and on the Arc B70.

## Not checked

`test-backend-ops` was not run, so Q5_0 in ops other than MUL_MAT and
GET_ROWS is untested, and so are 3D Q5_0 expert weights (MUL_MAT_ID): no
model with such weights was at hand. If GET_ROWS with `ne[0] == 256` turns
out to trip the 1e-7 NMSE threshold for Q5_0 as it does for the asymmetric
types, it belongs in that rejection list; Q8_0, which has the same symmetric
f16 arithmetic, is not in it.
