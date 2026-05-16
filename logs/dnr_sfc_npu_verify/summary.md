# DNR SFC NPU Verification Summary (Final)

- Total: 22
- PASS: 22
- FAIL: 0

Validation mode:

- Import ONNX source selected per model from fixed variants:
  - `model.sim.onnx`
  - `model.sim.clipfix.onnx`
  - `model.sim.correctshape.onnx`
  - `model.sim.correctshape.clipfix.onnx`
- Optimize stage includes:
  - `--replace_non_const_fc_with_batch_matmul`
  - `--convert_nchw_to_nhwc`
- Quantize stage:
  - `uint8`, `channel`, `input_type=uint8`, `output_type=uint8`

Authoritative detailed report:

- `/home/cmj/works/ASS/logs/dnr_sfc_npu_verify/summary_full_with_replacefc_and_nchw2nhwc.md`
