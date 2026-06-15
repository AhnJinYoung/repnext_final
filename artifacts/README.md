# Compiled Artifacts

This directory contains regenerated deployment artifacts for the final RepNeXt
optimization report. The original ignored build outputs were removed during the
repo cleanup, so these files were rebuilt from the tracked conversion scripts
and the same model settings used in the report.

## Tracks

- Intel CPU / OpenVINO input:
  - `full_repnext/repnext_m5_tanhgelu_real_full_96_logits.onnx`
  - `full_repnext/repnext_m5_tanhgelu_real_full_256_logits.onnx`
  - `w48_tpu/repnext_m5_tanhgelu_real_full_192_logits.onnx`
  - `w48_tpu/repnext_m5_tanhgelu_real_full_256_logits.onnx`

- Raspberry Pi 5 CPU-only / LiteRT:
  - `full_repnext/onnx2tf_tanhgelu_96_logits/*_float16.tflite`
  - `full_repnext/onnx2tf_tanhgelu_96_logits/*_dynamic_range_quant.tflite`
  - `full_repnext/onnx2tf_tanhgelu_96_logits/*_integer_quant.tflite`
  - `w48_tpu/onnx2tf_tanhgelu_192_logits/*_float16.tflite`
  - `w48_tpu/onnx2tf_tanhgelu_192_logits/*_dynamic_range_quant.tflite`

- Raspberry Pi 5 + Coral EdgeTPU:
  - `full_repnext/repnext_m5_tanhgelu_real_full_96_logits_full_integer_quant_dwpatched_edgetpu.tflite`
  - `w48_tpu/repnext_m5_tanhgelu_real_full_192_logits_full_integer_quant_dwpatched_edgetpu.tflite`

## Compiler Results

- Full RepNeXt 96px EdgeTPU:
  - `2519 / 2519` operations mapped to EdgeTPU.
  - One EdgeTPU subgraph.
  - Compiler log: `logs/full_repnext/compile_96.log`.

- W48 192px EdgeTPU:
  - `960 / 960` operations mapped to EdgeTPU.
  - One EdgeTPU subgraph.
  - Compiler log: `logs/w48_tpu/compile_192.log`.

## Notes

- `*_dwpatched.tflite` files are full INT8 LiteRT models after the depthwise
  Conv2D-to-DepthwiseConv2D compatibility patch.
- `*_edgetpu.tflite` files are the binaries to run with the Coral EdgeTPU
  delegate.
- Benchmark result JSON files are kept under `benchmark/results/final/`.
- Report/demo visualizations are kept under `demo/`.
