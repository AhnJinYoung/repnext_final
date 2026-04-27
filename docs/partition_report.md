# RepNeXt-M5 ADE20K Partition Report

**Target**: Raspberry Pi 5 BCM2712, 4x Cortex-A76 @ 2.4GHz, NEON+FP16+dotprod, 2x Coral USB
**Compiler target for CPU subgraphs**: `llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a76 -mattr=+neon,+fp16,+dotprod`

## Summary

| Configuration | Avg latency (ms) | TPU op mapping | Notes |
|---|---:|---:|---|
| PyTorch CPU (RPi5, 4T) | 4222.626 | n/a | baseline |
| TFLite INT8 CPU (4T) | pending | n/a | CPU interpreter |
| TFLite INT8 + 1x EdgeTPU | pending | pending | automatic delegate partition |
| TFLite INT8 + 2x EdgeTPU data parallel | pending | pending | batch 2 throughput: pending |
| TFLite INT8 + 2x EdgeTPU pipeline split | pending | pending | requires split submodels |
| TVM(A76 tuned) CPU subgraph | pending | pending | Phase D, cache-bounded tile search |

## Phase A Status

Measured on the `.200` RPi5 target:

- RPi5 connection: OK.
- Coral USB detected by `lsusb`: 1 device currently visible, not 2. Data-parallel and pipeline-split 2x TPU measurements need the second Coral visible before they are meaningful.
- `edgetpu_compiler`: available in WSL, not available as an arm64 apt package on the RPi5 Coral repo.
- ONNX conversion blocker: RepNeXt ONNX Conv nodes lacked `kernel_shape`; `convert_tflite.py` now patches this from initializer weight shapes before calling onnx2tf.
- Current remaining blocker: onnx2tf can produce float32/float16 TFLite, but full INT8 export fails because RepNeXt's group/depthwise convolutions prevent SavedModel output. The `-dgc` fallback path ran for several minutes and exited without producing an artifact.
- Float TFLite is not a TPU substitute: `edgetpu_compiler` rejects the float32 TFLite (`CONV_2D` builtin opcode version 6), confirming that full INT8 PTQ is required before compiler mapping can be measured.

## EdgeTPU Compiler Op Placement

| Op | Count | Placement | Compiler status |
|---|---:|---|---|
| pending | pending | pending | compiler log not available or unparsable |

## 2x TPU Scheduling Notes

Data parallel mode binds two full interpreters with `make_interpreter("model.tflite,:0")` and `make_interpreter("model.tflite,:1")`. It improves throughput for independent frames but does not reduce single-frame latency.

Pipeline split mode needs two compiled submodels with a compatible boundary tensor. For RepNeXt, the useful split point is after a large stage boundary only if the intermediate tensor transfer is cheaper than the saved compute time. This remains pending until split TFLite graphs are produced.

## CPU Fallback / BYOC Notes

RepNeXt's expected fallback is GELU exported as `Erf`. The first mitigation to measure is tanh-GELU export or ReLU fine-tuning because removing CPU fallback transitions can beat TVM on tiny activation-only subgraphs. If fallback remains, TVM should be limited to large conv or matmul blocks and built for Cortex-A76 with cache-bounded schedules.
