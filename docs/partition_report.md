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

## Iter 3 Segment Split Result

Forced EdgeTPU compiler segmentation on the stage2 downsample TFLite was tested with
`-n 2` and `-n 4`.

| Configuration | Compile mapping | Measured latency |
|---|---|---:|
| single segment, 1x Coral | 448 TPU ops / 116 CPU ops | 1464.8 ms |
| `-n 2`, TPU0 -> TPU1 sequential pipeline | 27 TPU ops / 537 CPU ops | **758.8 ms** |
| `-n 4` | 48 TPU ops total, not benchmarked as a 4-stage chain | pending |

The compiler warns that `num_segments=1` is recommended, and the split segments map
poorly by op count. The measured two-Coral pipeline is still a useful BYOC candidate
for this isolated partition, but full end-to-end validation must include host tensor
copy and boundary conversion overhead.

Iter 4 chain runner results:

| Split | Avg latency | p95 | Boundary handling |
|---|---:|---:|---|
| `-n 2` | **757.0 ms** | **759.2 ms** | single tensor boundary |
| `-n 3` | 766.6 ms | 775.3 ms | single tensor boundaries |
| `-n 4` | 772.9 ms | 779.4 ms | multi-input/multi-output boundaries by tensor name |

Measured tensor set/get/cast overhead was small (roughly 1-3 ms total depending on
split count), so the slowdown from `-n 3` and `-n 4` is mostly extra segment execution
and less favorable compiler partitioning rather than Python tensor copying.

## Iter 5 Exact Layout Rewrite

The safe rewrite target was `Transpose(p) -> Concat(axis=a) -> Transpose(inv_p)`.
This was replaced with `Concat(axis=p[a])`, preserving tensor values exactly.

| Model | Total ops | TRANSPOSE ops | EdgeTPU ops | CPU ops | Avg latency |
|---|---:|---:|---:|---:|---:|
| original stage2 single segment | 564 | 116 | 448 | 116 | 1472.8 ms |
| concat-folded stage2 single segment | 553 | 105 | 448 | 105 | **1365.4 ms** |

CPU TFLite output matched byte-for-byte after rewrite on a random int8 input. The
same rewrite was not kept for the fastest `-n 2` chain because it changed the forced
segment boundary and measured slower (768.5 ms vs 757.0 ms).
