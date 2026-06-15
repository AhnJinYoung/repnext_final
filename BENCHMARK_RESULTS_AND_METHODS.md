# Benchmark Results and Methods

This file summarizes the final reported results. No new benchmark numbers are
introduced here; values are copied from `benchmark/results/final/`.

## Implemented Optimization Methods

### Intel CPU Track

- Used TVM/Relay-style graph partitioning concepts to isolate expensive RepNeXt
  regions and reason about fusion boundaries.
- Used PyTorch compile as a full-model CPU compiler baseline.
- Used OpenVINO for CPU graph compilation of ONNX partitions.
- Used LiteRT/TFLite INT8 for the middle segment where quantized kernels gave
  useful speedups.
- Applied persistent interpreter/session reuse to remove per-frame setup cost.
- Used ReLU sparse-equivalent variant for compiler compatibility experiments.
- Measured stage-level latency to expose prefix/middle/suffix bottlenecks.

### Raspberry Pi 5 CPU-Only Track

- Exported RepNeXt logits models to ONNX, then converted to LiteRT/TFLite.
- Evaluated float32, float16, dynamic-range quantized, integer-quantized, and
  full-INT8 variants.
- Used lower input resolutions to reduce activation size and memory bandwidth.
- Kept CPU-only variants as the accuracy-preserving fallback when EdgeTPU INT8
  quantization damaged segmentation quality.

### Raspberry Pi 5 + Coral EdgeTPU Track

- Replaced original GELU with tanh-GELU approximation to preserve accuracy while
  staying closer to compiler-supported operations than hard ReLU.
- Used real ADE20K calibration tensors instead of random calibration data.
- Exported full low-resolution logits models so there is no CPU prefix/suffix in
  the TPU-compiled low-res path.
- Patched depthwise-shaped TFLite Conv2D ops into DepthwiseConv2D so the EdgeTPU
  compiler could map them.
- Compiled full INT8 models with `edgetpu_compiler`.
- Added the w48 shrink track to fit the whole graph into one EdgeTPU subgraph.

## Key Accuracy Results

| Model / Method | Backend | Input | mIoU | Pixel Acc. | Source |
|---|---:|---:|---:|---:|---|
| Native GELU | PyTorch | 512 | 0.2224 | 0.7894 | `ade20k_val40_pytorch_gelu_512.json` |
| Tanh-GELU | PyTorch | 512 | 0.2225 | 0.7891 | `ade20k_val40_pytorch_tanhgelu_512.json` |
| ReLU replacement | PyTorch | 512 | 0.0034 | 0.1142 | `ade20k_val40_pytorch_relu_sparse_512.json` |
| Low-res dynamic-range | LiteRT CPU | 256 | 0.2135 | 0.7874 | `ade20k_val40_256_dynamic_range_quant.json` |
| Low-res float16 | LiteRT CPU | 256 | 0.2155 | 0.7895 | `ade20k_val40_256_float16.json` |
| Full INT8 | LiteRT CPU | 256 | 0.0027 | 0.0427 | `ade20k_val40_256_full_integer_quant.json` |
| Full low-res 96 integer | LiteRT CPU | 96 | 0.0086 | 0.2521 | `ade20k_val40_96_integer_quant.json` |
| W48 full INT8 | LiteRT CPU | 192 | 0.0031 | 0.2007 | `ade20k_val40_w48_192_full_int8_tflite.json` |

The main accuracy conclusion is that resolution reduction alone is acceptable up
to 256px, but full INT8 quantization severely damages RepNeXt segmentation
quality without QAT or distillation.

## Key Latency Results

| Track | Model / Method | Avg Latency |
|---|---|---:|
| Intel CPU | PyTorch ReLU sparse baseline | 3972.2 ms |
| Intel CPU | PyTorch compile full model | 2608.0 ms |
| Intel CPU | OpenVINO + LiteRT partitioned pipeline | 2210.7 ms |
| RPi CPU-style LiteRT | 256 dynamic-range quantized | 316.6 ms |
| RPi CPU-style LiteRT | 256 float16 | 355.7 ms |
| RPi CPU-style LiteRT | 256 full INT8 | 148.8 ms |
| RPi CPU-style LiteRT | 96 integer quantized | 74.2 ms |
| TPU-ready LiteRT | W48 192 full INT8 before EdgeTPU compile | 46.6 ms |

## Uploaded Compiled Binaries

The regenerated deployable binaries are under `artifacts/`.

- Full RepNeXt 96px EdgeTPU:
  - `artifacts/full_repnext/repnext_m5_tanhgelu_real_full_96_logits_full_integer_quant_dwpatched_edgetpu.tflite`
  - Compiler result: 2519 / 2519 ops mapped to EdgeTPU.

- W48 192px EdgeTPU:
  - `artifacts/w48_tpu/repnext_m5_tanhgelu_real_full_192_logits_full_integer_quant_dwpatched_edgetpu.tflite`
  - Compiler result: 960 / 960 ops mapped to EdgeTPU.

- CPU LiteRT binaries:
  - `artifacts/full_repnext/onnx2tf_tanhgelu_96_logits/*.tflite`
  - `artifacts/w48_tpu/onnx2tf_tanhgelu_192_logits/*.tflite`

- ONNX binaries for Intel/OpenVINO-style execution:
  - `artifacts/full_repnext/*.onnx`
  - `artifacts/w48_tpu/*.onnx`

## Important Limitation

TVM contributed the compiler-oriented graph analysis and optimization direction,
but it was not used as the final EdgeTPU binary exporter because TVM does not
provide a direct route from optimized Relay graphs to EdgeTPU-compatible
fully-quantized `.tflite`. For the Coral path, the final practical export chain
was ONNX -> onnx2tf/LiteRT -> depthwise patch -> EdgeTPU compiler.
