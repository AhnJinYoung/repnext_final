# Minor Revision: TPU Track Baseline, Target Candidate, and Figures

## 1. TPU Track Baseline Correction

The TPU-track baseline should be the native unoptimized RepNeXt run, not a
low-resolution or INT8-optimized run. The baseline therefore has the same
accuracy family as the other native baselines.

| Variant | Input | Backend | Samples | Latency avg | mIoU | Pixel acc | Decision |
|---|---:|---|---:|---:|---:|---:|---|
| Original RepNeXt GELU | 512 | PyTorch | 5 | 3659.911 ms | 0.2582 | 0.8024 | Valid sanity baseline |
| Tanh-GELU sparse rewrite | 512 | PyTorch | 5 | 3987.130 ms | 0.2585 | 0.8024 | Accuracy preserved |
| ReLU sparse rewrite | 512 | PyTorch | 5 | 3587.694 ms | 0.0027 | 0.0275 | Invalid |
| ReLU low-res INT8 | 96 -> 24 logits | CPU TFLite | 50 | 46.720 ms | 0.0037 | 0.0915 | Invalid |
| Tanh-GELU low-res PyTorch | 64 -> 16 logits | PyTorch | 50 | 873.654 ms | 0.0480 | 0.4409 | Too low |
| Tanh-GELU low-res PyTorch | 96 -> 24 logits | PyTorch | 50 | 861.202 ms | 0.0627 | 0.5205 | Too low |
| Tanh-GELU low-res INT8 | 64 -> 16 logits | CPU TFLite | 50 | 39.259 ms | 0.0056 | 0.2217 | Invalid |

The important conclusion is that ReLU replacement and post-training full INT8
collapse the model. Tanh-GELU preserves the original model behavior, so all
TPU-track follow-up work should stay on tanh-GELU.

## 2. Accuracy/Latency Target

The requested target is:

- mIoU greater than `0.15`
- latency less than `2000 ms`

The previous low-resolution sweep shows that 64px and 96px are too small. The
recorded resolution sweep points to 192px as the useful target:

| Candidate | Input | Avg latency | mIoU | Pixel acc | Decision |
|---|---:|---:|---:|---:|---|
| Tanh-GELU low-res PyTorch | 96 | 861.202 ms | 0.0627 | 0.5205 | Too low |
| Tanh-GELU low-res PyTorch | 192 | 360.469 ms | 0.1636 | 0.7228 | Meets target before INT8 |
| Tanh-GELU low-res PyTorch | 256 | 563.313 ms | 0.2150 | 0.7890 | Better accuracy, likely harder for TPU |

Therefore the best next TPU-track candidate is:

```text
RepNeXt-M5 tanh-GELU, input 192 -> 48x48 logits
```

This is the smallest currently known resolution that crosses the requested
accuracy threshold. The remaining problem is preserving the 192px behavior after
full INT8 quantization and EdgeTPU compilation. The w48 192px binary already
compiles cleanly, but it is not accuracy-valid without QAT/distillation.

## 3. Compiled Binary Status

Already available compiler-clean TPU binary:

```text
artifacts/w48_tpu/repnext_m5_tanhgelu_real_full_192_logits_full_integer_quant_dwpatched_edgetpu.tflite
```

Compiler result:

```text
960 / 960 operations mapped to EdgeTPU
1 EdgeTPU subgraph
```

Full RepNeXt 192px target build was generated under:

```text
artifacts/full_repnext_192_target/
artifacts/logs/full_repnext_192_target/
```

Generated full RepNeXt 192px binaries:

```text
artifacts/full_repnext_192_target/repnext_m5_tanhgelu_real_full_192_logits.onnx
artifacts/full_repnext_192_target/onnx2tf_tanhgelu_192_logits/repnext_m5_tanhgelu_real_full_192_logits_float32.tflite
artifacts/full_repnext_192_target/onnx2tf_tanhgelu_192_logits/repnext_m5_tanhgelu_real_full_192_logits_float16.tflite
artifacts/full_repnext_192_target/onnx2tf_tanhgelu_192_logits/repnext_m5_tanhgelu_real_full_192_logits_dynamic_range_quant.tflite
artifacts/full_repnext_192_target/onnx2tf_tanhgelu_192_logits/repnext_m5_tanhgelu_real_full_192_logits_integer_quant.tflite
artifacts/full_repnext_192_target/onnx2tf_tanhgelu_192_logits/repnext_m5_tanhgelu_real_full_192_logits_full_integer_quant.tflite
artifacts/full_repnext_192_target/repnext_m5_tanhgelu_real_full_192_logits_full_integer_quant_dwpatched.tflite
```

EdgeTPU compile result:

```text
Compilation failed due to large activation tensors in model.
```

This confirms that full RepNeXt 192px is the correct accuracy target, but cannot
be sent directly to Coral as a full model. The practical route is to train a
TPU-fit 192px student, such as the w48 graph, with distillation/QAT so it keeps
the full-model 192px behavior while remaining EdgeTPU-compilable.

## 4. New Figures

Generated figures for the revised comparison:

```text
demo/runtime_graphs/coral_tpu_latency_accuracy.png
demo/runtime_graphs/best_methods_by_track_latency_accuracy.png
demo/seg_compare/best_track_methods_comparison.png
```

The qualitative comparison now uses the same names as the runtime graphs and
includes only accuracy-valid live-demo candidates:

- Native 512
- RPi5 LiteRT 256
- TPU target 192

The demo-sample mean from the regenerated overlay figure was:

| Variant | Demo mIoU | Demo pixel acc |
|---|---:|---:|
| Native 512 | 0.3295 | 0.8127 |
| RPi5 LiteRT 256 | 0.3657 | 0.8051 |
| TPU target 192 | 0.3172 | 0.7445 |

These demo-image values are for visualization only; the benchmark tables above
remain the quantitative basis for model selection.

Runtime graphs were also regenerated with all methods below `0.15` mIoU removed.
This removes the current full-INT8 TPU binaries from the advisor-facing demo
graphs. They remain important compiler artifacts, but they should not be shown
as live segmentation demo candidates until QAT/distillation recovers accuracy.
