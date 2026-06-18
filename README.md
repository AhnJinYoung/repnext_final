# RepNeXt Scene Understanding Optimization via DL Compiler

Final deliverable repo for optimizing RepNeXt-M5 ADE20K semantic segmentation on:

- Intel CPU: 11th Gen Intel Core i5-1135G7
- Raspberry Pi 5 ARM CPU without Coral TPU
- Raspberry Pi 5 ARM CPU with Google Coral USB Edge TPU x2

The main write-up is [report.tex](report.tex).

Korean end-to-end reproduction and user manual:
[USER_MANUAL_KO.txt](USER_MANUAL_KO.txt).

## Main Results

| Track | Easy name | Latency | Accuracy | Meaning |
|---|---:|---:|---:|---|
| Intel CPU | OpenVINO+LiteRT | `2210.749 ms` | `0.0034 mIoU` for ReLU/sparse family | fastest Intel compiler pipeline, not accuracy-valid |
| RPi5 ARM CPU | LiteRT 256 | `351.377 ms` | `0.2135 mIoU` | best no-training accuracy-valid demo |
| RPi5 + Coral TPU x2 | TPU target 192 | `360.469 ms` | `0.1636 mIoU` | accuracy-valid TPU-track target; INT8 EdgeTPU compile still needs QAT/distillation |

The key conclusion is that the **RPi5 CPU LiteRT 256** model is the best current live-demo path. The **TPU target 192** model is the useful TPU-track accuracy target: it keeps mIoU above 0.15 at about 360 ms/frame, but preserving that accuracy after full INT8 EdgeTPU compilation still needs QAT/distillation.

## Repo Layout

```text
report.tex                         final report
demo/runtime_graphs/               report graphs
demo/seg_compare/                  demo segmentation visualizations
demo/runtime_graph_viz.py          regenerates latency/accuracy graphs
demo/video_segmentation_demo.py    image/video segmentation benchmark runner
demo/make_realtime_synced_video.py source-time-synced comparison video builder
demo/make_source_native_rpi5_realtime_video.py
                                    source/native/RPi5 three-panel demo video
benchmark/ade20k_accuracy_benchmark.py
benchmark/results/                 selected final benchmark JSONs
conversion/                        export, calibration, and TFLite patch helpers
qat_tpu_pipeline/                  GPU-server QAT/distillation pipeline
scripts/export_compile_edgetpu.sh  checkpoint -> INT8 TFLite -> Edge TPU binary
```

## QAT / Distillation for TPU Accuracy

Use [qat_tpu_pipeline](qat_tpu_pipeline) on a GPU server:

```bash
cd repnext-optimization
export ADE_ROOT=/path/to/ade20k
export TEACHER_WEIGHTS=repnext_m5_ade20k.pth
bash qat_tpu_pipeline/run_all.sh
```

The final compiled binary will be:

```text
build/w48_192_qat_edgetpu/*_edgetpu.tflite
```

See [qat_tpu_pipeline/README.md](qat_tpu_pipeline/README.md) for detailed commands and overrides.

## Regenerate Figures

```bash
python3 demo/runtime_graph_viz.py
```

Generated figures are written to:

```text
demo/runtime_graphs/
```

## Video Segmentation Demo

Install video I/O helpers once if your Python does not already have them:

```bash
python3 -m pip install -r demo/video_requirements.txt
```

Run the slow native baseline:

```bash
python3 demo/video_segmentation_demo.py run \
  --input-video input.mp4 \
  --output-video demo/video_runs/native512_overlay.mp4 \
  --output-frames demo/video_runs/native512_frames \
  --metrics demo/video_runs/native512_metrics.json \
  --name "Native 512" \
  --backend pytorch \
  --weights /workspace/tvm/handoff/repnext_m5_ade20k.pth \
  --activation gelu \
  --size 512 \
  --max-frames 30
```

For LiteRT environments without video I/O libraries, first extract frames with
system Python:

```bash
python3 demo/video_segmentation_demo.py extract \
  --input-video input.mp4 \
  --output-frames demo/video_runs/extracted_frames \
  --max-frames 120
```

Run the optimized LiteRT binary candidate on those frames:

```bash
/workspace/tvm/local-convert-env/bin/python3 demo/video_segmentation_demo.py run \
  --input-frames demo/video_runs/extracted_frames \
  --output-frames demo/video_runs/litert192_frames \
  --metrics demo/video_runs/litert192_metrics.json \
  --name "TPU target 192 LiteRT" \
  --backend tflite \
  --model artifacts/full_repnext_192_target/onnx2tf_tanhgelu_192_logits/repnext_m5_tanhgelu_real_full_192_logits_dynamic_range_quant.tflite \
  --size 192 \
  --fps 30
```

Encode the optimized overlay frames afterward:

```bash
python3 demo/video_segmentation_demo.py encode \
  --input-frames demo/video_runs/litert192_frames \
  --output-video demo/video_runs/litert192_overlay.mp4 \
  --fps 3
```

The module writes per-frame latency and FPS to the `--metrics` JSON file. Use
those JSONs to compare choppy native inference against the optimized model.

```bash
python3 demo/video_segmentation_demo.py summarize \
  demo/video_runs/native512_metrics.json \
  demo/video_runs/litert192_metrics.json \
  --out demo/video_runs/video_demo_comparison.md
```

For presentation videos, use the realtime-sync builders. They keep the original
source-video timeline fixed. If a model is slower than the source FPS, the
panel holds the latest completed segmentation frame, so the native baseline
looks choppy while the optimized path updates more often.

Native baseline vs RPi5 CPU only:

```bash
python3 demo/make_native_vs_rpi5_realtime_video.py \
  --root demo/video_runs_rpi_only/city_follow_walk_10s \
  --output demo/video_runs_rpi_only/city_follow_walk_10s/city_follow_walk_10s_native_vs_rpi5_realtime_sync_24fps.mp4 \
  --fps 24
```

Source video + native baseline + RPi5 CPU:

```bash
python3 demo/make_source_native_rpi5_realtime_video.py \
  --root demo/video_runs_rpi_only/city_follow_walk_10s \
  --output demo/video_runs_rpi_only/source_native_rpi5_3panel/city_follow_walk_source_native_rpi5_24fps.mp4 \
  --fps 24
```

## Accuracy Benchmark

```bash
python3 benchmark/ade20k_accuracy_benchmark.py \
  --ade-root /path/to/ade20k \
  --backend tflite \
  --model /path/to/model.tflite \
  --size 256 \
  --limit 40 \
  --threads 4 \
  --normalize zero-one \
  --out benchmark/results/my_accuracy.json
```

## Notes

Large generated artifacts are intentionally not committed:

- ONNX/TFLite build products
- teacher caches
- local virtualenvs
- full conversion logs
- model checkpoints

They can be regenerated from the scripts above.
