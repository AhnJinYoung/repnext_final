# RepNeXt Scene Understanding Optimization via DL Compiler

Final deliverable repo for optimizing RepNeXt-M5 ADE20K semantic segmentation on:

- Intel CPU: 11th Gen Intel Core i5-1135G7
- Raspberry Pi 5 ARM CPU without Coral TPU
- Raspberry Pi 5 ARM CPU with Google Coral USB Edge TPU x2

The main write-up is [report.tex](report.tex).

## Main Results

| Track | Easy name | Latency | Accuracy | Meaning |
|---|---:|---:|---:|---|
| Intel CPU | OpenVINO+LiteRT | `2210.749 ms` | `0.0034 mIoU` for ReLU/sparse family | fastest Intel compiler pipeline, not accuracy-valid |
| RPi5 ARM CPU | LiteRT 256 | `351.377 ms` | `0.2135 mIoU` | best no-training accuracy-valid demo |
| RPi5 + Coral TPU x2 | w48 TPU | `84.130 ms` | `0.0031 mIoU` | compiler-clean TPU graph, needs QAT/distillation |

The key conclusion is that the **RPi5 CPU LiteRT 256** model is the best current demo, while the **w48 TPU** model proves the full graph can map to Coral TPU but still needs training for accuracy.

## Repo Layout

```text
report.tex                         final report
demo/runtime_graphs/               report graphs
demo/seg_compare/                  demo segmentation visualizations
demo/runtime_graph_viz.py          regenerates latency/accuracy graphs
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
