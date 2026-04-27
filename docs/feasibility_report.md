# RepNeXt-M5 (ADE20K) — Coral EdgeTPU Feasibility

**Date**: 2026-04-23
**Model**: RepNeXt-M5 + FPN + SemanticFPNHead, 150 classes (ADE20K)
**Checkpoint**: `repnext_m5_ade20k.pth` (104 MB)
**Input**: `(1, 3, 512, 512)` → output `(1, 150, 512, 512)`
**ONNX**: 1708 nodes, opset 17

Repo: https://github.com/suous/RepNeXt (`suous/RepNeXt` @ v1.0)

---

## 1. TL;DR

**Verdict: high TPU compatibility (96.2%)**, a single op class (`GELU`/`Erf`) creates 55 CPU fallbacks — removable by swapping the activation to ReLU/HardSwish. Unlike ViT-based encoders (BLIP 89.3%), RepNeXt is a pure CNN: **no LayerNorm, no attention, no rank>4 tensors**. Expected outcome after activation swap: **~100% EdgeTPU op coverage**, similar to InceptionResNetV2 (100% / 160 ms in the existing pipeline).

**Phase A/C measurement status (2026-04-23)**:
- RPi5 `.200` is reachable; measured hardware currently exposes **1x Coral USB** via `lsusb` (`18d1:9302`), not 2x. Two-TPU data-parallel and pipeline rows remain pending until the second accelerator is visible.
- RPi5 PyTorch CPU baseline measured with 4 threads, `runs=5`, `warmup=1`: **3763.202 ms avg**.
- WSL has `edgetpu_compiler`; RPi5 does not expose an `edgetpu-compiler` apt package for arm64. Compile path is WSL conversion/compile -> RPi5 benchmark.
- ONNX -> TFLite conversion now requires a preprocessing patch: all 822 Conv nodes lacked `kernel_shape`; `convert_tflite.py` fills it from weight shapes before invoking onnx2tf. Float32/float16 TFLite export succeeds after this patch, but full INT8 `-oiqt` still fails inside onnx2tf's SavedModel handoff. No authoritative EdgeTPU mapping log yet.

## 2. Op coverage (ONNX graph)

| Class | Count | % | Notes |
|---|---:|---:|---|
| OK (TPU) | 1643 | 96.2% | Conv 822, Add 490, Mul 110, Div 55, Concat 54, BN 54, Split 51, Relu 7 |
| RISKY (may fall back) | 10 | 0.6% | `Resize` — bilinear upsample in the seg head (FPN + final logits upsample) |
| UNSUPPORTED (CPU fallback) | 55 | 3.2% | **`Erf` ×55** — this is `GELU` decomposed by ONNX export |
| Unknown | 0 | 0% | — |

Command to reproduce:
```bash
python export_onnx.py --size 512 --opset 17
python analyze_ops.py repnext_m5_ade20k.onnx
```

## 3. What maps where

- **Backbone ops are 100% TPU-friendly in principle**. RepNeXt-M5 uses:
  - `Conv2d` (incl. depthwise, grouped, asymmetric kernels: 3×3, 1×3, 3×1, 2×2, 7×7, 5×3, 3×5, 1×7, 7×1, 1×5, 5×1, 1×11, 11×1)
  - `BatchNorm2d` (folded into Conv after INT8 quantization)
  - `GELU` ← problem
  - residual `Add`, `torch.chunk` (→ `Split`), `torch.cat` (→ `Concat`)
- **Neck (FPN)**: 1×1 lateral convs + 3×3 fpn convs + nearest upsample + add. Fully supported.
- **Head (SemanticFPNHead)**: Conv+BN+ReLU+Upsample chains + 1×1 `conv_seg` + final bilinear upsample to input res. `Resize` (bilinear) is typically supported on EdgeTPU but counts as RISKY because of occasional output-size restrictions; same op was used fine in the InceptionResNetV2 pipeline.
- Tensors are all 4D `(B, C, H, W)` — **no rank-limit issue** (unlike MoViNet's 5D).

## 4. The 55 Erf (GELU) problem

`nn.GELU` in PyTorch (default, `approximate='none'`) exports to `x * 0.5 * (1 + Erf(x / sqrt(2)))`. `Erf` has no EdgeTPU mapping → every GELU = CPU fallback = TPU↔USB transfer. With 55 fallbacks this would behave similarly to BLIP ViT (26 LayerNorm fallbacks → 160ms on InceptionResNetV2-class encoder vs much slower on BLIP).

### Mitigations (in order of effort)

| Option | Effort | Accuracy impact | Outcome |
|---|---|---|---|
| `nn.GELU(approximate='tanh')` | trivial | ~identical | ONNX still exports `Tanh`-based; `Tanh` is EdgeTPU-supported ✅ |
| Replace GELU → ReLU, short fine-tune on ADE20K | 1–2 days | 0–1 mIoU drop expected | full 100% TPU mapping, no fallback |
| Replace GELU → HardSwish | trivial | small drop without retrain | HardSwish is RISKY — sometimes supported |
| Keep GELU, accept 55 CPU fallbacks | none | 0 | many transitions → latency likely 10–30× slower than CPU-only |

**Recommendation**: try `GELU(approximate='tanh')` first — no training cost, high chance of pushing coverage to ~100%. If that fails at quantization time, fine-tune with ReLU.

## 5. Remaining unknowns / measured blockers

- [x] RPi5 PyTorch CPU baseline: **4222.626 ms avg**, min 3812.209 ms, p95 4360.129 ms, N=50, warmup=10, 4 threads, input `(1, 3, 512, 512)`, output `(1, 150, 512, 512)`.
- [x] ONNX conversion blocker identified: Conv nodes exported without `kernel_shape`, causing onnx2tf to infer kernel rank 0. `convert_tflite.py` now patches `kernel_shape` from initializer dims.
- [ ] Full INT8 TFLite PTQ: still blocked by onnx2tf `-oiqt` SavedModel handoff after float32/float16 export succeeds; `-dgc` group-conv rewrite also failed to produce an artifact.
- [x] Float32 TFLite compiler attempt: rejected by `edgetpu_compiler` (`CONV_2D` builtin opcode version 6), so this cannot be used as a TPU mapping proxy.
- [ ] `edgetpu_compiler` log on converted INT8 TFLite: pending until full INT8 TFLite is produced.
- [ ] 2x Coral measurements: pending because `.200` currently shows only 1x Coral USB.
- [ ] Whether 1708-node model fits EdgeTPU on-chip cache (8 MB SRAM limit) vs USB parameter streaming.
- [ ] Actual RPi5 + USB Coral latency.

Expected-to-work based on the architectural analysis above; no blocker comparable to MoViNet's 5D tensors or BLIP's LayerNorm density.

## 6. Comparison with prior results

| Model | EdgeTPU op map | Blocking issue |
|---|---:|---|
| InceptionResNetV2 (current pipeline) | **100.0%** | — |
| **RepNeXt-M5 (this model)** | **96.8%** (with tanh-GELU: ~100%) | GELU→Erf (fixable) |
| BLIP ViT-B encoder | 89.3% | LayerNorm ×26, Erf ×12 |
| Depth-Anything-V2 ViT-L (next report) | ~62.8% | LayerNorm ×52, Erf ×24 |
| MoViNet-A0 | 9.5% | 5D tensors (hard architectural block) |

Measured benchmark table:

| Configuration | Avg latency (ms) | TPU op mapping | Notes |
|---|---:|---:|---|
| PyTorch CPU (RPi5, 4T) | **3763.202** | n/a | measured, N=5 |
| TFLite INT8 CPU (4T) | pending | n/a | waiting on INT8 TFLite |
| TFLite INT8 + 1x EdgeTPU | pending | pending | waiting on compiler log |
| TFLite INT8 + 2x EdgeTPU data parallel | pending | pending | `.200` currently sees 1x Coral |
| TFLite INT8 + 2x EdgeTPU pipeline split | pending | pending | split model not produced |
| TVM(A76 tuned) CPU subgraph | pending | pending | Phase D after Phase C decision |

## 7. Files in this folder

```
RepNeXt-tpu/
├── RepNeXt/                       # upstream repo (suous/RepNeXt)
├── repnext_m5_ade20k.pth          # checkpoint (2100 backbone + 60 neck/head keys)
├── export_onnx.py                 # self-contained builder + loader + ONNX export
├── analyze_ops.py                 # ONNX op enumerator vs EdgeTPU ref table
├── repnext_m5_ade20k.onnx         # 1708 nodes, 512×512
├── repnext_m5_ade20k.onnx.data    # external weights (102.9 MB)
├── repnext_ops.json               # machine-readable per-op breakdown
└── feasibility_report.md          # this file
```
