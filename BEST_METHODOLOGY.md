# Best Methodology — RepNeXt-M5 ADE20K

가장 빠른 (또는 가장 잘 매핑되는) 설정을 트랙별로 기록.
각 entry 갱신 시 직전 best 는 "Previous best" 로 한 줄 강등.

---

## CPU best

**현재 best: PyTorch `torch.compile` (default inductor) — 2608.0 ms / p95 2706.5 ms**
(full RepNeXt-M5, 1×3×512×512, RPi5 Cortex-A76 ×4, threads=4)

### 재현 방법
```bash
ssh rpi5@<RPI5_IP>
source ~/coral-env/bin/activate           # Python 3.9 + torch 2.8.0+cpu
cd ~/creative_design/RepNeXt-tpu          # contains RepNeXt/, export_onnx.py
python ~/repnext-pipeline/runs/20260427_iter1_pytorch_compile/pytorch_compile_baseline.py \
    --weights ~/repnext-pipeline/repnext_m5_ade20k.pth \
    --shape 1,3,512,512 --threads 4 --mode compile \
    --tag pt_compile --out result.json --warmup 10 --runs 50
```

### 핵심 설정
- `torch.compile(model)` — default backend (inductor)
- `torch.set_num_threads(4)` — A76 모든 코어 사용
- `torch.inference_mode()` 컨텍스트
- 첫 호출에서 컴파일 트리거 (≈92 s) → warmup 10 → 측정 50

### 측정 결과
| metric | value |
|--------|------:|
| avg_ms | 2608.0 |
| p50_ms | 2625.6 |
| p95_ms | 2706.5 |
| min_ms | 2410.8 |
| compile_setup_ms | 4396.6 (graph capture) |
| first_run_ms | 92199.7 (cold inductor compile) |

**Baseline 대비 개선:** PyTorch eager 4-thread = 4222 ms → torch.compile = 2608 ms (**1.62× 빠름**).

### Previous best
- _없음_ (iter 1 신규 baseline)

### 참고: 부분 그래프 측정 (stage2 downsample partition only, 1×3×512×512)
- TVM 0.18.0 LLVM `cortex-a76 +v8.2a +fullfp16 +dotprod` opt_level=3, **튜닝 없음** = **783.9 ms**
- 결과 파일: `benchmark/results/iter1_cpu_tvm_baseline_stage2.json`
- full model 로 확장 후 재측정 필요 — full ONNX (`*_relu_sparse_equiv_simplified_kernelshape.onnx`)
  를 RPi5 로 복사하여 동일 절차로 재컴파일 예정 (iter 2 후보).

---

## TPU best

**현재 best: stage2 downsample partition int8+dwpatch, single Coral USB — 1464.8 ms / p95 1523.8 ms**
(input 1×512×512×3 int8; 친구가 올린 4/24 17:32 measurement 1425 ms 와 4/27 재측정 1464 ms 평균)

### 재현 방법
```bash
ssh rpi5@<RPI5_IP>
source ~/coral-env/bin/activate
python3 ~/repnext-pipeline/runs/20260427_iter1_tpu_search/tpu_compare_tflites.py \
    --tflite ~/repnext-pipeline/repnext_stage2_int8_dwpatched_edgetpu.tflite \
    --device 0 --warmup 10 --runs 50 \
    --tag tpu_dwpatched --out result.json
```

### 핵심 설정 (산출 파이프라인)
1. PyTorch `RepNeXtSeg().load_state_dict(repnext_m5_ade20k.pth)`.
2. `export_partition_onnx.py` 로 stage2 downsample partition 만 ONNX export
   (입력 `1×3×512×512`, 출력 stage2 마지막 BN/ReLU).
3. onnxsim + kernelshape 정렬 패치 → `repnext_m5_relu_tpu_stage2_downsample_512_simplified_kernelshape.onnx`.
4. onnx → TF SavedModel (`onnx2tf`, NHWC layout) → `saved_model_repnext_stage2_int8_tf/`.
5. TFLite full integer quantization (calib `calib_512_50_nhwc_float32.npy`) →
   `repnext_m5_..._full_integer_quant.tflite`.
6. **Depthwise patch** (`patch_depthwise_tflite.py`) — 일부 dw conv 패턴을 EdgeTPU
   친화 형태로 재작성 → `repnext_stage2_int8_dwpatched.tflite`.
7. `edgetpu_compiler -s` → `repnext_stage2_int8_dwpatched_edgetpu.tflite` (5.77 MB).

### 측정 결과 (4/27 재측정)
| metric | value |
|--------|------:|
| avg_ms | 1464.8 |
| p50_ms | 1454.1 |
| p95_ms | 1523.8 |
| min_ms | 1436.6 |
| sha8 | `ea017c4a67cabd7c` |

> 같은 tflite 의 4/24 측정값(친구) 과 4/27 측정값(현재) 이 +40 ms 차이 — Coral USB 온도/USB
> bandwidth 변동으로 추정. p95-p50 = 70 ms (4.8%) 로 잡음 임계 내.

### 비교 (deprecated 후보)
- `repnext_m5_relu_tpu_stage2_downsample_512_simplified_kernelshape_full_integer_quant_edgetpu.tflite`
  (1.9 MB, no-dwpatch) — 측정 시 `RuntimeError: Input tensor 101 lacks data` 로 실패.
  TFLite 변환 후 일부 input 이 변수로 남아있는 듯, deprecated.

### Previous best
- _없음_ (iter 1 신규 baseline; 친구의 4/24 1425 ms 는 같은 tflite 의 첫 측정으로 본 entry 의 일부)

## Iter 3 candidate: 2-Coral pipeline split for stage2 partition

**New fastest measured stage2-only TPU path: `edgetpu_compiler -n 2` split + TPU0 -> TPU1 pipeline —
758.8 ms / p95 764.1 ms** (stage2 downsample partition only, not full E2E).

Reproduction:
```bash
# WSL/host, because RPi5 compiler wrapper payload may be missing under /tmp
edgetpu_compiler -s -n 2 -o iter3_segments/n2 \
  repnext_m5_relu_sparse_stage2_downsample_512_int8_dwpatched.tflite

# RPi5
source ~/coral-env/bin/activate
python ~/repnext-pipeline/benchmark.py \
  --skip-pytorch --edgetpu __none__.tflite --tflite __none__.tflite \
  --split-a ~/repnext-pipeline/iter3_stage2_seg0of2_edgetpu.tflite \
  --split-b ~/repnext-pipeline/iter3_stage2_seg1of2_edgetpu.tflite \
  --devices 0,1 --warmup 10 --runs 50 \
  --out ~/repnext-pipeline/runs/20260427_iter3_pipeline_split_n2_50runs.json
```

Important caveat: compiler op mapping is poor after forced segmentation
(27 TPU ops / 537 CPU ops total across the two segments), and the compiler warns that
one segment is recommended. The measured latency is still strong for this isolated
partition, but full BYOC E2E needs explicit tensor handoff/copy measurement.

Iter 4 sweep:

| split | avg ms | p95 ms | result |
|---|---:|---:|---|
| `-n 2` | **757.0** | **759.2** | best |
| `-n 3` | 766.6 | 775.3 | slower |
| `-n 4` | 772.9 | 779.4 | slower, multi-boundary chain |

CPU TVM pass follow-up: `opt_level=3 --disabled-pass AlterOpLayout` measured
796.8 ms / p95 813.0 ms, slower than the existing opt_level=3 baseline, so keep
TVM's default layout rewrite enabled.

Single-Coral layout rewrite follow-up:

An exact `Transpose -> Concat -> inverse Transpose` rewrite removes 11 transpose ops
while preserving byte-identical CPU TFLite output. This improves the single-Coral
stage2 path from 1472.8 ms to **1365.4 ms**. It should be used for single-TPU
deployment, but not for the current fastest two-TPU `-n 2` pipeline, where it measured
slower at 768.5 ms.
