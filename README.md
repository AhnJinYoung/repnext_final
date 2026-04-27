# RepNeXt Optimization for RPi5 + Coral Edge TPU

RepNeXt-M5 (ADE20K semantic segmentation backbone) 를 Raspberry Pi 5 + Google
Coral USB Edge TPU 에서 동작시키기 위한 변환 / 컴파일 / 벤치마크 자료 모음.

대용량 바이너리(가중치, ONNX, TFLite, calibration npy)는 git 에 포함되지
않으며, 아래 경로에 별도로 보관/생성해야 합니다.

---

## 디렉토리 구성

```
repnext-optimization/
├── conversion/        # PyTorch -> ONNX -> int8 TFLite 변환 스크립트
├── benchmark/         # RPi5 / Coral 벤치마크 스크립트
│   └── results/       # 측정 결과 JSON
├── docs/              # 타당성 분석, partition 분석 리포트
└── logs/              # edgetpu_compiler / tflite 변환 로그
```

---

## 모델 / 데이터 위치 (git 미포함)

### 로컬 (Windows, 작업 PC)
경로: `C:\Users\jjw74\Desktop\SNU\SNU 3-2\Creative Integrated Design\RepNeXt-tpu\`

| 파일 | 설명 |
|------|------|
| `repnext_m5_ade20k.pth` | 원본 PyTorch 가중치 (≈100 MB) |
| `repnext_m5_ade20k_relu_sparse_equiv.onnx(+.data)` | 전체 모델 ONNX (ReLU/Sparse equivalent) |
| `repnext_m5_ade20k_relu_sparse_equiv_simplified_kernelshape.onnx` | onnxsim + kernelshape 패치 적용본 |
| `repnext_m5_relu_tpu_stage2_downsample_512_simplified_kernelshape.onnx` | **현재 사용 중인 partition (stage2 downsample, 512×512)** |
| `repnext_m5_relu_tpu_stage2_downsample_512_int8_dwpatched_edgetpu.tflite` | **현재 사용 중인 EdgeTPU 컴파일 결과 (5.77 MB)** |
| `calib_512_5_nhwc_float32.npy` | 캘리브레이션 샘플 (5장, 512×512, NHWC float32) |
| `calibration_image_sample_data_20x128x128x3_float32.npy` | 초기 실험용 캘리브 샘플 |
| `saved_model_repnext_sparse_stage2_downsample_512_int8/` | TFLite 변환용 SavedModel |

### Raspberry Pi 5 (`rpi5@<RPI5_IP>:/home/rpi5/`)
경로: `~/repnext-pipeline/`

| 파일 | sha256 (앞 16자리) | 용도 |
|------|---------------------|------|
| `repnext_m5_ade20k.pth` | `46d70cfea436b8b0` | 원본 가중치 (필요시 재변환) |
| `repnext_m5_relu_tpu_stage2_downsample_512_simplified_kernelshape.onnx` | `b02aeac7f2de4892` | onnxsim + kernelshape 패치본 |
| `repnext_stage2_int8_dwpatched_edgetpu.tflite` | `ea017c4a67cabd7c` | EdgeTPU 컴파일된 stage2 (= 로컬 `_relu_tpu_stage2_..._edgetpu.tflite` 와 동일) |
| `calib_512_50.npy`, `calib_512_50_nhwc_float32.npy` | — | 캘리브레이션 샘플 (50장) |
| `saved_model_repnext_stage2_int8_tf/` | — | TFLite full integer quant 결과 |
| `benchmark.py` | — | Coral 벤치마크 (single TPU / dual TPU data-parallel) |
| `results_*.json` | — | 벤치마크 결과 |

추가로 `~/creative_design/RepNeXt-tpu/` 에 PyTorch CPU baseline (`results_repnext.json`,
`results_repnext_cpu_probe.json`) 가 들어있음.

---

## 파이프라인

전체 흐름: **PyTorch (.pth) → ONNX → simplified ONNX → SavedModel → INT8 TFLite → EdgeTPU TFLite → 벤치마크**

```
.pth ─(1)─► .onnx ─(2)─► simplified.onnx ─(3)─► SavedModel ─(4)─► int8.tflite ─(5)─► dwpatched.tflite ─(6)─► edgetpu.tflite ─(7)─► benchmark
```

### (1) PyTorch → ONNX
```bash
cd RepNeXt-tpu
python export_onnx.py            # 전체 모델
# 또는
python export_partition_onnx.py  # stage 단위로 잘라 partition export
```
산출물: `repnext_m5_ade20k_relu_sparse_equiv.onnx`,
`repnext_m5_relu_tpu_stage2_downsample_512.onnx` 등

### (2) ONNX simplification + kernelshape 패치
`onnxsim` 적용 후, EdgeTPU 컴파일러가 요구하는 kernel shape 정렬을 수행.
산출물: `*_simplified_kernelshape.onnx` (스크립트는 `export_partition_onnx.py`
내 후처리 단계에 포함).

### (3) ONNX → TF SavedModel
`onnx2tf` (또는 `onnx_tf`) 로 변환. 출력 디렉토리: `saved_model_repnext_*_int8/`.
입력 layout 은 NHWC (`conversion_config.json` 참조).

### (4) SavedModel → INT8 TFLite (full integer quantization)
```bash
python conversion/convert_tflite.py
```
캘리브레이션은 `calib_512_50_nhwc_float32.npy` (또는 `calib_512_5_*.npy`) 를 사용.
설정 파일: [`conversion/conversion_config.json`](conversion/conversion_config.json),
[`conversion/conversion_config_rpi5.json`](conversion/conversion_config_rpi5.json).

### (5) Depthwise conv 패치
EdgeTPU 컴파일러가 일부 depthwise conv 패턴에서 fail 하는 문제를 회피하기
위해 텐서를 재구성/replace.
```bash
python conversion/patch_depthwise_tflite.py \
  --in  repnext_m5_..._int8.tflite \
  --out repnext_..._int8_dwpatched.tflite
```

### (6) EdgeTPU 컴파일
```bash
edgetpu_compiler -s repnext_..._int8_dwpatched.tflite
```
로그 예시: [`logs/repnext_m5_relu_tpu_stage2_downsample_512_int8_dwpatched_edgetpu.log`](logs/),
[`logs/rpi5_repnext_stage2_int8_dwpatched_edgetpu.log`](logs/).

> 현재 전체 모델은 EdgeTPU 단일 segment 로 들어가지 않아 **stage2 downsample
> partition** 만 컴파일하여 평가하고 있음. 자세한 분석:
> [`docs/partition_report.md`](docs/partition_report.md),
> [`docs/feasibility_report.md`](docs/feasibility_report.md).

### (7) 벤치마크 (RPi5 + Coral)
```bash
# RPi5 측
ssh rpi5@<RPI5_IP>
cd ~/repnext-pipeline
source ~/coral-env/bin/activate
python benchmark.py            # 50 runs, warmup 10, 1×/2× TPU 모드
```
스크립트 사본: [`benchmark/benchmark_rpi5.py`](benchmark/benchmark_rpi5.py).
PyTorch CPU baseline (4-thread): [`benchmark/benchmark_local.py`](benchmark/benchmark_local.py).

---

## 현재 측정 결과 (RepNeXt-M5 ADE20K, 입력 512×512, runs=50, warmup=10)

| Mode | avg (ms) | p50 | p95 | throughput |
|------|---------:|----:|----:|-----------:|
| PyTorch CPU 4-thread (RPi5) | 4222.6 | 4270.0 | 4360.1 | 0.24 ips |
| Coral USB TPU ×1 (device 0) | 1425.2 | 1423.9 | 1438.0 | 0.70 ips |
| Coral USB TPU ×1 (device 1) | 1428.3 | 1424.6 | 1453.1 | 0.70 ips |
| Coral USB TPU ×2 data-parallel | 1614.0 | 1608.6 | 1667.7 | **1.24 ips** |
| **PyTorch torch.compile (full, iter 1)** | **2608.0** | **2625.6** | **2706.5** | 0.38 ips |
| TVM CPU (stage2 partition only, iter 1) | 783.9 | 785.0 | 803.4 | 1.28 ips |
| Coral USB TPU ×1 (re-measured iter 1) | 1464.8 | 1454.1 | 1523.8 | 0.68 ips |
| Coral USB TPU ×2 data-parallel (iter 2) | 1739.6 | 1735.9 | 1826.6 | 1.15 ips |
| TVM CPU (full model, iter 2) | _build OK 532s, inference timeout_ | — | — | — |

원본 JSON: [`benchmark/results/`](benchmark/results/).

> 현 시점 측정 대상은 stage2 downsample partition (5.77 MB EdgeTPU TFLite).
> 전체 모델 end-to-end 측정은 추후 작업.

---

## 환경

- **RPi5**: BCM2712, Cortex-A76 ×4, 8 GB, Raspberry Pi OS Bookworm
- **Edge TPU**: Coral USB ×2 (`/sys/bus/usb/devices/3-1`, `/sys/bus/usb/devices/5-1`)
- **Python venv (RPi5)**: `~/coral-env` (TFLite runtime + pycoral),
  변환용은 `~/repnext-convert-env`, `~/repnext-convert-v1-env`
- **Compiler**: `edgetpu_compiler` (gasket-driver 기반)
- **로컬 변환 환경**: `onnx==1.19`, `onnx-graphsurgeon==0.5.1`,
  `tensorflow==2.20.0`, `ml_dtypes`

---

## 참고 문서
- [`docs/feasibility_report.md`](docs/feasibility_report.md) — RepNeXt 의 EdgeTPU 타당성 분석
- [`docs/partition_report.md`](docs/partition_report.md) — 모델 partition 전략과 op-level 진단
- [`docs/repnext_ops.json`](docs/repnext_ops.json) — op 통계
- 상위 프로젝트 핸드오프: `MoviNet-optimization-with-tvm/rpi5_coral_tvm_handoff/HANDOFF_RPI5_CORAL_REPNEXT_DEPTHANYTHING.md`

---

## Iter 3 Snapshot

| Mode | avg (ms) | p50 | p95 | throughput |
|------|---------:|----:|----:|-----------:|
| Coral USB TPU x2 pipeline split, 2 segments | **758.8** | **758.1** | **764.1** | 1.32 ips |
| TVM CPU stage2 partition, opt_level=2 | 850.5 | 845.7 | 911.6 | 1.18 ips |
| Coral USB TPU x2 pipeline chain, n=2 (iter 4) | **757.0** | **757.2** | **759.2** | 1.32 ips |
| Coral USB TPU x2 pipeline chain, n=3 (iter 4) | 766.6 | 765.9 | 775.3 | 1.30 ips |
| Coral USB TPU x2 pipeline chain, n=4 (iter 4) | 772.9 | 771.4 | 779.4 | 1.29 ips |
| TVM CPU stage2 opt3 without AlterOpLayout (iter 4) | 796.8 | 796.4 | 813.0 | 1.26 ips |
| Coral USB TPU x1 concat-layout rewrite (iter 5) | **1365.4** | **1348.8** | **1440.2** | 0.73 ips |
| Coral USB TPU x2 concat-layout rewrite + n=2 chain (iter 5) | 768.5 | 766.1 | 778.1 | 1.30 ips |

Notes:
- `edgetpu_compiler -n 2` and `-n 4` both warn that one segment is recommended.
- Even with poor per-segment TPU op mapping, the measured 2-Coral sequential pipeline
  for the stage2 partition is faster than the previous single-segment EdgeTPU latency.
- TVM opt_level=2 is slower than the existing opt_level=3 stage2 baseline, so CPU
  work should move toward partition-level tuning rather than lowering global opt level.
- iter 4 confirms `n=2` is the best forced segment split. `n=3` and `n=4` add boundary
  complexity and are slower. Disabling TVM `AlterOpLayout` is also slower than the
  existing opt_level=3 baseline.
- iter 5 folds exact `Transpose -> Concat -> inverse Transpose` islands. This removes
  11 transpose ops (116 -> 105) with byte-identical CPU TFLite output. It improves
  single EdgeTPU latency by about 7.3%, but makes the forced `n=2` chain slower than
  the original `n=2` chain because the split boundary changes shape/count.
