# Optimization Log

매 iteration 한 줄로 시도/측정/다음 가설을 기록.
형식: `### iter N — YYYY-MM-DD HH:MM` 후 CPU/TPU 두 트랙.

---

## iter 1 — 2026-04-27

### 환경 점검
- TVM: `~/tvm-src/build/libtvm.so` 빌드됨 (0.18.0). `PYTHONPATH=~/tvm-src/python python3` 로 import.
- PyTorch: 시스템 python3.12 → `Dynamo not supported on 3.12+`. `~/coral-env` (py3.9) 의
  `torch 2.8.0+cpu` 사용.
- **edgetpu_compiler 손상:** `/usr/local/bin/edgetpu_compiler` 가 wrapper script 인데
  실제 바이너리(`/tmp/edgetpu_compiler_pkg/...`) 가 재부팅으로 삭제됨. **재설치 필요**
  (다음 iteration 전에 사용자에게 요청).
- Coral USB ×2 (`1a6e:089a`) 정상 인식.

### CPU 실험
- (a) **TVM Relay baseline (stage2 partition)** — `llvm cortex-a76 +v8.2a +fullfp16 +dotprod`,
  opt_level=3, AutoTVM/MetaSchedule 튜닝 없음.
  결과: **avg 783.9 ms, p95 803.4 ms** (build 138 s). 결과 JSON
  `iter1_cpu_tvm_baseline_stage2.json`.
- (b) **PyTorch torch.compile baseline (full model)** — `torch.compile(model)` default
  inductor backend, threads=4.
  결과: **avg 2608.0 ms, p95 2706.5 ms** (compile 92 s, setup 4.4 s). 결과 JSON
  `iter1_pytorch_compile_full.json`.
  → PyTorch eager 4-thread baseline (4222 ms) 대비 **1.62× 빠름**. CPU best 갱신.

### TPU 실험
- 컴파일러 부재로 재컴파일 불가. 기존 두 EdgeTPU tflite 비교 측정.
- (a) `repnext_stage2_int8_dwpatched_edgetpu.tflite` (5.77 MB, current best 후보):
  **avg 1464.8 ms, p95 1523.8 ms** (재측정 50 runs). sha8 `ea017c4a67cabd7c`.
- (b) `repnext_..._full_integer_quant_edgetpu.tflite` (1.9 MB, no-dwpatch):
  **실패** — `RuntimeError: Input tensor 101 lacks data`. 변환 시 일부 input 이
  baked-in 되지 않은 partial tflite. **deprecated 표시**.

### 다음 가설 (iter 2 후보)
- **CPU**: TVM 으로 full RepNeXt-M5 ONNX (`*_relu_sparse_equiv_simplified_kernelshape.onnx`,
  103 MB) 컴파일 → torch.compile (2608 ms) 와 fair compare. 그 다음 MetaSchedule
  short tuning (32 trials × 3 task) 으로 첫 튜닝 단계.
- **TPU**: edgetpu_compiler 재설치 후, partition 경계를 stem→stage1, stage1→stage2 로
  옮긴 두 ONNX export 후 각각 컴파일·측정. 가장 큰 단일 EdgeTPU 매핑이 어디서
  가능한지 확인.

---

## Pruned (deprecated 산출물 삭제 기록)

### iter 1 — 2026-04-27
- **삭제 대상으로 표시 (실 삭제는 iter 2 에서):**
  - `~/repnext-pipeline/repnext_m5_relu_tpu_stage2_downsample_512_simplified_kernelshape_full_integer_quant_edgetpu.tflite`
    (1.9 MB) — invoke 시 input 누락 에러, 사용 불가.
  - `~/repnext-pipeline/repnext_m5_relu_tpu_stage2_downsample_512_simplified_kernelshape_full_integer_quant_edgetpu.log`
- **삭제 보류:**
  - `~/repnext-pipeline/repnext_m5_relu_tpu_stage2_downsample_512_simplified_kernelshape.onnx`
    (6.3 MB, current best 의 source) — 유지.
  - 로컬 `RepNeXt-tpu/*` legacy logs — best path 와 무관하지만 분석 자료라 유지.
    iter 3 이후 best path 가 안정되면 일괄 정리.
