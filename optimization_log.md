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

## iter 2 — 2026-04-27

### CPU 실험: TVM Relay full RepNeXt-M5 (no tuning)
- 입력: `repnext_m5_ade20k_relu_sparse_equiv_simplified_kernelshape.onnx` (103 MB,
  Pi5 로 scp).
- target/opt 동일: `cortex-a76 +v8.2a +fullfp16 +dotprod`, opt_level=3.
- **결과: 빌드는 532 s 에 성공, 그러나 첫 inference 가 22 분이 지나도 끝나지 않아 budget
  초과(30분) 시점에 kill.** 결과: `iter2_cpu_tvm_full_FAILED.json`.
- 원인 가설: AutoTVM/MetaSchedule 으로 full-model conv 들의 schedule 이 등록되어 있지
  않아 default fallback schedule 이 catastrophic 하게 느림. stage2 partition (783 ms)
  은 잘 돌아갔던 것과 대조 — 이전 단계(stem/stage0/stage1) 의 큰 spatial conv 가 병목으로 추정.
- iter 3 가설: opt_level=2 로 fallback 줄이고, `relay.transform.SimplifyInference`,
  `FoldConstant`, `FuseOps(fuse_opt_level=2)` 적용 후 재측정.
  안 되면 MetaSchedule short tune 32 trials × top-K task.

### TPU 실험: dual-Coral 2× data-parallel (그래프 레벨)
- best tflite (`repnext_stage2_int8_dwpatched_edgetpu.tflite`) 를 device 0/1 에 동시 로드,
  threading 으로 병렬 invoke. warmup 10, runs 50.
- 결과: `iter2_tpu_2x_dataparallel.json`
  - device0 only: 1466 ms (p95 1530)
  - device1 only: 1490 ms (p95 1568)
  - 2× parallel wallclock/iter: 1740 ms (p95 1827) → **throughput 1.15 ips**
- 친구의 4/24 측정 (1614 ms / 1.24 ips) 보다 약간 저조 — USB hub 전력/온도 차이로 추정.
  TPU best (1× latency 1465 ms) 는 미갱신, 단 2× throughput 데이터 신규 추가.

### 다음 가설 (iter 3)
- **CPU**: TVM 빌드 옵션을 opt_level=2 로 낮추고 `SimplifyInference` + `FoldConstant`
  적용한 full-model 재측정. 또한 stage 단위 pipeline 모듈(stage0/1/2 각각 컴파일 후 PyTorch
  로 sequential dispatch) 로 빠른 반쪽 답 확보.
- **TPU**: edgetpu_compiler 재설치 후, `--num_segments 2/4` 로 multi-segment
  분할이 단일 segment 대비 latency 어떻게 바뀌는지 측정. 컴파일러 재설치 못 하면 입력 해상도 sweep
  (256/384/512) 측정으로 latency vs accuracy trade-off 데이터 수집.

---

## Pruned (deprecated 산출물 삭제 기록)

### iter 1 — 2026-04-27
- 삭제 대상 표시 (실 삭제는 iter 2 에서 수행):
  - `~/repnext-pipeline/repnext_m5_relu_tpu_stage2_downsample_512_simplified_kernelshape_full_integer_quant_edgetpu.tflite`
    — invoke 실패.
  - `~/repnext-pipeline/repnext_m5_relu_tpu_stage2_downsample_512_simplified_kernelshape_full_integer_quant_edgetpu.log`

### iter 2 — 2026-04-27
- **RPi5 에서 실 삭제 수행 (`rm`):**
  - `/home/rpi5/repnext-pipeline/repnext_m5_relu_tpu_stage2_downsample_512_simplified_kernelshape_full_integer_quant_edgetpu.tflite` (1.9 MB)
  - `/home/rpi5/repnext-pipeline/repnext_m5_relu_tpu_stage2_downsample_512_simplified_kernelshape_full_integer_quant_edgetpu.log`
- 유지: `~/repnext-pipeline/runs/20260427_iter1_*` (분석 자료), `runs/20260427_iter2_*`.
- 로컬 `RepNeXt-tpu/*` legacy logs — best path 와 무관하지만 분석 자료라 보류.

## iter 3 - 2026-04-27

### TPU: EdgeTPU compiler segment split + 2-Coral pipeline
- RPi5 runtime sees both Coral USB devices:
  `/sys/bus/usb/devices/5-1`, `/sys/bus/usb/devices/3-1`.
- RPi5 `edgetpu_compiler` wrapper is present, but its `/tmp/edgetpu_compiler_pkg/...`
  payload is gone again. Segment compile was run in WSL with
  `edgetpu_compiler v16.0.384591198`.
- Input TFLite: `repnext_m5_relu_sparse_stage2_downsample_512_int8_dwpatched.tflite`.
- `-n 2` result: segment 0 = 25 TPU ops / 395 CPU ops, segment 1 = 2 TPU ops / 142 CPU ops.
- `-n 4` result: segment TPU ops = 25, 3, 2, 18. Compiler warns that `num_segments = 1`
  is recommended.
- Fixed PyCoral pipeline benchmark device binding from old `"model,:0"` style to
  `make_interpreter(path, device="usb:N")`.
- 2-segment sequential pipeline on TPU0 -> TPU1, warmup 10 / runs 50:
  **avg 758.8 ms, p50 758.1 ms, p95 764.1 ms**.
  Result: `benchmark/results/iter3_tpu_pipeline_split_n2_50runs.json`.
- Interpretation: per-segment op mapping is poor, but the measured stage2 partition
  latency is much faster than the previous single EdgeTPU 1464.8 ms. Need to validate
  full BYOC handoff / tensor-copy overhead before calling this an E2E win.

### CPU: TVM Relay stage2 opt_level=2
- Same stage2 downsample ONNX, TVM 0.18, `opt_level=2`, warmup 5 / runs 20.
- build 84.0 s, **avg 850.5 ms, p50 845.7 ms, p95 911.6 ms**.
  Result: `benchmark/results/iter3_cpu_tvm_stage2_opt2.json`.
- Existing `opt_level=3` stage2 baseline (783.9 ms) remains better. Next CPU work
  should focus on partition-level MetaSchedule/AutoTVM rather than lowering global
  Relay opt level.

## iter 4 - 2026-04-27

### TPU graph split sweep: n=2/3/4 chain benchmark
- Added `benchmark/tpu_pipeline_chain.py` to run arbitrary sequential EdgeTPU segment
  chains and report per-stage invoke time plus tensor set/get/cast overhead.
- `n=4` compiler output has multi-input/multi-output boundaries, so the chain runner
  now passes tensors by TFLite tensor name instead of assuming one output -> one input.
- Same stage2 partition, warmup 10 / runs 50:
  - `n=2`: **avg 757.0 ms, p95 759.2 ms**.
    Stage invokes: 620.9 ms + 134.9 ms. Tensor transfer/set/get: 0.78 ms + 0.37 ms.
  - `n=3`: avg 766.6 ms, p95 775.3 ms.
    Stage invokes: 566.0 ms + 116.6 ms + 82.4 ms.
  - `n=4`: avg 772.9 ms, p95 779.4 ms.
    Stage invokes: 539.1 ms + 92.1 ms + 110.0 ms + 27.7 ms.
- Result: forced split helps versus the single EdgeTPU path, but more than two segments
  is slower. Current TPU graph best remains `-n 2` pipeline.

### CPU TVM graph pass attempt: disable AlterOpLayout
- Added `--disabled-pass` support to `benchmark/tvm_cpu_benchmark.py`.
- Tested `opt_level=3 --disabled-pass AlterOpLayout` on the stage2 partition.
- Result: build 114.8 s, **avg 796.8 ms, p95 813.0 ms**.
- This is slower than the existing opt_level=3 baseline (783.9 ms), so TVM's default
  `AlterOpLayout` should stay enabled for this graph. Simple Relay pass on/off did
  not improve CPU speed.

## iter 5 - 2026-04-27

### Math-exact transpose layout rewrite
- Added `conversion/analyze_transpose_tflite.py` to inspect TFLite transpose patterns.
- Original stage2 TFLite:
  - 564 ops total, 116 `TRANSPOSE`.
  - Permutations: `(0,3,1,2)` x61 and `(0,2,3,1)` x55.
  - No identity transpose and no direct inverse transpose cancellation pairs.
- Added `conversion/rewrite_transpose_concat_tflite.py`.
  - Rewrites exact `Transpose(p) -> Concat(axis=a) -> Transpose(inv_p)` islands into
    `Concat(axis=p[a])`.
  - Also checks `Transpose -> Split -> inverse Transpose` islands, but no safe split
    islands matched on this graph.
- Rewrite result:
  - concat islands rewritten: 3.
  - transpose ops removed: 11.
  - op count: 564 -> 553.
  - transpose count: 116 -> 105.
- Mathematical integrity check:
  - CPU TFLite original vs rewritten on the same random int8 input: byte-identical
    output, `max_abs_diff=0`, `num_diff=0`.

### EdgeTPU compile and latency
- `edgetpu_compiler -s -a` on concat-folded model:
  - total ops 553.
  - EdgeTPU ops 448, CPU ops 105.
  - `TRANSPOSE` CPU ops reduced from 116 to 105; mapped TPU op count unchanged.
- Single Coral comparison, warmup 10 / runs 50:
  - original: avg 1472.8 ms, p95 1542.4 ms.
  - concat-folded: **avg 1365.4 ms, p95 1440.2 ms**.
  - This is about a 7.3% latency improvement for the single-segment EdgeTPU path.
- Forced `-n 2` pipeline on concat-folded model:
  - avg 768.5 ms, p95 778.1 ms.
  - Slower than original `-n 2` chain avg 757.0 ms, because the rewrite changes the
    forced split boundary into a multi-output/multi-input handoff.
- Current recommendation:
  - use concat-fold rewrite for single EdgeTPU stage2 deployment.
  - keep original `-n 2` split for the fastest two-Coral stage2 pipeline.
