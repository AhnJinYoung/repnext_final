| Run | Backend | Size | Frames | Avg infer | Infer FPS | Wall FPS | Speedup vs first |
|---|---|---:|---:|---:|---:|---:|---:|
| Native PyTorch 512 | pytorch | 512 | 8 | 1774.0 ms | 0.56 | 0.48 | 1.00x |
| Intel CPU LiteRT 192 | tflite | 192 | 120 | 93.4 ms | 10.71 | 7.46 | 19.00x |
| RPi5 CPU LiteRT 256 | tflite | 256 | 120 | 348.7 ms | 2.87 | 2.43 | 5.09x |
