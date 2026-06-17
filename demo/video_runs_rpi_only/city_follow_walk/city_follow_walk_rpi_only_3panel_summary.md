| Run | Backend | Size | Frames | Avg infer | Infer FPS | Wall FPS | Speedup vs first |
|---|---|---:|---:|---:|---:|---:|---:|
| Native PyTorch 512 | pytorch | 512 | 8 | 2333.7 ms | 0.43 | 0.40 | 1.00x |
| Intel CPU LiteRT 192 | tflite | 192 | 120 | 98.1 ms | 10.19 | 7.11 | 23.79x |
| RPi5 CPU LiteRT 256 | tflite | 256 | 120 | 345.3 ms | 2.90 | 2.50 | 6.76x |
