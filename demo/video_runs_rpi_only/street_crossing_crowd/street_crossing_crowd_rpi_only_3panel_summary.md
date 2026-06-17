| Run | Backend | Size | Frames | Avg infer | Infer FPS | Wall FPS | Speedup vs first |
|---|---|---:|---:|---:|---:|---:|---:|
| Native PyTorch 512 | pytorch | 512 | 8 | 1819.2 ms | 0.55 | 0.49 | 1.00x |
| Intel CPU LiteRT 192 | tflite | 192 | 120 | 97.4 ms | 10.26 | 7.17 | 18.67x |
| RPi5 CPU LiteRT 256 | tflite | 256 | 120 | 345.3 ms | 2.90 | 2.49 | 5.27x |
