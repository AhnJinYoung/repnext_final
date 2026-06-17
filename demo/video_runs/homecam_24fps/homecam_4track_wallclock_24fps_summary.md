| Run | Backend | Size | Frames | Avg infer | Infer FPS | Wall FPS | Speedup vs first |
|---|---|---:|---:|---:|---:|---:|---:|
| Native PyTorch 512 | pytorch | 512 | 24 | 2122.6 ms | 0.47 | 0.42 | 1.00x |
| Intel CPU LiteRT 192 | tflite | 192 | 24 | 109.3 ms | 9.15 | 4.97 | 19.41x |
| RPi5 CPU LiteRT 256 | tflite | 256 | 24 | 346.6 ms | 2.89 | 2.14 | 6.12x |
| RPi5 + Coral TPU INT8 192 | tflite | 192 | 24 | 81.0 ms | 12.35 | 4.87 | 26.21x |
