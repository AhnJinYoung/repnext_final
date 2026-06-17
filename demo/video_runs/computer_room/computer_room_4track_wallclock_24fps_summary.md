| Run | Backend | Size | Frames | Avg infer | Infer FPS | Wall FPS | Speedup vs first |
|---|---|---:|---:|---:|---:|---:|---:|
| Native PyTorch 512 | pytorch | 512 | 24 | 2278.2 ms | 0.44 | 0.39 | 1.00x |
| Intel CPU LiteRT 192 | tflite | 192 | 24 | 152.1 ms | 6.58 | 3.82 | 14.98x |
| RPi5 CPU LiteRT 256 | tflite | 256 | 24 | 347.1 ms | 2.88 | 2.14 | 6.56x |
| RPi5 + Coral TPU INT8 192 | tflite | 192 | 24 | 81.7 ms | 12.23 | 4.95 | 27.87x |
