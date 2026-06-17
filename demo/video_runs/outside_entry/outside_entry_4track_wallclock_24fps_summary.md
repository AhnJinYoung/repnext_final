| Run | Backend | Size | Frames | Avg infer | Infer FPS | Wall FPS | Speedup vs first |
|---|---|---:|---:|---:|---:|---:|---:|
| Native PyTorch 512 | pytorch | 512 | 24 | 2162.6 ms | 0.46 | 0.40 | 1.00x |
| Intel CPU LiteRT 192 | tflite | 192 | 24 | 96.6 ms | 10.35 | 7.33 | 22.39x |
| RPi5 CPU LiteRT 256 | tflite | 256 | 24 | 343.7 ms | 2.91 | 2.50 | 6.29x |
| RPi5 + Coral TPU INT8 192 | tflite | 192 | 24 | 80.8 ms | 12.38 | 6.83 | 26.77x |
