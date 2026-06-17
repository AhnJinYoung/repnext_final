| Run | Backend | Size | Frames | Avg infer | Infer FPS | Wall FPS | Speedup vs first |
|---|---|---:|---:|---:|---:|---:|---:|
| Native PyTorch 512 | pytorch | 512 | 24 | 2666.7 ms | 0.37 | 0.33 | 1.00x |
| Intel CPU LiteRT 192 | tflite | 192 | 24 | 179.1 ms | 5.58 | 4.46 | 14.89x |
| RPi5 CPU LiteRT 256 | tflite | 256 | 24 | 344.5 ms | 2.90 | 2.66 | 7.74x |
| RPi5 + Coral TPU INT8 192 | tflite | 192 | 24 | 81.2 ms | 12.31 | 8.50 | 32.83x |
