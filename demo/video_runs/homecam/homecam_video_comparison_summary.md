# Homecam Video Segmentation Demo Comparison

Source: C-MOR sample surveillance recording (`source_homecam.mp4`).

| Panel | Backend | Size | Frames | Avg infer | Infer FPS | Speedup |
|---|---|---:|---:|---:|---:|---:|
| Native 512 PyTorch | PyTorch | 512 | 5 | 3029.4 ms | 0.33 | 1.00x |
| LiteRT 192 dynamic-range | TFLite/LiteRT | 192 | 5 | 288.4 ms | 3.47 | 10.50x |

Output video: `demo/video_runs/homecam/homecam_comparison_original_native_litert192.mp4`
