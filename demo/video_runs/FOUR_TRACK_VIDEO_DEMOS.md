# Four-track video segmentation demos

Sources are public C-MOR sample surveillance/timelapse recordings. Each demo uses the same four-panel wall-clock playback: Native PyTorch 512, Intel CPU LiteRT 192, RPi5 CPU LiteRT 256, and RPi5 + Coral TPU INT8 192. The output video is encoded at 24 FPS; each panel advances according to its measured per-frame inference latency.

| Demo | Source FPS | Output video | Native | Intel CPU | RPi5 CPU | RPi5 + TPU |
|---|---:|---|---:|---:|---:|---:|
| homecam_24fps | 6.00 | `demo/video_runs/homecam_24fps/homecam_4track_wallclock_24fps.mp4` | 2122.6 ms | 109.3 ms | 346.6 ms | 81.0 ms |
| computer_room | 23.98 | `demo/video_runs/computer_room/computer_room_4track_wallclock_24fps.mp4` | 2278.2 ms | 152.1 ms | 347.1 ms | 81.7 ms |
| outside_entry | 7.00 | `demo/video_runs/outside_entry/outside_entry_4track_wallclock_24fps.mp4` | 2162.6 ms | 96.6 ms | 343.7 ms | 80.8 ms |
| castle_timelapse | 24.00 | `demo/video_runs/castle_timelapse/castle_timelapse_4track_wallclock_24fps.mp4` | 2666.7 ms | 179.1 ms | 344.5 ms | 81.2 ms |

Recommended live-demo candidates:

- `castle_timelapse`: true 24 FPS source, largest Native-vs-TPU contrast in this batch.
- `computer_room`: true 23.98 FPS source, clear indoor scene with people/door structure.
- `outside_entry`: outdoor surveillance scene; original FPS is lower, but latency contrast is strong.
- `homecam_24fps`: previous entry-area sample retained for continuity with earlier results.
