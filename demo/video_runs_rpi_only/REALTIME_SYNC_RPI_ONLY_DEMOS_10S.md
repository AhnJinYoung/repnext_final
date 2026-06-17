# 10-second real-time synced RPi-only demos

These videos keep source time fixed at about 10 seconds. The output is 24 FPS, but each panel updates only when that model would finish inference in a live latest-frame/drop-frame pipeline.

| Demo | Duration | Output video | Native sparse | Intel CPU | RPi5 CPU |
|---|---:|---|---:|---:|---:|
| city_follow_walk_10s | 10.01s | `demo/video_runs_rpi_only/city_follow_walk_10s/city_follow_walk_10s_rpi_only_3panel_realtime_sync_24fps.mp4` | 5 / 2808.5 ms | 240 / 100.1 ms | 240 / 360.0 ms |
| street_crossing_crowd_10s | 10.00s | `demo/video_runs_rpi_only/street_crossing_crowd_10s/street_crossing_crowd_10s_rpi_only_3panel_realtime_sync_24fps.mp4` | 5 / 2076.3 ms | 300 / 97.3 ms | 300 / 378.8 ms |
