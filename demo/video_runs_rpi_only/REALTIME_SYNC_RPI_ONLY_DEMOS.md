# Real-time synced RPi-only demos

These videos keep the original source-time duration. The output is 24 FPS, but each model panel only updates when that model would have completed an inference in a live frame-dropping pipeline. Slow models therefore stutter instead of stretching the video duration.

| Demo | Duration | Output video | Native sparse | Intel CPU | RPi5 CPU |
|---|---:|---|---:|---:|---:|
| city_follow_walk | 5.00s | `demo/video_runs_rpi_only/city_follow_walk/city_follow_walk_rpi_only_3panel_realtime_sync_24fps.mp4` | 3 / 1836.2 ms | 120 / 98.1 ms | 120 / 345.3 ms |
| street_crossing_crowd | 4.00s | `demo/video_runs_rpi_only/street_crossing_crowd/street_crossing_crowd_rpi_only_3panel_realtime_sync_24fps.mp4` | 3 / 2210.3 ms | 120 / 97.4 ms | 120 / 345.3 ms |
