# Native vs Raspberry Pi 5 CPU Realtime Video Comparisons

These videos reuse the already generated segmentation frames. The source video
timeline is fixed to about 10 seconds and encoded at 24 FPS. If a model cannot
finish inference before the next source frames arrive, the video holds the most
recent completed segmentation result, so slower models appear choppy without
stretching the clip duration.

| Clip | Source | Native PyTorch 512 | RPi5 CPU LiteRT 256 | Output |
|---|---|---:|---:|---|
| university student | `students_university_10s/source_students_university.mp4` | 1849.0 ms/frame | 381.1 ms/frame | `students_university_10s/students_university_10s_native_vs_rpi5_realtime_sync_24fps.mp4` |
| city follow walk | `city_follow_walk_10s/source.mp4` | 2808.5 ms/frame | 360.0 ms/frame | `city_follow_walk_10s/city_follow_walk_10s_native_vs_rpi5_realtime_sync_24fps.mp4` |
| city street | `busy_city_street_10s/source_busy_city_street.mp4` | 1812.4 ms/frame | 391.6 ms/frame | `busy_city_street_10s/busy_city_street_10s_native_vs_rpi5_realtime_sync_24fps.mp4` |

Generation command pattern:

```bash
.venv-video/bin/python demo/make_native_vs_rpi5_realtime_video.py \
  --root demo/video_runs_rpi_only/<run_dir> \
  --output demo/video_runs_rpi_only/<run_dir>/<name>_native_vs_rpi5_realtime_sync_24fps.mp4 \
  --fps 24 \
  --panel-width 640 \
  --panel-height 360
```
