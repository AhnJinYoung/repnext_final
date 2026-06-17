# RPi5-only eye-level video demos

These demos use eye-level or street-level moving-object stock videos rather than CCTV footage. Each output is a 16-second, 24 FPS wall-clock comparison with three panels only: Native PyTorch 512, Intel CPU LiteRT 192, and the final demo path, RPi5 CPU LiteRT 256. Native PyTorch processes only 8 frames to demonstrate slowness; optimized paths process 120 frames from the same clip.

| Demo | Source FPS | Output video | Native frames / avg | Intel frames / avg | RPi5 frames / avg | RPi5 speedup vs native |
|---|---:|---|---:|---:|---:|---:|
| city_follow_walk | 23.98 | `demo/video_runs_rpi_only/city_follow_walk/city_follow_walk_rpi_only_3panel_24fps.mp4` | 8 / 2333.7 ms | 120 / 98.1 ms | 120 / 345.3 ms | 6.76x |
| street_crossing_crowd | 30.00 | `demo/video_runs_rpi_only/street_crossing_crowd/street_crossing_crowd_rpi_only_3panel_24fps.mp4` | 8 / 1819.2 ms | 120 / 97.4 ms | 120 / 345.3 ms | 5.27x |
| suit_crossing_street | 23.98 | `demo/video_runs_rpi_only/suit_crossing_street/suit_crossing_street_rpi_only_3panel_24fps.mp4` | 8 / 1774.0 ms | 120 / 93.4 ms | 120 / 348.7 ms | 5.09x |

Recommended advisor demo:

- `city_follow_walk`: best eye-level walking scene with foreground person and urban background.
- `suit_crossing_street`: cleanest single moving person / street scene.
- `street_crossing_crowd`: more crowded moving-object stress case.
