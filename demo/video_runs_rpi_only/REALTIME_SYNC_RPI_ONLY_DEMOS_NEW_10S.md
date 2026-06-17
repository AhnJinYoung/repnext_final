# Additional 10-second real-time synced RPi-only demos

New eye-level/street-level moving-object videos. Source files are saved in each run directory as `source_<title>.mp4`. The output is 24 FPS and source-time-synced; frames are dropped if inference cannot keep up.

| Demo | Duration | Source file | Output video | Native sparse | Intel CPU | RPi5 CPU |
|---|---:|---|---|---:|---:|---:|
| anonymous_woman_street_10s | 10.01s | `demo/video_runs_rpi_only/anonymous_woman_street_10s/source_anonymous_woman_street.mp4` | `demo/video_runs_rpi_only/anonymous_woman_street_10s/anonymous_woman_street_10s_rpi_only_3panel_realtime_sync_24fps.mp4` | 5 / 2087.9 ms | 300 / 119.6 ms | 300 / 362.4 ms |
| students_university_10s | 10.00s | `demo/video_runs_rpi_only/students_university_10s/source_students_university.mp4` | `demo/video_runs_rpi_only/students_university_10s/students_university_10s_rpi_only_3panel_realtime_sync_24fps.mp4` | 5 / 1849.0 ms | 300 / 117.7 ms | 300 / 381.1 ms |
| busy_city_street_10s | 10.00s | `demo/video_runs_rpi_only/busy_city_street_10s/source_busy_city_street.mp4` | `demo/video_runs_rpi_only/busy_city_street_10s/busy_city_street_10s_rpi_only_3panel_realtime_sync_24fps.mp4` | 5 / 1812.4 ms | 300 / 120.6 ms | 300 / 391.6 ms |
