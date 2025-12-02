# Count Persons

Multi-camera people detection and re-identification demo built with Flask, OpenVINO, and MongoDB. It ingests multiple video streams, detects unique visitors, tracks groups, and publishes hourly/daily stats to Mongo.

## Features
- Person detection via `person-detection-retail-0013` and re-ID via `person-reidentification-retail-0287`.
- Multi-camera composited stream with live counts, FPS overlay, and group clustering.
- Hourly and daily metric snapshots saved to MongoDB (`hourly_stats`, `daily_stats`).
- Re-identification registry with daily dedupe, group/individual separation, and embedding-based anti-duplication.
- CPU-friendly knobs: per-camera frame skipping, live-stream auto-restart, registry pruning.

## Requirements
- Python 3.10 (venv included in repo for reference)
- Dependencies: `pip install -r requirements.txt` (or ensure `flask`, `opencv-python`, `openvino`, `pymongo`, `python-dotenv`, `numpy`)
- MongoDB instance (defaults to `mongodb://localhost:27017`)

## Quick Start
1. (Optional) Create/activate virtualenv.
2. Create or edit `.env` and set values:
   ```
   MONGO_URI=mongodb://localhost:27017
   DB_NAME=person_analytics
   FRAME_PROCESS_INTERVAL=1
   ALLOW_STREAM_RESTART=true
   ```
3. Adjust `CAM_SOURCES` in `app.py` to your video files, RTSP URLs, or camera indices.
4. Run `python app.py` (or `flask run` if configured). Visit `http://localhost:5000`.

## Environment Knobs
- `MONGO_URI`, `DB_NAME`: destination for hourly/daily snapshots.
- `FRAME_PROCESS_INTERVAL`: run detection every N frames per camera (1 = every frame).
- `ALLOW_STREAM_RESTART`, `RESTART_BACKOFF_SECONDS`: auto-retry live streams when they drop.

## Metrics & Reporting
- `/video_feed`: MJPEG stream showing merged camera view with IDs and FPS.
- `/metrics`: JSON with totals, individuals, groups, hourly history, last daily snapshot.
- Background threads push hourly snapshots at every `:00` and a daily summary at midnight, then reset counters for the next day.

## Scalability Notes
- Each camera has its own reader thread; finished demo videos are removed automatically, live sources restart after failures.
- Processing still scales roughly linearly with camera count; use `FRAME_PROCESS_INTERVAL`, lower FPS, or hardware accelerators for bigger deployments.
- The re-ID registry now grows freely during the day and is cleared once at midnight; monitor memory usage if you track very large crowds.
- Ensure MongoDB is reachable; otherwise snapshot threads will log insert errors but keep the app running.
