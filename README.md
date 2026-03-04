# byte-tracker

A self-contained ByteTrack implementation extracted from Ultralytics YOLO. This tracker works with any detector and has no Ultralytics runtime dependency.

## Features

- ByteTrack algorithm with Kalman filtering
- Self-contained (numpy, scipy, lap only)
- Works with any detector
- Simple config object (no YAML)

## Installation

```
pip install git+https://github.com/JooZef315/byte-tracker.git
```

## Quickstart

```python
import numpy as np
from byte_tracker import Tracker, TrackerConfig, DEFAULT_TRACKER_CONFIG

# Use defaults
trk = Tracker(DEFAULT_TRACKER_CONFIG)

# Or customize config
cfg = TrackerConfig(
    track_high_thresh=0.6,
    track_low_thresh=0.1,
    new_track_thresh=0.7,
    track_buffer=45,
    match_thresh=0.8,
    fuse_score=True,
)
trk = tracker(cfg)

# Detections from your detector
detections = {
    "xywh": np.array([[100, 200, 50, 80], [300, 400, 60, 90]]),  # [cx, cy, w, h]
    "conf": np.array([0.9, 0.85]),
    "cls": np.array([0, 2]),
}

tracks = trk.update(detections, frame=None)
for track_row in tracks:
    x1, y1, x2, y2, track_id, score, cls_id, det_idx = track_row
    print(
        f"Track {int(track_id)}: bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], class={int(cls_id)}"
    )
```

## Configuration

`TrackerConfig` is the public configuration type. Defaults are also exposed as `DEFAULT_TRACKER_CONFIG`.

Fields and defaults:

- `track_high_thresh`: `0.7`
- `track_low_thresh`: `0.1`
- `new_track_thresh`: `0.7`
- `track_buffer`: `60`
- `match_thresh`: `0.8`
- `fuse_score`: `True`

## API

- `tracker(config: TrackerConfig, dict, or None)`
- `tracker.update(detections, frame=None) -> np.ndarray`
- `tracker.reset()`

## Detection Format

```python
detections = {
    "xywh": np.ndarray,  # shape: (N, 4) - [center_x, center_y, width, height]
    "conf": np.ndarray,  # shape: (N,) - confidence scores [0, 1]
    "cls": np.ndarray,   # shape: (N,) - class IDs (integers)
}
```

**Important**: Bounding boxes must be in **center format** (xywh), not corner format (xyxy).

## Output Format

The tracker returns a numpy array of shape `(M, 8)`:

```
[x1, y1, x2, y2, track_id, score, class_id, detection_idx]
```

## Files

- `__init__.py` - Public entry point (tracker + config)
- `byte_tracker.py` - ByteTrack algorithm (BYTETracker)
- `utils/strack.py` - Single-track state (STrack)
- `basetrack.py` - Base tracking classes
- `utils/kalman_filter.py` - Kalman filter for motion prediction
- `utils/matching.py` - Detection-track matching algorithms
- `utils/ops.py` - Bounding box operations

## License

AGPL-3.0 (see `LICENSE`). Portions of this project were extracted from Ultralytics YOLO and remain under AGPL-3.0; see `THIRD_PARTY_NOTICES.md` for details.

## References

- ByteTrack paper: https://arxiv.org/abs/2110.06864
- Ultralytics: https://github.com/ultralytics/ultralytics
