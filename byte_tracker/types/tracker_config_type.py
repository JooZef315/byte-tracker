from dataclasses import dataclass

@dataclass(frozen=True)
class TrackerConfig:
    """Configuration for the ByteTrack tracker."""

    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    track_buffer: int = 45
    match_thresh: float = 0.8
    fuse_score: bool = True


DEFAULT_TRACKER_CONFIG = TrackerConfig()