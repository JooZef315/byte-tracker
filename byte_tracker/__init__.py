from dataclasses import dataclass
from typing import Mapping, Optional, Union

import numpy as np


class _SimpleNamespace:
    """Simple namespace for storing tracker configuration and results."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        """Allow subscripting with boolean arrays for filtering."""
        if isinstance(key, (list, tuple)) or (
            hasattr(key, "__iter__") and not isinstance(key, str)
        ):
            # Handle boolean indexing or list of indices
            import numpy as np

            key = np.asarray(key)

            # Create a new SimpleNamespace with filtered data
            result = _SimpleNamespace()
            for attr_name, attr_value in self.__dict__.items():
                if isinstance(attr_value, np.ndarray) and len(attr_value) > 0:
                    # Filter numpy arrays
                    result.__dict__[attr_name] = attr_value[key]
                else:
                    # Keep non-array attributes as-is
                    result.__dict__[attr_name] = attr_value
            return result
        else:
            # Single index access
            return self.__dict__[key]

    def __len__(self):
        """Return length based on first array attribute."""
        for attr_value in self.__dict__.values():
            if hasattr(attr_value, "__len__"):
                return len(attr_value)
        return 0


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


def _coerce_config(config: Optional[Union[TrackerConfig, Mapping[str, object]]]) -> TrackerConfig:
    if config is None:
        return DEFAULT_TRACKER_CONFIG
    if isinstance(config, TrackerConfig):
        return config
    if isinstance(config, Mapping):
        return TrackerConfig(**config)
    raise TypeError(
        "config must be a TrackerConfig instance, a mapping of config values, or None"
    )


class Tracker:
    """
    BYTETracker with simple interface for use with any detector.

    Usage:
        tracker = tracker(config=DEFAULT_TRACKER_CONFIG)

        # Get detections from your detector (in xywh format with scores and classes)
        # detections should be a dict or object with:
        #   - xywh: np.ndarray of shape (N, 4) - center x, center y, width, height
        #   - conf: np.ndarray of shape (N,) - confidence scores
        #   - cls: np.ndarray of shape (N,) - class IDs

        tracks = tracker.update(detections, frame)

        # tracks is np.ndarray of shape (M, 8):
        # [x1, y1, x2, y2, track_id, score, class_id, detection_idx]
    """

    def __init__(self, config: Optional[Union[TrackerConfig, Mapping[str, object]]] = None):
        """
        Initialize tracker with configuration.

        Args:
            config: TrackerConfig instance, a mapping of config values, or None to use defaults.
        """
        # Initialize ByteTracker
        from .byte_tracker import BYTETracker

        args = _coerce_config(config)
        self.tracker = BYTETracker(args)

    def update(self, detections, frame: np.ndarray = None) -> np.ndarray:
        """
        Update tracker with new detections.

        Args:
            detections: Detection results with attributes:
                - xywh: np.ndarray of shape (N, 4) - [center_x, center_y, width, height]
                - conf: np.ndarray of shape (N,) - confidence scores
                - cls: np.ndarray of shape (N,) - class IDs

        Returns:
            np.ndarray of shape (M, 8): [x1, y1, x2, y2, track_id, score, class_id, detection_idx]
            where M is the number of active tracks
        """
        # Convert detections to SimpleNamespace if it's a dict
        if isinstance(detections, dict):
            det_obj = _SimpleNamespace(**detections)
        else:
            det_obj = detections

        # Ensure arrays are numpy arrays
        if not isinstance(det_obj.xywh, np.ndarray):
            det_obj.xywh = np.array(det_obj.xywh)
        if not isinstance(det_obj.conf, np.ndarray):
            det_obj.conf = np.array(det_obj.conf)
        if not isinstance(det_obj.cls, np.ndarray):
            det_obj.cls = np.array(det_obj.cls)

        return self.tracker.update(det_obj, img=frame)

    def reset(self):
        """Reset tracker state (clears all tracks)."""
        self.tracker.reset()


__all__ = [
    "Tracker",
    "TrackerConfig",
    "DEFAULT_TRACKER_CONFIG",
]
