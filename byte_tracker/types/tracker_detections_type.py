from typing_extensions import TypedDict
import numpy as np


class TrackerDetectionsType(TypedDict):
    xywh: np.ndarray
    conf: np.ndarray
    cls: np.ndarray
