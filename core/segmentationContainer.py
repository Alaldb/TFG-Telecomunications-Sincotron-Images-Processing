from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

class SegmentationMethod(Enum):
    ICM="ICM"
    GRAPH_CUTS="GraphCuts"

@dataclass
class SegmentationContainer:
    original_image: np.ndarray
    mask: np.ndarray
    final_image: np.ndarray
    num_states: int
    parameters: dict
    initial_labels: np.ndarray
    method: SegmentationMethod
    method_configuration: dict = field(default_factory=dict)

