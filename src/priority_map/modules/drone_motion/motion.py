from abc import ABC, abstractmethod

import numpy as np


def normalize_vector(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return np.zeros(2, dtype=np.float32)
    return vector / magnitude


class MotionProvider(ABC):
    @abstractmethod
    def get_came_from(self, frame, flow_transform) -> np.ndarray | None:
        pass
