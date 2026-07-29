import numpy as np

from priority_map.modules.drone_motion.motion import (
    MotionProvider,
    normalize_vector,
)


class FlowMotion(MotionProvider):
    def get_came_from(self, frame, flow_transform) -> np.ndarray:
        if flow_transform is None:
            return np.zeros(2, dtype=np.float32)

        dx, dy = flow_transform
        if not np.isfinite(dx) or not np.isfinite(dy):
            return np.zeros(2, dtype=np.float32)

        return normalize_vector((-dx, dy))
