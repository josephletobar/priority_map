import numpy as np

from priority_map.modules.drone_motion.motion import (
    MotionProvider,
    normalize_vector,
)


class GpsMotion(MotionProvider):
    def __init__(self):
        self.previous_position = None

    def get_came_from(self, frame, flow_transform) -> np.ndarray | None:
        easting = getattr(frame, "easting", None)
        northing = getattr(frame, "northing", None)
        if easting is None or northing is None:
            return None

        current_position = np.array(
            [float(easting), float(northing)],
            dtype=np.float64,
        )
        previous_position = self.previous_position
        self.previous_position = current_position
        if previous_position is None:
            return None

        came_from_world = np.array(
            [
                previous_position[0] - current_position[0],
                previous_position[1] - current_position[1],
                0.0,
            ],
            dtype=np.float64,
        )
        if np.linalg.norm(came_from_world) == 0:
            return np.zeros(2, dtype=np.float32)

        orientation = getattr(frame, "orientation", None)
        if orientation is None:
            return None

        quaternion = np.asarray(orientation, dtype=np.float64)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            return None

        quaternion_norm = np.linalg.norm(quaternion)
        if quaternion_norm == 0:
            return None
        x, y, z, w = quaternion / quaternion_norm

        body_to_world = np.array(
            [
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )
        came_from_local = body_to_world.T @ came_from_world
        return normalize_vector(came_from_local[:2])
