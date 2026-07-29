import numpy as np

from priority_map.modules.drone_motion.flow_motion import FlowMotion
from priority_map.modules.drone_motion.gps_motion import GpsMotion


class DroneMotion:
    def __init__(self):
        self.gps = GpsMotion()
        self.flow = FlowMotion()

    def get_came_from(self, frame, flow_transform) -> np.ndarray:
        gps_came_from = self.gps.get_came_from(frame, flow_transform)
        if gps_came_from is not None:
            return gps_came_from
        return self.flow.get_came_from(frame, flow_transform)
