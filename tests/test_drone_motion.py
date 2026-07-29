import unittest
from types import SimpleNamespace

import numpy as np

from priority_map.modules.drone_motion import DroneMotion
from priority_map.modules.drone_motion.flow_motion import FlowMotion
from priority_map.modules.drone_motion.gps_motion import GpsMotion


IDENTITY_ORIENTATION = (0.0, 0.0, 0.0, 1.0)


def frame(easting=None, northing=None, orientation=None):
    return SimpleNamespace(
        easting=easting,
        northing=northing,
        orientation=orientation,
    )


class FlowMotionTests(unittest.TestCase):
    def test_converts_flow_to_normalized_came_from_vector(self):
        came_from = FlowMotion().get_came_from(frame(), (3.0, 4.0))

        np.testing.assert_allclose(
            came_from,
            np.array([-0.6, 0.8], dtype=np.float32),
        )

    def test_missing_or_stationary_flow_returns_zero_vector(self):
        flow_motion = FlowMotion()

        np.testing.assert_array_equal(
            flow_motion.get_came_from(frame(), None),
            np.zeros(2, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            flow_motion.get_came_from(frame(), (0.0, 0.0)),
            np.zeros(2, dtype=np.float32),
        )


class GpsMotionTests(unittest.TestCase):
    def test_uses_previous_minus_current_position(self):
        gps_motion = GpsMotion()

        first = gps_motion.get_came_from(
            frame(0.0, 0.0, IDENTITY_ORIENTATION),
            None,
        )
        second = gps_motion.get_came_from(
            frame(1.0, 0.0, IDENTITY_ORIENTATION),
            None,
        )

        self.assertIsNone(first)
        np.testing.assert_allclose(
            second,
            np.array([-1.0, 0.0], dtype=np.float32),
        )

    def test_rotates_world_delta_into_current_local_axes(self):
        gps_motion = GpsMotion()
        half_sqrt = np.sqrt(0.5)
        ninety_degrees_about_z = (0.0, 0.0, half_sqrt, half_sqrt)

        gps_motion.get_came_from(
            frame(0.0, 0.0, IDENTITY_ORIENTATION),
            None,
        )
        came_from = gps_motion.get_came_from(
            frame(0.0, 1.0, ninety_degrees_about_z),
            None,
        )

        np.testing.assert_allclose(
            came_from,
            np.array([-1.0, 0.0], dtype=np.float32),
            atol=1e-6,
        )


class DroneMotionSelectionTests(unittest.TestCase):
    def test_falls_back_to_flow_without_usable_gps(self):
        drone_motion = DroneMotion()

        came_from = drone_motion.get_came_from(frame(), (3.0, 4.0))

        np.testing.assert_allclose(
            came_from,
            np.array([-0.6, 0.8], dtype=np.float32),
        )

    def test_missing_orientation_falls_back_to_flow(self):
        drone_motion = DroneMotion()
        drone_motion.get_came_from(frame(0.0, 0.0, None), (0.0, 0.0))

        came_from = drone_motion.get_came_from(
            frame(1.0, 0.0, None),
            (3.0, 4.0),
        )

        np.testing.assert_allclose(
            came_from,
            np.array([-0.6, 0.8], dtype=np.float32),
        )

    def test_valid_moving_gps_takes_precedence_over_flow(self):
        drone_motion = DroneMotion()
        drone_motion.get_came_from(
            frame(0.0, 0.0, IDENTITY_ORIENTATION),
            (0.0, 0.0),
        )

        came_from = drone_motion.get_came_from(
            frame(1.0, 0.0, IDENTITY_ORIENTATION),
            (0.0, 4.0),
        )

        np.testing.assert_allclose(
            came_from,
            np.array([-1.0, 0.0], dtype=np.float32),
        )

    def test_stationary_gps_does_not_fall_back_to_flow(self):
        drone_motion = DroneMotion()
        current_frame = frame(1.0, 2.0, IDENTITY_ORIENTATION)
        drone_motion.get_came_from(current_frame, (0.0, 0.0))

        came_from = drone_motion.get_came_from(
            current_frame,
            (3.0, 4.0),
        )

        np.testing.assert_array_equal(
            came_from,
            np.zeros(2, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
