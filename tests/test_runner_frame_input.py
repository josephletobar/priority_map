import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from priority_map.modules.Direction import Direction
from priority_map.modules.drone_motion import DroneMotion
from priority_map.runner import PriorityMapRunner


class RunnerFrameInputTests(unittest.TestCase):
    def bare_runner(self):
        runner = PriorityMapRunner.__new__(PriorityMapRunner)
        runner.max_image_edge = 0
        runner.frames_processed = 7
        runner.gps_csv_path = None
        runner.debug = False
        runner.direction = Direction()
        runner.drone_motion = DroneMotion()
        return runner

    def test_runner_can_initialize_without_a_dataset(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("priority_map.runner.SceneUnderstanding"),
            patch("priority_map.runner.Segment"),
            patch("priority_map.runner.GraphBuilder"),
            patch("priority_map.runner.Heatmap"),
            patch("priority_map.runner.FlowLocalizer"),
            patch("priority_map.runner.GpsLocalizer"),
            patch("priority_map.runner.VideoOutput"),
            patch("priority_map.runner.PanoramaBuilder"),
        ):
            runner = PriorityMapRunner(output_dir=temp_dir, record=False)
            try:
                self.assertIsNone(runner.dataset_root)
                self.assertIsNone(runner.query_images_dir)
                self.assertFalse(runner.has_next())
            finally:
                runner.close()

    def test_in_memory_frame_is_prepared_without_mutating_caller_array(self):
        runner = self.bare_runner()
        original = np.full((4, 5, 3), 200, dtype=np.uint8)
        original_copy = original.copy()

        packet = runner._frame_from_input(original)

        np.testing.assert_array_equal(original, original_copy)
        self.assertEqual(packet.image_name, "frame_000007.png")
        self.assertIsNone(packet.image_path)
        self.assertEqual(packet.frame_index, 7)
        np.testing.assert_array_equal(packet.image[:, :, 0], 200)
        np.testing.assert_array_equal(packet.image[:, :, 1], 130)
        np.testing.assert_array_equal(packet.image[:, :, 2], 160)

    def test_image_path_is_loaded_and_metadata_can_be_supplied(self):
        runner = self.bare_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "incoming.png"
            self.assertTrue(
                cv2.imwrite(
                    str(image_path),
                    np.full((3, 4, 3), 100, dtype=np.uint8),
                )
            )

            packet = runner._frame_from_input(
                image_path,
                frame_index=12,
                easting=1.5,
                northing=2.5,
                altitude=3.5,
                orientation=(0.0, 0.0, 0.0, 1.0),
            )

        self.assertEqual(packet.image_name, "incoming.png")
        self.assertEqual(packet.image_path, str(image_path))
        self.assertEqual(packet.frame_index, 12)
        self.assertEqual(packet.easting, 1.5)
        self.assertEqual(packet.northing, 2.5)
        self.assertEqual(packet.altitude, 3.5)
        self.assertEqual(packet.orientation, (0.0, 0.0, 0.0, 1.0))

    def test_run_frame_skips_folder_reader_when_image_is_supplied(self):
        runner = self.bare_runner()
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        runner.sam_step = 2
        runner.task_description = "Find cars"
        runner.segmentation = MagicMock()
        runner.segmentation.get_segmentations.return_value = SimpleNamespace(
            segmentations=[],
            sam3_seconds=0.0,
            flow_transform=None,
        )
        came_from = np.array([-1.0, 0.0], dtype=np.float32)
        runner.drone_motion = MagicMock()
        runner.drone_motion.get_came_from.return_value = came_from
        runner.direction = MagicMock()
        runner.direction.get_direction.return_value = np.array(
            [1.0, 0.0],
            dtype=np.float32,
        )
        runner.flow_localizer = MagicMock()
        runner.gps_localizer = MagicMock()
        runner.masks = set()
        runner.heatmap = MagicMock()
        numerical_heatmap = np.zeros(image.shape[:2], dtype=np.float32)
        numerical_heatmap[1, 2] = 100.0
        runner.heatmap.draw_heatmap.side_effect = (
            lambda prepared_image, _: (
                prepared_image.copy(),
                np.zeros_like(prepared_image),
                numerical_heatmap,
            )
        )
        runner.heatmap_video_output = MagicMock()
        runner.graph_builder = MagicMock()
        runner.last_graph_frame = None
        runner.graph_view = "spatial"
        runner.panoramic = False
        runner.video_output = MagicMock()
        runner.video_output.handle_frame.return_value = True
        runner.video_output.last_key = -1
        runner.task = "Find cars"

        with tempfile.TemporaryDirectory() as temp_dir:
            runner.output_dir = Path(temp_dir)
            runner.observations_csv = runner.output_dir / "observations.csv"
            with patch.object(runner, "get_next_frame") as get_next_frame:
                result = runner.run_frame(image)

        get_next_frame.assert_not_called()
        self.assertTrue(result.keep_running)
        self.assertEqual(result.image_name, "frame_000007.png")
        self.assertIsNone(result.image_path)
        self.assertEqual(result.frame_index, 7)
        np.testing.assert_array_equal(
            result.direction,
            np.array([1.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(result.came_from, came_from)
        runner.direction.get_direction.assert_called_once_with(
            numerical_heatmap,
            came_from,
        )
        self.assertEqual(result.numerical_heatmap.shape, image.shape[:2])
        self.assertEqual(runner.frames_processed, 8)

    def test_debug_mode_draws_direction_and_came_from_arrows(self):
        runner = self.bare_runner()
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        came_from = np.array([-1.0, 0.0], dtype=np.float32)

        with patch("priority_map.runner.cv2.arrowedLine") as arrowed_line:
            unchanged = runner._draw_debug_directions(
                image,
                direction,
                came_from,
            )
            arrowed_line.assert_not_called()

            runner.debug = True
            output = runner._draw_debug_directions(
                image,
                direction,
                came_from,
            )

        self.assertIs(unchanged, image)
        self.assertIsNot(output, image)
        self.assertEqual(arrowed_line.call_count, 2)
        direction_call, came_from_call = arrowed_line.call_args_list
        self.assertEqual(direction_call.args[1], (50, 40))
        self.assertEqual(direction_call.args[2], (50, 20))
        self.assertEqual(direction_call.args[3], (255, 255, 255))
        self.assertEqual(came_from_call.args[1], (50, 40))
        self.assertEqual(came_from_call.args[2], (30, 40))
        self.assertEqual(came_from_call.args[3], (255, 0, 255))

    def test_debug_arrows_skip_zero_vectors(self):
        runner = self.bare_runner()
        runner.debug = True
        image = np.zeros((80, 100, 3), dtype=np.uint8)

        with patch("priority_map.runner.cv2.arrowedLine") as arrowed_line:
            runner._draw_debug_directions(
                image,
                np.array([1.0, 0.0], dtype=np.float32),
                np.zeros(2, dtype=np.float32),
            )

        arrowed_line.assert_called_once()


if __name__ == "__main__":
    unittest.main()
