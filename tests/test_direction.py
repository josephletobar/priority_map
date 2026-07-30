import unittest
from types import SimpleNamespace

import numpy as np

from priority_map.modules.Direction import Direction
from priority_map.modules.Heatmap import Heatmap


class DirectionTests(unittest.TestCase):
    def setUp(self):
        self.direction = Direction()

    def heatmap_with_target(self, x, y, heat=80.0):
        numerical_heatmap = np.zeros((101, 101), dtype=np.float32)
        numerical_heatmap[y, x] = heat
        return numerical_heatmap

    def test_returns_navigation_unit_vectors(self):
        cases = [
            ((80, 50), np.array([1.0, 0.0], dtype=np.float32)),
            ((20, 50), np.array([-1.0, 0.0], dtype=np.float32)),
            ((50, 20), np.array([0.0, 1.0], dtype=np.float32)),
            ((50, 80), np.array([0.0, -1.0], dtype=np.float32)),
        ]

        for (x, y), expected in cases:
            with self.subTest(target=(x, y)):
                np.testing.assert_allclose(
                    self.direction.get_direction(self.heatmap_with_target(x, y)),
                    expected,
                )

    def test_normalizes_diagonal_vector(self):
        direction = self.direction.get_direction(
            self.heatmap_with_target(80, 20)
        )

        np.testing.assert_allclose(
            direction,
            np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2),
        )
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=6)

    def test_centered_heat_defaults_forward_and_empty_heat_holds(self):
        centered = self.heatmap_with_target(50, 50)
        empty = np.zeros((101, 101), dtype=np.float32)

        np.testing.assert_allclose(
            self.direction.get_direction(centered),
            np.array([0.0, 1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            self.direction.get_direction(empty),
            np.zeros(2, dtype=np.float32),
        )

    def test_requires_a_two_dimensional_numerical_heatmap(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            self.direction.get_direction(
                np.zeros((10, 10, 3), dtype=np.uint8)
            )

    def test_patch_with_more_total_heat_beats_a_hotter_single_pixel(self):
        numerical_heatmap = np.zeros((101, 101), dtype=np.float32)
        numerical_heatmap[50, 80] = 100
        numerical_heatmap[45:56, 10:21] = 5

        np.testing.assert_allclose(
            self.direction.get_direction(numerical_heatmap),
            np.array([-1.0, 0.0], dtype=np.float32),
        )

    def test_points_to_the_center_of_the_hottest_patch(self):
        numerical_heatmap = np.zeros((101, 101), dtype=np.float32)
        numerical_heatmap[22, 62] = 100

        np.testing.assert_allclose(
            self.direction.get_direction(numerical_heatmap),
            np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2),
        )

    def test_came_from_and_coverage_reduce_the_search_space(self):
        numerical_heatmap = np.zeros((101, 101), dtype=np.float32)
        numerical_heatmap[50, 80] = 100
        numerical_heatmap[20, 50] = 90
        numerical_heatmap[50, 20] = 80

        direction = self.direction.get_direction(
            numerical_heatmap,
            came_from=np.array([1.0, 0.0], dtype=np.float32),
            coverage_directions=[
                np.array([0.0, 1.0], dtype=np.float32),
            ],
        )

        np.testing.assert_allclose(
            direction,
            np.array([-1.0, 0.0], dtype=np.float32),
        )

    def test_geographic_validator_rejects_patch_and_uses_next_best(self):
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[40:60, 80:100] = 100.0
        heatmap[0:20, 40:60] = 50.0

        decision = Direction().get_decision(
            heatmap,
            candidate_validator=lambda direction: direction[0] <= 0.5,
        )

        self.assertGreater(decision.direction[1], 0.0)
        self.assertLessEqual(decision.direction[0], 0.5)

    def test_all_geographic_candidates_blocked_returns_hold(self):
        decision = Direction().get_decision(
            np.ones((20, 20), dtype=np.float32),
            candidate_validator=lambda _: False,
        )

        np.testing.assert_array_equal(
            decision.direction,
            np.zeros(2, dtype=np.float32),
        )

    def test_heat_outside_a_forbidden_cone_remains_available(self):
        numerical_heatmap = np.zeros((101, 101), dtype=np.float32)
        numerical_heatmap[50, 80] = 100

        direction = self.direction.get_direction(
            numerical_heatmap,
            came_from=np.array([0.0, 1.0], dtype=np.float32),
        )

        np.testing.assert_allclose(
            direction,
            np.array([1.0, 0.0], dtype=np.float32),
        )

    def test_excluded_only_hot_patch_uses_an_available_patch(self):
        numerical_heatmap = np.zeros((101, 101), dtype=np.float32)
        numerical_heatmap[50, 80] = 100

        direction = self.direction.get_direction(
            numerical_heatmap,
            came_from=np.array([1.0, 0.0], dtype=np.float32),
        )

        np.testing.assert_allclose(
            direction,
            np.array([0.0, 1.0], dtype=np.float32),
        )

    def test_all_patches_forbidden_still_returns_a_direction(self):
        numerical_heatmap = np.zeros((101, 101), dtype=np.float32)
        numerical_heatmap[50, 80] = 100

        direction = self.direction.get_direction(
            numerical_heatmap,
            coverage_directions=[
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([-1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([0.0, -1.0], dtype=np.float32),
            ],
        )

        np.testing.assert_allclose(
            direction,
            np.array([1.0, 0.0], dtype=np.float32),
        )

    def test_exclusion_angle_is_configurable(self):
        diagonal = np.array([1.0, 1.0], dtype=np.float32)
        candidate = np.array([1.0, 0.0], dtype=np.float32)

        self.assertEqual(
            Direction(exclusion_angle_degrees=30).coverage_count(
                candidate,
                [diagonal],
            ),
            0,
        )
        self.assertEqual(
            Direction(exclusion_angle_degrees=60).coverage_count(
                candidate,
                [diagonal],
            ),
            1,
        )

    def test_exclusion_angle_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            Direction(exclusion_angle_degrees=0)
        with self.assertRaises(ValueError):
            Direction(exclusion_angle_degrees=181)

    def test_multiple_hard_forbidden_bearings_handle_a_corner(self):
        numerical_heatmap = np.zeros((101, 101), dtype=np.float32)
        numerical_heatmap[20, 20] = 100
        numerical_heatmap[80, 80] = 75

        direction = self.direction.get_direction(
            numerical_heatmap,
            forbidden_directions=[
                np.array([-1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ],
        )

        self.assertGreater(direction[0], 0)
        self.assertLess(direction[1], 0)

    def test_fully_covered_scene_uses_least_covered_patch(self):
        numerical_heatmap = np.zeros((101, 101), dtype=np.float32)
        numerical_heatmap[50, 80] = 100
        numerical_heatmap[50, 20] = 75
        coverage_directions = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([-1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
        ]

        decision = self.direction.get_decision(
            numerical_heatmap,
            coverage_directions=coverage_directions,
        )

        np.testing.assert_allclose(decision.direction, [-1.0, 0.0])
        self.assertEqual(decision.coverage_count, 1)
        self.assertEqual(decision.patch_heat, 75.0)

    def test_empty_heatmap_with_coverage_explores_an_uncovered_patch(self):
        decision = self.direction.get_decision(
            np.zeros((101, 101), dtype=np.float32),
            coverage_directions=[
                np.array([1.0, 0.0], dtype=np.float32),
            ],
        )

        self.assertGreater(float(np.linalg.norm(decision.direction)), 0.0)
        self.assertEqual(decision.patch_heat, 0.0)
        self.assertEqual(decision.coverage_count, 0)
        self.assertFalse(
            self.direction.is_blocked_by_came_from(
                decision.direction,
                np.array([1.0, 0.0], dtype=np.float32),
            )
        )

    def test_came_from_remains_hard_when_every_patch_has_coverage(self):
        numerical_heatmap = np.zeros((101, 101), dtype=np.float32)
        numerical_heatmap[50, 20] = 100
        came_from = np.array([-1.0, 0.0], dtype=np.float32)

        direction = self.direction.get_direction(
            numerical_heatmap,
            came_from=came_from,
            coverage_directions=[
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([-1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([0.0, -1.0], dtype=np.float32),
            ],
        )

        self.assertFalse(
            self.direction.is_blocked_by_came_from(
                direction,
                came_from,
            )
        )

    def test_forbidden_cone_removes_the_entire_patch(self):
        numerical_heatmap = np.zeros((101, 101), dtype=np.float32)
        numerical_heatmap[21, 61] = 100
        numerical_heatmap[50, 20] = 75

        direction = self.direction.get_direction(
            numerical_heatmap,
            came_from=np.array([1.0, 0.0], dtype=np.float32),
        )

        np.testing.assert_allclose(
            direction,
            np.array([-1.0, 0.0], dtype=np.float32),
        )


class NumericalHeatmapTests(unittest.TestCase):
    def test_returns_post_blur_values_used_for_direction(self):
        image = np.zeros((80, 101, 3), dtype=np.uint8)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[40, 80] = 1
        region = SimpleNamespace(
            mask=mask,
            score=75.0,
            centroid=(80, 40),
            label="target",
            color=None,
        )

        heatmap_text, heatmap_only, numerical_heatmap = Heatmap(
            blur_spread=15,
            dilation_scale=0,
        ).draw_heatmap(image, [region])

        self.assertEqual(numerical_heatmap.shape, image.shape[:2])
        self.assertEqual(numerical_heatmap.dtype, np.float32)
        self.assertGreater(float(numerical_heatmap.max()), 0.0)
        self.assertGreater(float(numerical_heatmap[39, 80]), 0.0)
        self.assertEqual(heatmap_only.shape, image.shape)
        self.assertEqual(heatmap_text.shape, image.shape)
        np.testing.assert_allclose(
            Direction().get_direction(numerical_heatmap),
            np.array([1.0, 0.0], dtype=np.float32),
        )

    def test_no_regions_returns_an_empty_numerical_heatmap(self):
        image = np.zeros((12, 16, 3), dtype=np.uint8)

        _, heatmap_only, numerical_heatmap = Heatmap().draw_heatmap(image, [])

        self.assertIsNone(heatmap_only)
        np.testing.assert_array_equal(
            numerical_heatmap,
            np.zeros(image.shape[:2], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
