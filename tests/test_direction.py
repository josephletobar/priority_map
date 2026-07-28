import unittest
from types import SimpleNamespace

import numpy as np

from priority_map.modules.Direction import get_direction
from priority_map.modules.Heatmap import Heatmap


class DirectionTests(unittest.TestCase):
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
                    get_direction(self.heatmap_with_target(x, y)),
                    expected,
                )

    def test_normalizes_diagonal_vector(self):
        direction = get_direction(self.heatmap_with_target(80, 20))

        np.testing.assert_allclose(
            direction,
            np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2),
        )
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=6)

    def test_centered_or_empty_heatmap_returns_zero_vector(self):
        centered = self.heatmap_with_target(50, 50)
        empty = np.zeros((101, 101), dtype=np.float32)

        np.testing.assert_array_equal(
            get_direction(centered),
            np.zeros(2, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            get_direction(empty),
            np.zeros(2, dtype=np.float32),
        )

    def test_requires_a_two_dimensional_numerical_heatmap(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            get_direction(np.zeros((10, 10, 3), dtype=np.uint8))


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
            get_direction(numerical_heatmap),
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
