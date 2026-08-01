import math
import unittest

import numpy as np

from priority_map.scripts.geospatial import (
    CesiumFrameMetadata,
    lonlat_to_local,
    segment_intersects_circle,
)


class CesiumFrameMetadataTests(unittest.TestCase):
    def metadata(self, **overrides):
        values = {
            "longitude": -121.87,
            "latitude": 37.21,
            "camera_height_m": 160.0,
            "ground_height_m": 100.0,
            "heading_degrees": 0.0,
            "vertical_fov_degrees": 60.0,
        }
        values.update(overrides)
        return CesiumFrameMetadata(**values)

    def test_image_center_projects_to_camera_ground_coordinate(self):
        metadata = self.metadata()
        longitude, latitude = metadata.project_pixel(
            (639.5, 359.5),
            (720, 1280, 3),
        )

        self.assertAlmostEqual(longitude, metadata.longitude, places=10)
        self.assertAlmostEqual(latitude, metadata.latitude, places=10)

    def test_image_edges_follow_heading_on_ground(self):
        image_shape = (720, 1280, 3)
        east_facing = self.metadata(heading_degrees=0.0)
        east_lon, east_lat = east_facing.project_pixel(
            (1279.5, 359.5),
            image_shape,
        )
        east_offset = lonlat_to_local(
            east_lon,
            east_lat,
            east_facing.longitude,
            east_facing.latitude,
        )
        self.assertGreater(east_offset[0], 0.0)
        self.assertAlmostEqual(east_offset[1], 0.0, places=6)

        rotated = self.metadata(heading_degrees=90.0)
        rotated_lon, rotated_lat = rotated.project_pixel(
            (1279.5, 359.5),
            image_shape,
        )
        rotated_offset = lonlat_to_local(
            rotated_lon,
            rotated_lat,
            rotated.longitude,
            rotated.latitude,
        )
        self.assertAlmostEqual(rotated_offset[0], 0.0, places=6)
        self.assertLess(rotated_offset[1], 0.0)

    def test_node_radius_is_half_of_one_steering_patch_diagonal(self):
        metadata = self.metadata()
        width, height = metadata.footprint_size_m((720, 1280, 3))

        radius = metadata.coverage_radius_m(
            (720, 1280, 3),
            grid_size=5,
        )

        self.assertAlmostEqual(
            radius,
            0.5 * math.hypot(width / 5.0, height / 5.0),
        )

    def test_local_ground_points_project_into_the_camera_view(self):
        metadata = self.metadata(heading_degrees=0.0)
        image_shape = (100, 100, 3)

        self.assertTrue(
            metadata.local_point_is_visible(
                [0.0, 0.0],
                [0.0, 0.0],
                image_shape,
            )
        )
        self.assertFalse(
            metadata.local_point_is_visible(
                [100.0, 0.0],
                [0.0, 0.0],
                image_shape,
            )
        )

    def test_observed_circle_coverage_uses_union_area(self):
        metadata = self.metadata()
        image_shape = (100, 100, 3)
        region = ([0.0, 0.0], 15.0)

        single_ratio = metadata.observed_circle_coverage_ratio(
            [0.0, 0.0],
            image_shape,
            [region],
        )
        duplicate_ratio = metadata.observed_circle_coverage_ratio(
            [0.0, 0.0],
            image_shape,
            [region, region],
        )

        self.assertGreater(single_ratio, 0.0)
        self.assertEqual(duplicate_ratio, single_ratio)

    def test_segment_circle_intersection_is_local_not_bearing_based(self):
        self.assertTrue(
            segment_intersects_circle(
                [0.0, 0.0],
                [5.0, 0.0],
                [3.0, 0.0],
                1.0,
            )
        )
        self.assertFalse(
            segment_intersects_circle(
                [0.0, 0.0],
                [5.0, 0.0],
                [20.0, 0.0],
                2.0,
            )
        )

    def test_invalid_ground_height_is_rejected(self):
        self.assertFalse(
            self.metadata(
                camera_height_m=100.0,
                ground_height_m=100.0,
            ).is_valid()
        )

if __name__ == "__main__":
    unittest.main()
