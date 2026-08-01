"""Geospatial conversion helpers for Cesium and GPS frame data."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


def meters_per_degree(latitude_degrees: float) -> tuple[float, float]:
    """Return approximate WGS84 east/north meters per degree at latitude."""
    latitude = math.radians(float(latitude_degrees))
    meters_latitude = (
        111132.92
        - 559.82 * math.cos(2.0 * latitude)
        + 1.175 * math.cos(4.0 * latitude)
        - 0.0023 * math.cos(6.0 * latitude)
    )
    meters_longitude = (
        111412.84 * math.cos(latitude)
        - 93.5 * math.cos(3.0 * latitude)
        + 0.118 * math.cos(5.0 * latitude)
    )
    return meters_longitude, meters_latitude


def lonlat_to_local(
    longitude: float,
    latitude: float,
    origin_longitude: float,
    origin_latitude: float,
) -> np.ndarray:
    meters_longitude, meters_latitude = meters_per_degree(origin_latitude)
    return np.asarray(
        [
            (float(longitude) - float(origin_longitude)) * meters_longitude,
            (float(latitude) - float(origin_latitude)) * meters_latitude,
        ],
        dtype=np.float64,
    )


def local_to_lonlat(
    east: float,
    north: float,
    origin_longitude: float,
    origin_latitude: float,
) -> tuple[float, float]:
    meters_longitude, meters_latitude = meters_per_degree(origin_latitude)
    if meters_longitude <= 0.0 or meters_latitude <= 0.0:
        raise ValueError("The geographic origin cannot be projected.")
    return (
        float(origin_longitude) + float(east) / meters_longitude,
        float(origin_latitude) + float(north) / meters_latitude,
    )


@dataclass(frozen=True)
class CesiumFrameMetadata:
    longitude: float
    latitude: float
    camera_height_m: float
    ground_height_m: float
    heading_degrees: float
    vertical_fov_degrees: float

    @property
    def altitude_above_ground_m(self) -> float:
        return float(self.camera_height_m - self.ground_height_m)

    def is_valid(self) -> bool:
        values = np.asarray(
            [
                self.longitude,
                self.latitude,
                self.camera_height_m,
                self.ground_height_m,
                self.heading_degrees,
                self.vertical_fov_degrees,
            ],
            dtype=np.float64,
        )
        return bool(
            np.all(np.isfinite(values))
            and -180.0 <= self.longitude <= 180.0
            and -90.0 <= self.latitude <= 90.0
            and self.altitude_above_ground_m > 0.0
            and 1.0 <= self.vertical_fov_degrees < 179.0
        )

    def image_basis_en(self) -> tuple[np.ndarray, np.ndarray]:
        heading = math.radians(float(self.heading_degrees))
        image_right = np.asarray(
            [math.cos(heading), -math.sin(heading)],
            dtype=np.float64,
        )
        image_up = np.asarray(
            [math.sin(heading), math.cos(heading)],
            dtype=np.float64,
        )
        return image_right, image_up

    def footprint_size_m(self, image_shape) -> tuple[float, float]:
        height, width = image_shape[:2]
        if width <= 0 or height <= 0 or not self.is_valid():
            raise ValueError("A valid image and Cesium frame metadata are required.")
        footprint_height = (
            2.0
            * self.altitude_above_ground_m
            * math.tan(math.radians(self.vertical_fov_degrees) / 2.0)
        )
        footprint_width = footprint_height * (float(width) / float(height))
        return footprint_width, footprint_height

    def coverage_radius_m(self, image_shape, grid_size: int = 5) -> float:
        footprint_width, footprint_height = self.footprint_size_m(image_shape)
        grid_size = max(int(grid_size), 1)
        return 0.5 * math.hypot(
            footprint_width / grid_size,
            footprint_height / grid_size,
        )

    def project_pixel(self, pixel, image_shape) -> tuple[float, float]:
        """Project an image pixel onto the local tangent ground plane."""
        if not self.is_valid():
            raise ValueError("Cesium frame metadata is invalid.")

        height, width = image_shape[:2]
        footprint_width, footprint_height = self.footprint_size_m(image_shape)
        pixel_x, pixel_y = map(float, pixel)
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        image_right_m = (
            (pixel_x - center_x) / max(float(width), 1.0)
        ) * footprint_width
        image_up_m = (
            (center_y - pixel_y) / max(float(height), 1.0)
        ) * footprint_height
        image_right, image_up = self.image_basis_en()
        offset_en = (
            image_right * image_right_m
            + image_up * image_up_m
        )
        return local_to_lonlat(
            offset_en[0],
            offset_en[1],
            self.longitude,
            self.latitude,
        )

    def image_direction_to_en(self, direction) -> np.ndarray:
        direction = np.asarray(direction, dtype=np.float64)
        if (
            direction.shape != (2,)
            or not np.all(np.isfinite(direction))
        ):
            return np.zeros(2, dtype=np.float64)
        magnitude = float(np.linalg.norm(direction))
        if magnitude <= np.finfo(np.float64).eps:
            return np.zeros(2, dtype=np.float64)

        image_right, image_up = self.image_basis_en()
        result = (
            image_right * (direction[0] / magnitude)
            + image_up * (direction[1] / magnitude)
        )
        result_magnitude = float(np.linalg.norm(result))
        if result_magnitude <= np.finfo(np.float64).eps:
            return np.zeros(2, dtype=np.float64)
        return result / result_magnitude

    def project_local_point(
        self,
        point_en,
        camera_center_en,
        image_shape,
    ) -> tuple[float, float]:
        """Project a local east/north ground point into the camera image."""
        point_en = np.asarray(point_en, dtype=np.float64)
        camera_center_en = np.asarray(camera_center_en, dtype=np.float64)
        if (
            point_en.shape != (2,)
            or camera_center_en.shape != (2,)
            or not np.all(np.isfinite(point_en))
            or not np.all(np.isfinite(camera_center_en))
        ):
            raise ValueError("Finite two-dimensional local positions are required.")

        height, width = image_shape[:2]
        footprint_width, footprint_height = self.footprint_size_m(image_shape)
        image_right, image_up = self.image_basis_en()
        offset = point_en - camera_center_en
        offset_right = float(np.dot(offset, image_right))
        offset_up = float(np.dot(offset, image_up))
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        return (
            center_x + offset_right * float(width) / footprint_width,
            center_y - offset_up * float(height) / footprint_height,
        )

    def local_point_is_visible(
        self,
        point_en,
        camera_center_en,
        image_shape,
    ) -> bool:
        pixel_x, pixel_y = self.project_local_point(
            point_en,
            camera_center_en,
            image_shape,
        )
        height, width = image_shape[:2]
        return bool(
            0.0 <= pixel_x < float(width)
            and 0.0 <= pixel_y < float(height)
        )

    def observed_circle_coverage_ratio(
        self,
        camera_center_en,
        image_shape,
        observed_regions,
    ) -> float:
        """Return the unioned screen area covered by local ground circles."""
        height, width = image_shape[:2]
        if width <= 0 or height <= 0:
            return 0.0

        footprint_width, footprint_height = self.footprint_size_m(image_shape)
        coverage_mask = np.zeros((height, width), dtype=np.uint8)
        for center_en, radius_m in observed_regions:
            center_en = np.asarray(center_en, dtype=np.float64)
            radius_m = float(radius_m)
            if (
                center_en.shape != (2,)
                or not np.all(np.isfinite(center_en))
                or not np.isfinite(radius_m)
                or radius_m <= 0.0
            ):
                continue

            pixel_x, pixel_y = self.project_local_point(
                center_en,
                camera_center_en,
                image_shape,
            )
            radius_x = radius_m * float(width) / footprint_width
            radius_y = radius_m * float(height) / footprint_height
            if (
                pixel_x + radius_x < 0.0
                or pixel_x - radius_x >= float(width)
                or pixel_y + radius_y < 0.0
                or pixel_y - radius_y >= float(height)
            ):
                continue

            cv2.ellipse(
                coverage_mask,
                (int(round(pixel_x)), int(round(pixel_y))),
                (
                    max(1, int(round(radius_x))),
                    max(1, int(round(radius_y))),
                ),
                0.0,
                0.0,
                360.0,
                1,
                thickness=-1,
            )

        return float(np.count_nonzero(coverage_mask)) / float(coverage_mask.size)


def segment_intersects_circle(
    start,
    end,
    center,
    radius: float,
) -> bool:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    radius = max(float(radius), 0.0)
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= np.finfo(np.float64).eps:
        return bool(np.linalg.norm(center - start) <= radius)
    t = float(np.clip(np.dot(center - start, segment) / length_squared, 0.0, 1.0))
    closest = start + segment * t
    return bool(np.linalg.norm(center - closest) <= radius)
