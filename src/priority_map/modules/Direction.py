import cv2
import numpy as np


class Direction:
    def get_direction(
        self,
        numerical_heatmap: np.ndarray,
        came_from: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a unit vector toward the region with the most total heat."""
        numerical_heatmap = np.asarray(numerical_heatmap, dtype=np.float32)
        if numerical_heatmap.ndim != 2:
            raise ValueError("numerical_heatmap must be a two-dimensional array.")
        if numerical_heatmap.size == 0:
            return np.zeros(2, dtype=np.float32)

        numerical_heatmap = np.maximum(
            np.nan_to_num(numerical_heatmap),
            0,
        )
        region_count, region_labels = cv2.connectedComponents(
            (numerical_heatmap > 0).astype(np.uint8),
        )
        if region_count <= 1:
            return np.zeros(2, dtype=np.float32)

        region_heat = np.bincount(
            region_labels.ravel(),
            weights=numerical_heatmap.ravel(),
        )
        region_heat[0] = 0
        hottest_region = int(np.argmax(region_heat))
        region_mask = region_labels == hottest_region
        region_y, region_x = np.nonzero(region_mask)
        weights = numerical_heatmap[region_mask]

        target_x = np.average(region_x, weights=weights)
        target_y = np.average(region_y, weights=weights)
        center_y = numerical_heatmap.shape[0] // 2
        center_x = numerical_heatmap.shape[1] // 2

        direction = np.array(
            [
                target_x - center_x,
                center_y - target_y,
            ],
            dtype=np.float32,
        )
        magnitude = np.linalg.norm(direction)
        if magnitude == 0:
            return np.zeros(2, dtype=np.float32)

        return direction / magnitude
