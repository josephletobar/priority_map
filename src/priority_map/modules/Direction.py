import numpy as np


class Direction:
    EXCLUSION_ANGLE_DEGREES = 60.0
    GRID_SIZE = 5

    def get_direction(
        self,
        numerical_heatmap: np.ndarray,
        came_from: np.ndarray | None = None,
        coverage_directions=None,
    ) -> np.ndarray:
        """Return a unit vector toward the patch with the most total heat."""
        numerical_heatmap = np.asarray(numerical_heatmap, dtype=np.float32)
        if numerical_heatmap.ndim != 2:
            raise ValueError("numerical_heatmap must be a two-dimensional array.")
        if numerical_heatmap.size == 0:
            return np.zeros(2, dtype=np.float32)

        numerical_heatmap = np.maximum(
            np.nan_to_num(numerical_heatmap),
            0,
        )
        forbidden_directions = []
        if came_from is not None:
            forbidden_directions.append(came_from)
        if coverage_directions is not None:
            forbidden_directions.extend(coverage_directions)

        height, width = numerical_heatmap.shape
        row_edges = self._patch_edges(height)
        column_edges = self._patch_edges(width)
        row_count = len(row_edges) - 1
        column_count = len(column_edges) - 1
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        patches = []
        for row_index, (top, bottom) in enumerate(
            zip(row_edges[:-1], row_edges[1:])
        ):
            for column_index, (left, right) in enumerate(
                zip(column_edges[:-1], column_edges[1:])
            ):
                target_x = (column_index + 0.5) * width / column_count - 0.5
                target_y = (row_index + 0.5) * height / row_count - 0.5
                patch_direction = np.array(
                    [
                        target_x - center_x,
                        center_y - target_y,
                    ],
                    dtype=np.float32,
                )
                magnitude = np.linalg.norm(patch_direction)
                if magnitude == 0:
                    continue
                patch_direction = patch_direction / magnitude

                patch_heat = float(
                    numerical_heatmap[top:bottom, left:right].sum(
                        dtype=np.float64,
                    )
                )
                patches.append(
                    (
                        patch_heat,
                        patch_direction,
                        self._is_forbidden(
                            patch_direction,
                            forbidden_directions,
                        ),
                    )
                )

        if not patches:
            return np.array([0.0, 1.0], dtype=np.float32)

        available_patches = [patch for patch in patches if not patch[2]]
        candidate_patches = available_patches or patches
        _, direction, _ = max(
            candidate_patches,
            key=lambda patch: (
                patch[0],
                patch[1][1],
                -abs(patch[1][0]),
            ),
        )
        return direction

    def _patch_edges(self, length):
        patch_count = min(self.GRID_SIZE, length)
        return np.rint(
            np.linspace(0, length, patch_count + 1),
        ).astype(np.int32)

    def _is_forbidden(self, direction, forbidden_directions):
        direction_magnitude = np.linalg.norm(direction)
        if direction_magnitude == 0:
            return False

        direction = direction / direction_magnitude
        cone_cosine = np.cos(np.deg2rad(self.EXCLUSION_ANGLE_DEGREES))
        for vector in forbidden_directions:
            vector = np.asarray(vector, dtype=np.float32)
            if vector.shape != (2,) or not np.all(np.isfinite(vector)):
                continue
            magnitude = np.linalg.norm(vector)
            if magnitude == 0:
                continue
            vector = vector / magnitude
            if np.dot(direction, vector) >= cone_cosine:
                return True
        return False
