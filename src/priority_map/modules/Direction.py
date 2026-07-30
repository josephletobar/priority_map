from dataclasses import dataclass

import numpy as np


@dataclass
class DirectionDecision:
    direction: np.ndarray
    patch_heat: float
    coverage_count: int


class Direction:
    EXCLUSION_ANGLE_DEGREES = 60.0
    GRID_SIZE = 5

    def get_direction(
        self,
        numerical_heatmap: np.ndarray,
        came_from: np.ndarray | None = None,
        coverage_directions=None,
    ) -> np.ndarray:
        return self.get_decision(
            numerical_heatmap,
            came_from=came_from,
            coverage_directions=coverage_directions,
        ).direction

    def get_decision(
        self,
        numerical_heatmap: np.ndarray,
        came_from: np.ndarray | None = None,
        coverage_directions=None,
    ) -> DirectionDecision:
        """Choose a non-backtracking patch using heat and graph coverage."""
        numerical_heatmap = np.asarray(numerical_heatmap, dtype=np.float32)
        if numerical_heatmap.ndim != 2:
            raise ValueError("numerical_heatmap must be a two-dimensional array.")
        if numerical_heatmap.size == 0:
            return self._hold_decision()

        numerical_heatmap = np.maximum(
            np.nan_to_num(numerical_heatmap),
            0,
        )
        coverage_directions = list(coverage_directions or [])
        if not np.any(numerical_heatmap) and not coverage_directions:
            return self._hold_decision()

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
                        self.is_blocked_by_came_from(
                            patch_direction,
                            came_from,
                        ),
                        self.coverage_count(
                            patch_direction,
                            coverage_directions,
                        ),
                    )
                )

        if not patches:
            return self._hold_decision()

        non_backtracking = [patch for patch in patches if not patch[2]]
        if not non_backtracking:
            return self._hold_decision()

        uncovered = [
            patch
            for patch in non_backtracking
            if patch[3] == 0
        ]
        if uncovered:
            candidate_patches = uncovered
        else:
            minimum_coverage = min(patch[3] for patch in non_backtracking)
            candidate_patches = [
                patch
                for patch in non_backtracking
                if patch[3] == minimum_coverage
            ]

        patch_heat, direction, _, selected_coverage = max(
            candidate_patches,
            key=lambda patch: (
                patch[0],
                patch[1][1],
                -abs(patch[1][0]),
            ),
        )
        return DirectionDecision(
            direction=np.asarray(direction, dtype=np.float32),
            patch_heat=patch_heat,
            coverage_count=selected_coverage,
        )

    def _hold_decision(self):
        return DirectionDecision(
            direction=np.zeros(2, dtype=np.float32),
            patch_heat=0.0,
            coverage_count=0,
        )

    def _patch_edges(self, length):
        patch_count = min(self.GRID_SIZE, length)
        return np.rint(
            np.linspace(0, length, patch_count + 1),
        ).astype(np.int32)

    def is_blocked_by_came_from(self, direction, came_from):
        if came_from is None:
            return False
        return self._is_forbidden(direction, [came_from])

    def coverage_count(self, direction, coverage_directions):
        direction_magnitude = np.linalg.norm(direction)
        if direction_magnitude == 0:
            return 0

        direction = direction / direction_magnitude
        cone_cosine = np.cos(np.deg2rad(self.EXCLUSION_ANGLE_DEGREES))
        count = 0
        for vector in coverage_directions or []:
            vector = np.asarray(vector, dtype=np.float32)
            if vector.shape != (2,) or not np.all(np.isfinite(vector)):
                continue
            magnitude = np.linalg.norm(vector)
            if magnitude == 0:
                continue
            vector = vector / magnitude
            if np.dot(direction, vector) >= cone_cosine:
                count += 1
        return count

    def _is_forbidden(self, direction, forbidden_directions):
        return self.coverage_count(direction, forbidden_directions) > 0
