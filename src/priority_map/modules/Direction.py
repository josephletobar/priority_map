from dataclasses import dataclass

import numpy as np

from priority_map.config import params as config


@dataclass
class DirectionDecision:
    direction: np.ndarray
    patch_heat: float


class Direction:
    EXCLUSION_ANGLE_DEGREES = config.EXCLUSION_ANGLE_DEGREES
    GRID_SIZE = 5

    def __init__(
        self,
        exclusion_angle_degrees: float = EXCLUSION_ANGLE_DEGREES,
    ):
        if not 0 < exclusion_angle_degrees <= 180:
            raise ValueError(
                "exclusion_angle_degrees must be greater than 0 "
                "and at most 180."
            )
        self.exclusion_angle_degrees = float(exclusion_angle_degrees)

    def get_direction(
        self,
        numerical_heatmap: np.ndarray,
        came_from: np.ndarray | None = None,
        candidate_validator=None,
    ) -> np.ndarray:
        return self.get_decision(
            numerical_heatmap,
            came_from=came_from,
            candidate_validator=candidate_validator,
        ).direction

    def get_decision(
        self,
        numerical_heatmap: np.ndarray,
        came_from: np.ndarray | None = None,
        candidate_validator=None,
    ) -> DirectionDecision:
        """Choose the hottest patch outside the came-from cone."""
        numerical_heatmap = np.asarray(numerical_heatmap, dtype=np.float32)
        if numerical_heatmap.ndim != 2:
            raise ValueError("numerical_heatmap must be a two-dimensional array.")
        if numerical_heatmap.size == 0:
            return self._hold_decision()

        numerical_heatmap = np.maximum(
            np.nan_to_num(numerical_heatmap),
            0,
        )
        if not np.any(numerical_heatmap):
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
                    )
                )

        if not patches:
            return self._hold_decision()

        non_backtracking = [
            patch
            for patch in patches
            if not self.is_blocked_by_came_from(patch[1], came_from)
        ]
        if not non_backtracking:
            return self._hold_decision()

        feasible = [
            patch
            for patch in non_backtracking
            if candidate_validator is None
            or candidate_validator(patch[1])
        ]
        candidate_patches = feasible or non_backtracking

        patch_heat, direction = max(
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
        )

    def _hold_decision(self):
        return DirectionDecision(
            direction=np.zeros(2, dtype=np.float32),
            patch_heat=0.0,
        )

    def _patch_edges(self, length):
        patch_count = min(self.GRID_SIZE, length)
        return np.rint(
            np.linspace(0, length, patch_count + 1),
        ).astype(np.int32)

    def is_blocked_by_came_from(self, direction, came_from):
        if came_from is None:
            return False

        direction = np.asarray(direction, dtype=np.float32)
        came_from = np.asarray(came_from, dtype=np.float32)
        if (
            direction.shape != (2,)
            or came_from.shape != (2,)
            or not np.all(np.isfinite(direction))
            or not np.all(np.isfinite(came_from))
        ):
            return False

        direction_magnitude = np.linalg.norm(direction)
        came_from_magnitude = np.linalg.norm(came_from)
        if direction_magnitude == 0 or came_from_magnitude == 0:
            return False

        cosine = np.dot(
            direction / direction_magnitude,
            came_from / came_from_magnitude,
        )
        cone_cosine = np.cos(np.deg2rad(self.exclusion_angle_degrees))
        return bool(cosine >= cone_cosine)
