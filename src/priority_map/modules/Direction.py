import numpy as np


def get_direction(numerical_heatmap: np.ndarray) -> np.ndarray:
    """Return a unit vector from the heatmap center to its hottest pixel."""
    numerical_heatmap = np.asarray(numerical_heatmap)
    if numerical_heatmap.ndim != 2:
        raise ValueError("numerical_heatmap must be a two-dimensional array.")
    if numerical_heatmap.size == 0 or np.max(numerical_heatmap) <= 0:
        return np.zeros(2, dtype=np.float32)

    hottest_y, hottest_x = np.unravel_index(
        np.argmax(numerical_heatmap),
        numerical_heatmap.shape,
    )
    center_y = numerical_heatmap.shape[0] // 2
    center_x = numerical_heatmap.shape[1] // 2

    direction = np.array(
        [
            hottest_x - center_x,
            center_y - hottest_y,
        ],
        dtype=np.float32,
    )
    magnitude = np.linalg.norm(direction)
    if magnitude == 0:
        return np.zeros(2, dtype=np.float32)

    return direction / magnitude
