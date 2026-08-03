from pathlib import Path

import cv2

from priority_map.config import params as config
from priority_map.runner import PriorityMapResult, PriorityMapRunner
from priority_map.modules.GraphAgent import ask_priority_map_db


def run_priority_map(
    image_folder: str | Path,
    output_dir: str | Path | None = None,
    task: str = "Find cars",
    debrief: str | None = None,
    mask: list[str] | None = None,
    sam_step: int = config.SAM_STEP,
    sam_thresh: float = config.SAM_TRESH,
    blur_spread: float = config.BLUR_SPREAD,
    dilation_scale: float = config.DILATION_SCALE,
    max_image_edge: int | None = config.MAX_IMAGE_EDGE,
    *,
    sam_model_path: str | Path,
    debug: bool = False,
    record: bool = True,
    panoramic: bool = False,
    gps_csv: str | Path | None = None,
    camera_intrinsics: str | Path | None = None,
    scene_model: str | None = None,
    vector_ema_alpha: float = config.VECTOR_EMA_ALPHA,
    direction_ema_alpha_min: float = config.DIRECTION_EMA_ALPHA_MIN,
    direction_ema_alpha_max: float = config.DIRECTION_EMA_ALPHA_MAX,
    exclusion_angle_degrees: float = config.EXCLUSION_ANGLE_DEGREES,
    max_observed_coverage_ratio: float = config.MAX_OBSERVED_COVERAGE_RATIO,
    coverage_lookahead_seconds: float = config.COVERAGE_LOOKAHEAD_SECONDS,
) -> PriorityMapResult:
    runner = PriorityMapRunner(
        image_folder=image_folder,
        output_dir=output_dir,
        task=task,
        debrief=debrief,
        mask=mask,
        sam_step=sam_step,
        sam_thresh=sam_thresh,
        blur_spread=blur_spread,
        dilation_scale=dilation_scale,
        max_image_edge=max_image_edge,
        sam_model_path=sam_model_path,
        debug=debug,
        record=record,
        panoramic=panoramic,
        gps_csv=gps_csv,
        camera_intrinsics=camera_intrinsics,
        scene_model=scene_model,
        vector_ema_alpha=vector_ema_alpha,
        direction_ema_alpha_min=direction_ema_alpha_min,
        direction_ema_alpha_max=direction_ema_alpha_max,
        exclusion_angle_degrees=exclusion_angle_degrees,
        max_observed_coverage_ratio=max_observed_coverage_ratio,
        coverage_lookahead_seconds=coverage_lookahead_seconds,
    )

    try:
        return runner.run()
    finally:
        runner.close()
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error:
            pass
