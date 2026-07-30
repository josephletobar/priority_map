from pathlib import Path

import cv2

from priority_map.config import params as config
from priority_map.runner import PriorityMapResult, PriorityMapRunner
from priority_map.modules.GraphAgent import review_priority_map_db


def run_priority_map(
    image_folder: str | Path | None = None,
    output_dir: str | Path | None = None,
    task: str = "Find cars",
    debrief: str | None = None,
    mask: list[str] | None = None,
    sam_step: int = config.SAM_STEP,
    sam_thresh: float = config.SAM_TRESH,
    blur_spread: float = config.BLUR_SPREAD,
    dilation_scale: float = config.DILATION_SCALE,
    max_image_edge: int | None = config.MAX_IMAGE_EDGE,
    sam_model_path: str | Path = config.SAM_MODEL_PATH,
    debug: bool = False,
    record: bool = True,
    panoramic: bool = False,
    gps_csv: str | Path | None = None,
    camera_intrinsics: str | Path | None = None,
    scene_model: str | None = None,
    vector_ema_alpha: float = config.VECTOR_EMA_ALPHA,
    max_direction_turn_degrees: float = config.MAX_DIRECTION_TURN_DEGREES,
    exclusion_angle_degrees: float = config.EXCLUSION_ANGLE_DEGREES,
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
        max_direction_turn_degrees=max_direction_turn_degrees,
        exclusion_angle_degrees=exclusion_angle_degrees,
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
