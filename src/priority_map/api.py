from pathlib import Path

import cv2

from priority_map.config import params as config
from priority_map.runner import PriorityMapResult, PriorityMapRunner


def run_priority_map(
    image_folder: str | Path | None = None,
    output_dir: str | Path | None = None,
    task: str = "Find cars",
    debrief: str | None = None,
    mask: list[str] | None = None,
    sam_step: int = config.SAM_STEP,
    sam_thresh: float = config.SAM_TRESH,
    blur_spread: float = config.BLUR_SPREAD,
    max_image_edge: int | None = config.MAX_IMAGE_EDGE,
    show: bool = False,
    record: bool = True,
    panoramic: bool = False,
    graph_agent: bool = False,
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
        max_image_edge=max_image_edge,
        show=show,
        record=record,
        panoramic=panoramic,
        graph_agent=graph_agent,
    )

    try:
        return runner.run()
    finally:
        runner.close()
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error as exc:
            print(f"Cleanup warning (OpenCV windows): {exc}")
