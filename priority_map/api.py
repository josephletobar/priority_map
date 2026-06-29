from pathlib import Path

import cv2

from priority_map.runner import PriorityMapResult, PriorityMapRunner


def run_priority_map(
    image_folder: str | Path | None = None,
    output_dir: str | Path | None = None,
    task: str = "Find cars",
    debrief: str = "debrief.txt",
    mask: list[str] | None = None,
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
