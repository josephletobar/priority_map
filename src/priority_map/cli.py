import argparse
import time

from priority_map.api import run_priority_map
from priority_map.config import params as config


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_folder",
        nargs="?",
        help="Folder of images to process.",
    )
    parser.add_argument(
        "--dataset-root",
        help="Dataset root. Supports plain image folder. Defaults to the runner debug path.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Folder where generated priority map outputs are written. "
            "Defaults to examples/YYYY-MM-DD_HH-MM-SS."
        ),
    )
    parser.add_argument("--task", default="Find cars")
    parser.add_argument("--debrief")
    parser.add_argument("--mask", nargs="*", default=[])
    parser.add_argument("--sam-step", type=int, default=config.SAM_STEP)
    parser.add_argument("--sam-thresh", type=float, default=config.SAM_TRESH)
    parser.add_argument("--blur-spread", type=float, default=config.BLUR_SPREAD)
    parser.add_argument(
        "--max-image-edge",
        type=int,
        default=config.MAX_IMAGE_EDGE,
        help="Resize input images so the longest edge is at most this many pixels. Use 0 to disable.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--panoramic", action="store_true")
    parser.add_argument(
        "--graph_agent",
        "--graph-agent",
        action="store_true",
        help="Enable the asynchronous graph agent. Off by default.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    t0 = time.perf_counter()
    args = parse_args(argv)
    image_folder = args.image_folder or args.dataset_root

    result = run_priority_map(
        image_folder=image_folder,
        output_dir=args.output_dir,
        task=args.task,
        debrief=args.debrief,
        mask=args.mask,
        sam_step=args.sam_step,
        sam_thresh=args.sam_thresh,
        blur_spread=args.blur_spread,
        max_image_edge=args.max_image_edge,
        show=args.show,
        panoramic=args.panoramic,
        graph_agent=args.graph_agent,
    )

    print(f"Output written to: {result.output_dir}")
    print(f"Frames processed: {result.frames_processed}")
    print(f"Total time: {(time.perf_counter() - t0):.2f} seconds")
    return result
