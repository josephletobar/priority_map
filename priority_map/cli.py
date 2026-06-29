import argparse
import time

from priority_map.api import run_priority_map


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
    parser.add_argument("--debrief", default="debrief.txt")
    parser.add_argument("--mask", nargs="*", default=[])
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
        show=args.show,
        panoramic=args.panoramic,
        graph_agent=args.graph_agent,
    )

    print(f"Output written to: {result.output_dir}")
    print(f"Frames processed: {result.frames_processed}")
    print(f"Total time: {(time.perf_counter() - t0):.2f} seconds")
    return result
