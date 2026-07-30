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
        "--img-folder",
        dest="img_folder",
        help="Folder of images to process. Overrides the positional image_folder.",
    )
    parser.add_argument(
        "--gps",
        "--gps-csv",
        dest="gps_csv",
        help="Optional per-frame GPS/pose CSV with a name column matching image filenames.",
    )
    parser.add_argument(
        "--camera-intrinsics",
        help="Optional camera intrinsics path. Stored on the runner but not used yet.",
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
        "--dilation-scale",
        type=float,
        default=config.DILATION_SCALE,
        help="Scale factor for heatmap dilation intensity (1.0 preserves default behavior).",
    )
    parser.add_argument("--sam-model-path", default=config.SAM_MODEL_PATH)
    parser.add_argument(
        "--scene-model",
        help="Scene VLM model. Use 'gemma' for default OpenRouter Gemma, or an OpenAI model like 'gpt-5.4'.",
    )
    parser.add_argument(
        "--max-image-edge",
        type=int,
        default=config.MAX_IMAGE_EDGE,
        help="Resize input images so the longest edge is at most this many pixels. Use 0 to disable.",
    )
    parser.add_argument(
        "--exclusion-angle-degrees",
        type=float,
        default=config.EXCLUSION_ANGLE_DEGREES,
        help="Half-angle around each forbidden steering bearing.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--panoramic", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    t0 = time.perf_counter()
    args = parse_args(argv)
    image_folder = args.img_folder or args.image_folder

    result = run_priority_map(
        image_folder=image_folder,
        output_dir=args.output_dir,
        task=args.task,
        debrief=args.debrief,
        mask=args.mask,
        sam_step=args.sam_step,
        sam_thresh=args.sam_thresh,
        blur_spread=args.blur_spread,
        dilation_scale=args.dilation_scale,
        max_image_edge=args.max_image_edge,
        sam_model_path=args.sam_model_path,
        debug=args.debug,
        panoramic=args.panoramic,
        gps_csv=args.gps_csv,
        camera_intrinsics=args.camera_intrinsics,
        scene_model=args.scene_model,
        exclusion_angle_degrees=args.exclusion_angle_degrees,
    )

    if args.debug:
        print(f"Output written to: {result.output_dir}")
        print(f"Frames processed: {result.frames_processed}")
        print(f"Total time: {(time.perf_counter() - t0):.2f} seconds")
    return result
