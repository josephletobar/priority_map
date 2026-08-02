import argparse
import time

from priority_map.api import run_priority_map
from priority_map.config import params as config


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img-folder",
        dest="img_folder",
        required=True,
        metavar="PATH",
        help="Folder of images to process.",
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
    parser.add_argument(
        "--sam-model-path",
        required=True,
        metavar="PATH",
        help="Path to the SAM model weights.",
    )
    parser.add_argument(
        "--scene-model",
        required=True,
        metavar="PROVIDER:MODEL",
        help=(
            "Scene VLM provider and model. Supported providers are "
            "openai, openrouter, and ollama."
        ),
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
        help="Half-angle around the came-from steering bearing.",
    )
    parser.add_argument(
        "--direction-ema-alpha-min",
        type=float,
        default=config.DIRECTION_EMA_ALPHA_MIN,
        help="Direction EMA alpha used for uniform heatmaps.",
    )
    parser.add_argument(
        "--direction-ema-alpha-max",
        type=float,
        default=config.DIRECTION_EMA_ALPHA_MAX,
        help="Direction EMA alpha used for highly varied heatmaps.",
    )
    parser.add_argument(
        "--max-observed-coverage-ratio",
        type=float,
        default=config.MAX_OBSERVED_COVERAGE_RATIO,
        help="Maximum observed-region share allowed in a predicted camera view.",
    )
    parser.add_argument(
        "--coverage-lookahead-seconds",
        type=float,
        default=config.COVERAGE_LOOKAHEAD_SECONDS,
        help="Seconds to project each candidate direction forward.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--panoramic", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    t0 = time.perf_counter()
    args = parse_args(argv)

    result = run_priority_map(
        image_folder=args.img_folder,
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
        direction_ema_alpha_min=args.direction_ema_alpha_min,
        direction_ema_alpha_max=args.direction_ema_alpha_max,
        exclusion_angle_degrees=args.exclusion_angle_degrees,
        max_observed_coverage_ratio=args.max_observed_coverage_ratio,
        coverage_lookahead_seconds=args.coverage_lookahead_seconds,
    )

    if args.debug:
        print(f"Output written to: {result.output_dir}")
        print(f"Frames processed: {result.frames_processed}")
        print(f"Total time: {(time.perf_counter() - t0):.2f} seconds")
    return result
