import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from dataclasses import dataclass
import csv
import time

import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from priority_map.config import params as config
from priority_map.scripts.video_helper import (
    label_mask,
    safe_imwrite,
    VideoOutput,
)
from priority_map.scripts.cluster_segmentations import cluster_segmentations

from priority_map.modules.SceneUnderstanding import SceneUnderstanding
from priority_map.modules.Heatmap import Heatmap
from priority_map.modules.Segment import Segment
from priority_map.modules.GraphBuilder import GraphBuilder
from priority_map.modules.GraphAgent import GraphAgent
from priority_map.modules.GeoLocalizer import GeoLocalizer
from priority_map.modules.PanoramaBuilder import PanoramaBuilder

load_dotenv()


# DEFAULT_IMAGE_FOLDER = Path(r"D:\UAV_VisLoc_dataset\05\drone")
# DEFAULT_IMAGE_FOLDER = Path(r"D:\Train\Train\query_images")
DEFAULT_IMAGE_FOLDER = Path(r"D:\dronevid2")
# DEFAULT_IMAGE_FOLDER = Path(r"D:\rtereg") 


def default_output_dir() -> Path:
    return Path("examples") / time.strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class PriorityMapResult:
    output_dir: Path
    observations_csv: Path
    video_path: Path | None
    heatmap_video_path: Path | None
    frames_processed: int


@dataclass
class PriorityFrameResult:
    image_name: str | None
    image_path: str | None
    frame_index: int | None
    heatmap_only: np.ndarray | None
    output_frame: np.ndarray | None
    latency_seconds: dict[str, float]
    keep_running: bool

    def __bool__(self):
        return self.keep_running


class PriorityMapRunner:
    def __init__(
        self,
        image_folder: str | Path | None = None,
        output_dir: str | Path | None = None,
        task="Find cars",
        debrief: str | None = None,
        mask=None,
        sam_step=config.SAM_STEP,
        sam_thresh=config.SAM_TRESH,
        blur_spread=config.BLUR_SPREAD,
        max_image_edge=config.MAX_IMAGE_EDGE,
        sam_model_path=config.SAM_MODEL_PATH,
        debug=False,
        record=True,
        panoramic=False,
        graph_agent=False,
    ):
        self.dataset_root = Path(image_folder) if image_folder is not None else DEFAULT_IMAGE_FOLDER
        self.task = task
        self.debrief = debrief
        self.task_description = task if not debrief else f"{task}: {debrief}"
        self.masks = mask or []
        self.sam_step = sam_step
        self.sam_thresh = sam_thresh
        self.blur_spread = blur_spread
        self.max_image_edge = max_image_edge
        self.sam_model_path = sam_model_path
        self.panoramic = panoramic
        self.debug = debug
        
        self.query_csv, self.query_images_dir = self._load_dataset_index()

        self.index = 0
        self.frames_processed = 0
        self.output_dir = Path(output_dir) if output_dir is not None else default_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.observations_csv = self.output_dir / "observations.csv"

        self.scene_understanding = SceneUnderstanding(debug=debug)
        self.segmentation = Segment(
            debug=debug,
            sam_thresh=sam_thresh,
            sam_model_path=sam_model_path,
        )
        self.graph_builder = GraphBuilder(output_dir=self.output_dir, debug=debug)
        self.graph_agent = (
            GraphAgent(self.graph_builder, self.task_description, debug=debug)
            if graph_agent
            else None
        )
        self.heatmap = Heatmap(blur_spread=blur_spread)
        self.geo_localizer = GeoLocalizer()
        self.masks = {m.lower() for m in self.masks}
        self.video_output = VideoOutput(
            output_dir=self.output_dir,
            show=debug,
            record=record,
            filename="video.avi",
            window_name="Drone Heatmap",
            debug=debug,
        )
        self.heatmap_video_output = VideoOutput(
            output_dir=self.output_dir,
            show=False,
            record=record,
            filename="heatmap.avi",
            window_name="Heatmap Only",
            debug=debug,
        )

        self.heat_panoramic_builder = PanoramaBuilder(alpha=0.9)
        self.panoramic_builder = PanoramaBuilder(alpha=0.15)
        self.last_graph_frame = None
        self._closed = False

    def _load_dataset_index(self):
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

        old_query_csv = self.dataset_root / "query.csv"
        old_query_images_dir = self.dataset_root / "query_images"

        if old_query_csv.exists():
            query_csv = pd.read_csv(old_query_csv)
            if "name" not in query_csv.columns:
                raise ValueError(f"{old_query_csv} must contain a 'name' column.")
            if "altitude" not in query_csv.columns:
                raise ValueError(f"{old_query_csv} must contain an 'altitude' column.")
            return query_csv, old_query_images_dir

        image_files = [
            path
            for path in sorted(self.dataset_root.iterdir())
            if path.is_file() and path.suffix.lower() in image_extensions
        ]
        if image_files:
            query_csv = pd.DataFrame({"name": [path.name for path in image_files]})
            return query_csv, self.dataset_root

        csv_files = sorted(self.dataset_root.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No images or query.csv file found in {self.dataset_root}"
            )

        raise FileNotFoundError(
            f"Found csv file(s), but only query.csv is supported in {self.dataset_root}"
        )

    def _row_value(self, row, *columns, default=0.0):
        for column in columns:
            if column in row and pd.notna(row[column]):
                return row[column]
        return default

    def has_next(self) -> bool:
        return self.index < len(self.query_csv)

    def reset(self):
        self.index = 0

    def should_run_sam(self, frame):
        return frame["frame_index"] % self.sam_step == 0

    def _resize_input_image(self, image):
        if not self.max_image_edge:
            return image

        max_edge = int(self.max_image_edge)
        if max_edge <= 0:
            return image

        height, width = image.shape[:2]
        longest_edge = max(height, width)
        if longest_edge <= max_edge:
            return image

        scale = max_edge / longest_edge
        resized_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)

    def get_next_frame(self):
        while self.has_next():
            frame_index = self.index
            row = self.query_csv.iloc[frame_index]
            self.index += 1

            self.image_name = self._row_value(row, "name", "filename", default="")
            image_path = (self.query_images_dir / str(self.image_name))
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            image = self._resize_input_image(image)

            image[:, :, 1] = (image[:, :, 1] * 0.65).astype(image.dtype)
            image[:, :, 2:3] = (image[:, :, 2:3] * 0.8).astype(image.dtype)

            return {
                "image": image,
                "image_path": str(image_path),
                # Position metadata is intentionally disabled for plain image-folder runs.
                "easting": self._row_value(row, "easting"),
                "northing": self._row_value(row, "northing"),
                "altitude": self._row_value(row, "altitude"),
                "orientation": None,
                "frame_index": frame_index,
            }

        return None
    
    def close(self):
        if self._closed:
            return

        self._closed = True
        errors = []

        # Release video writers/windows before slower shutdown work so Ctrl+C
        # cannot leave the output file waiting on unrelated cleanup.
        cleanup_steps = [
            ("main video output", self.video_output.close),
            ("heatmap video output", self.heatmap_video_output.close),
            ("SAM preview output", self.segmentation.close),
            ("graph builder", self.graph_builder.close),
        ]
        if self.graph_agent is not None:
            cleanup_steps.append(("graph agent", self.graph_agent.close))

        for name, close in cleanup_steps:
            try:
                close()
            except Exception as exc:
                errors.append((name, exc))

        for name, exc in errors:
            print(f"Cleanup warning ({name}): {exc}")

    def run_frame(self):
        if self.graph_agent is not None:
            self.graph_agent.poll_finished()

        frame_t0 = time.perf_counter()
        frame = self.get_next_frame()
        if frame is None:
            total_seconds = time.perf_counter() - frame_t0
            return PriorityFrameResult(
                image_name=None,
                image_path=None,
                frame_index=None,
                heatmap_only=None,
                output_frame=None,
                latency_seconds={
                    "total": total_seconds,
                    "vlm": 0.0,
                    "sam3": 0.0,
                    "other": total_seconds,
                },
                keep_running=False,
            )

        image = frame["image"]
        out = image
        vlm_seconds = 0.0
        sam3_seconds = 0.0

        # Position/geolocation is disabled for now so a plain folder of images just works.
        # position = (
        #     frame["easting"],
        #     frame["northing"],
        #     frame["altitude"]
        # )

        scene_dict = None
        scene_edge_intents = []
        recent_graph_context = {"nodes": [], "edges": [], "model_edges": []}
        if self.should_run_sam(frame):
            vlm_t0 = time.perf_counter()
            recent_graph_context = self.graph_builder.get_recent_graph_context(limit=10)
            scene_result = self.scene_understanding.get_labels(
                image,
                self.task_description,
                recent_graph_context=recent_graph_context,
            )
            if scene_result is not None:
                scene_dict = scene_result.labels
                scene_edge_intents = scene_result.edge_intents
            vlm_seconds = time.perf_counter() - vlm_t0
        # print(scene_dict)

        # Get Segmentations From Image
        segmentation_result = self.segmentation.get_segmentations(image, scene_dict)
        segmentations = segmentation_result.segmentations
        sam3_seconds = segmentation_result.sam3_seconds
        if segmentations is None:
            segmentations = []

        clustered = cluster_segmentations(segmentations)

        # Set Where to Save Observations CSV
        csv_path = self.observations_csv
        if not csv_path.exists():
            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image_name", "label", "centroid_x", "centroid_y"])

        # Save Clustered Observations to CSV
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            for cluster in clustered:
                cx, cy = cluster.centroid
                writer.writerow([
                    self.image_name,
                    cluster.label,
                    cx,
                    cy,
                ])

        # Set Segmentation Types to Display MASKS
        mask_frame = None
        if self.masks:
            mask_frame = label_mask(self.masks, image, segmentations)

        # Create Heatmap
        heatmap_text, heatmap_only = self.heatmap.draw_heatmap(image, clustered)
        if heatmap_text is not None and heatmap_only is not None:
            out = heatmap_text
            self.heatmap_video_output.handle_frame(
                heatmap_text,
                header=f"Task: {self.task}",
            )

        # Save Heatmap Individual Heatmap Images
        heatmap_images_dir = self.output_dir / "heatmap_imgs"
        heatmap_images_dir.mkdir(parents=True, exist_ok=True)
        safe_imwrite(str(heatmap_images_dir / str(self.image_name)), heatmap_only)

        if scene_dict is not None:
            add_result = self.graph_builder.add_nodes(clustered)
            self.graph_builder.resolve_scene_edge_intents(
                scene_edge_intents,
                add_result,
                recent_graph_context,
            )
            graph_frame = self.graph_builder.render_2d_graph_frame()
            if graph_frame is not None:
                self.last_graph_frame = graph_frame

            if self.graph_agent is not None:
                self.graph_agent.start_async_if_ready()

        if self.last_graph_frame is not None and out is not None:
            heatmap_height = out.shape[0]
            graph_resized = cv2.resize(self.last_graph_frame, (int(self.last_graph_frame.shape[1] * heatmap_height / self.last_graph_frame.shape[0]), heatmap_height))
            out = cv2.hconcat([out, graph_resized])

        # Create Panoramic Images (If Enabled, Experimental)
        if self.panoramic:
            panorama_save = 10
            heat_panorama_dir = self.output_dir / "heat_panorama"
            panorama_dir = self.output_dir / "panorama"
            heat_panorama_dir.mkdir(parents=True, exist_ok=True)
            panorama_dir.mkdir(parents=True, exist_ok=True)
            transform = self.segmentation.transform_dx, self.segmentation.transform_dy

            # Base image panorama
            panorama = self.panoramic_builder.create_panorama(transform, image)
            if self.index % panorama_save == 0:
                safe_imwrite(str(panorama_dir / f"panorama_{self.index}.png"), panorama)

            # Heatmap overlay panorama
            heat_panorama = self.heat_panoramic_builder.create_panorama(transform, heatmap_only)
            if heat_panorama is not None:
                heat_panorama = cv2.GaussianBlur(heat_panorama, (21, 401), 0)
                heat_panorama = cv2.addWeighted(
                    panorama,    # base panorama
                    0.6,      # weight of base image
                    heat_panorama,  # heatmap color overlay
                    0.4,      # weight of heatmap
                    0         # constant brightness offset added to every pixel
                )
                if self.index % panorama_save == 0:
                    safe_imwrite(str(heat_panorama_dir / f"heat_panorama_{self.index}.png"), heat_panorama)


        # Write Output Frame To Video

        keep_running = self.video_output.handle_frame(
            out,
            header=f"Task: {self.task}",
            side_image=mask_frame,
            side_header=f"Mask(s): {', '.join(sorted(self.masks)) or 'None'}",
        )
        self.frames_processed += 1
        total_seconds = time.perf_counter() - frame_t0
        return PriorityFrameResult(
            image_name=str(self.image_name),
            image_path=frame["image_path"],
            frame_index=frame["frame_index"],
            heatmap_only=heatmap_only,
            output_frame=out,
            latency_seconds={
                "total": total_seconds,
                "vlm": vlm_seconds,
                "sam3": sam3_seconds,
                "other": max(total_seconds - vlm_seconds - sam3_seconds, 0.0),
            },
            keep_running=keep_running,
        )

    def result(self):
        return PriorityMapResult(
            output_dir=self.output_dir,
            observations_csv=self.observations_csv,
            video_path=self.video_output.video_path,
            heatmap_video_path=self.heatmap_video_output.video_path,
            frames_processed=self.frames_processed,
        )

    def run(self):
        while self.has_next():
            frame_result = self.run_frame()
            if not frame_result.keep_running:
                break
        return self.result()
