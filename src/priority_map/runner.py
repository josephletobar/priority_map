import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from dataclasses import dataclass
import csv
import time

import cv2
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


DEFAULT_IMAGE_FOLDER = Path(r"D:\UAV_VisLoc_dataset\05\drone")
# DEFAULT_IMAGE_FOLDER = Path(r"D:\Train\Train\query_images")
# DEFAULT_IMAGE_FOLDER = Path(r"C:\Users\jletobar3\Projects\dronevid2")


def default_image_folder() -> Path:
    return DEFAULT_IMAGE_FOLDER


def default_output_dir() -> Path:
    return Path("examples") / time.strftime("%Y-%m-%d_%H-%M-%S")


@dataclass
class PriorityMapResult:
    output_dir: Path
    observations_csv: Path
    video_path: Path | None
    heatmap_video_path: Path | None
    frames_processed: int


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
        show=False,
        record=True,
        panoramic=False,
        graph_agent=False,
    ):
        self.dataset_root = Path(image_folder) if image_folder is not None else default_image_folder()
        self.task = task
        self.debrief = debrief
        self.task_description = task if not debrief else f"{task}: {debrief}"
        self.masks = mask or []
        self.sam_step = sam_step
        self.sam_thresh = sam_thresh
        self.blur_spread = blur_spread
        self.panoramic = panoramic
        
        self.query_csv, self.query_images_dir = self._load_dataset_index()

        self.index = 0
        self.frames_processed = 0
        self.output_dir = Path(output_dir) if output_dir is not None else default_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.observations_csv = self.output_dir / "observations.csv"

        self.scene_understanding = SceneUnderstanding()
        self.segmentation = Segment(show_preview=show, sam_thresh=sam_thresh)
        self.graph_builder = GraphBuilder(output_dir=self.output_dir)
        self.graph_agent = (
            GraphAgent(self.graph_builder, self.task_description)
            if graph_agent
            else None
        )
        self.heatmap = Heatmap(blur_spread=blur_spread)
        self.geo_localizer = GeoLocalizer()
        self.masks = {m.lower() for m in self.masks}
        self.video_output = VideoOutput(
            output_dir=self.output_dir,
            show=show,
            record=record,
            filename="video.avi",
            window_name="Drone Heatmap",
        )
        self.heatmap_video_output = VideoOutput(
            output_dir=self.output_dir,
            show=False,
            record=record,
            filename="heatmap.avi",
            window_name="Heatmap Only",
        )

        self.heat_panoramic_builder = PanoramaBuilder(alpha=0.9)
        self.panoramic_builder = PanoramaBuilder(alpha=0.15)
        self.show = show
        self.last_graph_frame = None
        self.graph_view = "semantic"
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

        frame = self.get_next_frame()
        if frame is None:
            return False

        image = frame["image"]
        out = image

        # Position/geolocation is disabled for now so a plain folder of images just works.
        # position = (
        #     frame["easting"],
        #     frame["northing"],
        #     frame["altitude"]
        # )

        scene_dict = None
        if self.should_run_sam(frame):
            scene_dict = self.scene_understanding.get_labels(image, self.task_description)
        # print(scene_dict)

        # Get Segmentations From Image
        segmentations = self.segmentation.get_segmentations(image, scene_dict)
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
            self.graph_builder.add_nodes(clustered)
            graph_frame = self.graph_builder.render_2d_graph_frame(view=self.graph_view)
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
            header=f"Task: {self.task} | Graph Level {1 if self.graph_view == 'semantic' else 2}",
            side_image=mask_frame,
            side_header=f"Mask(s): {', '.join(sorted(self.masks)) or 'None'}",
        )
        self._handle_graph_view_key()
        self.frames_processed += 1
        return keep_running

    def _handle_graph_view_key(self):
        key = self.video_output.last_key
        if key == ord("1"):
            self._set_graph_view("semantic")
        elif key == ord("2"):
            self._set_graph_view("base")

    def _set_graph_view(self, view):
        if self.graph_view == view:
            return

        self.graph_view = view
        graph_frame = self.graph_builder.render_2d_graph_frame(view=self.graph_view)
        if graph_frame is not None:
            self.last_graph_frame = graph_frame
        print(f"Graph view: {self.graph_view}")

    def run(self):
        while self.has_next():
            if not self.run_frame():
                break
        return PriorityMapResult(
            output_dir=self.output_dir,
            observations_csv=self.observations_csv,
            video_path=self.video_output.video_path,
            heatmap_video_path=self.heatmap_video_output.video_path,
            frames_processed=self.frames_processed,
        )
