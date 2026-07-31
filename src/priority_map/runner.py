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
from priority_map.modules.Direction import Direction
from priority_map.modules.Segment import Segment
from priority_map.modules.GraphBuilder import GraphBuilder
from priority_map.modules.drone_motion import DroneMotion
from priority_map.modules.object_localizing.localizer import LocalizationContext
from priority_map.modules.object_localizing.flow_localizer import FlowLocalizer
from priority_map.modules.object_localizing.gps_localizer import GpsLocalizer
from priority_map.modules.PanoramaBuilder import PanoramaBuilder

load_dotenv()


# DEFAULT_IMAGE_FOLDER = Path(r"D:\UAV_VisLoc_dataset\05\drone")
# DEFAULT_IMAGE_FOLDER = Path(r"D:\Train\Train\query_images")
# DEFAULT_IMAGE_FOLDER = Path(r"D:\dronevid2")
# DEFAULT_IMAGE_FOLDER = Path(r"D:\rtereg") 
DEFAULT_IMAGE_FOLDER = None

def default_output_dir() -> Path:
    return Path("examples") / time.strftime("%Y-%m-%d_%H-%M-%S")


def is_ignored_input_artifact(path_or_name) -> bool:
    path = Path(str(path_or_name))
    return (
        path.name.startswith("._")
        or path.name == ".DS_Store"
        or "__MACOSX" in path.parts
    )


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
    numerical_heatmap: np.ndarray | None
    direction: np.ndarray | None
    came_from: np.ndarray | None
    output_frame: np.ndarray | None
    latency_seconds: dict[str, float]
    keep_running: bool

    def __bool__(self):
        return self.keep_running


@dataclass
class FramePacket:
    image: np.ndarray
    frame_index: int
    image_name: str
    image_path: str | None
    easting: float | None = None
    northing: float | None = None
    altitude: float | None = None
    orientation: tuple[float, float, float, float] | None = None


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
        dilation_scale=config.DILATION_SCALE,
        max_image_edge=config.MAX_IMAGE_EDGE,
        sam_model_path=config.SAM_MODEL_PATH,
        debug=False,
        record=True,
        panoramic=False,
        gps_csv: str | Path | None = None,
        camera_intrinsics: str | Path | None = None,
        scene_model: str | None = None,
        vector_ema_alpha: float = config.VECTOR_EMA_ALPHA,
    ):
        self.dataset_root = Path(image_folder) if image_folder is not None else DEFAULT_IMAGE_FOLDER
        self.gps_csv_path = Path(gps_csv) if gps_csv is not None else None
        self.camera_intrinsics_path = Path(camera_intrinsics) if camera_intrinsics is not None else None
        self.task = task
        self.debrief = debrief
        self.task_description = task if not debrief else f"{task}: {debrief}"
        self.masks = mask or []
        self.sam_step = sam_step
        self.sam_thresh = sam_thresh
        self.blur_spread = blur_spread
        self.dilation_scale = dilation_scale
        self.max_image_edge = max_image_edge
        self.sam_model_path = sam_model_path
        self.panoramic = panoramic
        self.debug = debug
        if not 0 < vector_ema_alpha <= 1:
            raise ValueError("vector_ema_alpha must be greater than 0 and at most 1.")
        self.vector_ema_alpha = vector_ema_alpha
        self._direction_ema = None
        self._came_from_ema = None
        
        self.query_csv, self.query_images_dir = self._load_dataset_index()

        self.index = 0
        self.frames_processed = 0
        self.output_dir = Path(output_dir) if output_dir is not None else default_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.observations_csv = self.output_dir / "observations.csv"

        self.scene_understanding = SceneUnderstanding(
            debug=debug,
            model=scene_model,
        )
        self.segmentation = Segment(
            debug=debug,
            sam_thresh=sam_thresh,
            sam_model_path=sam_model_path,
        )
        self.graph_builder = GraphBuilder(output_dir=self.output_dir, debug=debug)
        self.graph_builder.set_original_task(task)
        self.heatmap = Heatmap(blur_spread=blur_spread, dilation_scale=dilation_scale)
        self.direction = Direction()
        self.drone_motion = DroneMotion()
        self.flow_localizer = FlowLocalizer()
        self.gps_localizer = GpsLocalizer()
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
            show=not debug,
            record=record,
            filename="heatmap.avi",
            window_name="Priority Heatmap",
            debug=debug,
        )

        self.heat_panoramic_builder = PanoramaBuilder(alpha=0.9)
        self.panoramic_builder = PanoramaBuilder(alpha=0.15)
        self.last_graph_frame = None
        self.graph_view = "spatial"
        self._closed = False

    def _load_dataset_index(self):
        if self.dataset_root is None:
            return pd.DataFrame({"name": []}), None

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

        image_files = [
            path
            for path in sorted(self.dataset_root.iterdir())
            if (
                path.is_file()
                and path.suffix.lower() in image_extensions
                and not is_ignored_input_artifact(path)
            )
        ]
        if image_files:
            query_csv = pd.DataFrame({"name": [path.name for path in image_files]})
            return query_csv, self.dataset_root

        raise FileNotFoundError(f"No images found in {self.dataset_root}")

    def _row_value(self, row, *columns, default=0.0):
        if row is None:
            return default

        for column in columns:
            if column in row and pd.notna(row[column]):
                return row[column]
        return default

    def has_next(self) -> bool:
        return self.index < len(self.query_csv)

    def reset(self):
        self.index = 0
        self._direction_ema = None
        self._came_from_ema = None

    def should_run_sam(self, frame):
        return frame.frame_index % self.sam_step == 0

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

    def _prepare_input_image(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array or a path to an image file.")
        if image.size == 0:
            raise ValueError("image must not be empty.")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            image = image.copy()
        else:
            raise ValueError("image must be grayscale, BGR, or BGRA.")

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        image = self._resize_input_image(image)
        image[:, :, 1] = (image[:, :, 1] * 0.65).astype(image.dtype)
        image[:, :, 2:3] = (image[:, :, 2:3] * 0.8).astype(image.dtype)
        return image

    def _gps_row_for_image(self, image_name):
        if self.gps_csv_path is None:
            return None

        gps_csv = pd.read_csv(self.gps_csv_path)

        if "name" not in gps_csv.columns:
            raise ValueError(f"{self.gps_csv_path} must contain a 'name' column.")

        matches = gps_csv[gps_csv["name"].astype(str) == str(image_name)]
        if matches.empty:
            return None

        return matches.iloc[0]

    def _orientation_from_row(self, row):
        orientation_columns = ("orient_x", "orient_y", "orient_z", "orient_w")
        if row is None or any(column not in row or pd.isna(row[column]) for column in orientation_columns):
            return None

        return tuple(float(row[column]) for column in orientation_columns)

    def get_next_frame(self):
        while self.has_next():
            frame_index = self.index
            row = self.query_csv.iloc[frame_index]
            self.index += 1

            image_name = self._row_value(row, "name", "filename", default="")
            if is_ignored_input_artifact(image_name):
                continue

            image_path = (self.query_images_dir / str(image_name))
            image = cv2.imread(str(image_path))

            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")

            image = self._prepare_input_image(image)

            gps_row = self._gps_row_for_image(image_name)
            return FramePacket(
                image=image,
                frame_index=frame_index,
                image_name=str(image_name),
                image_path=str(image_path),
                easting=self._row_value(gps_row, "easting", default=None),
                northing=self._row_value(gps_row, "northing", default=None),
                altitude=self._row_value(gps_row, "altitude", default=None),
                orientation=self._orientation_from_row(gps_row),
            )

        return None

    def _frame_from_input(
        self,
        image: np.ndarray | str | Path,
        *,
        image_name: str | None = None,
        frame_index: int | None = None,
        easting: float | None = None,
        northing: float | None = None,
        altitude: float | None = None,
        orientation: tuple[float, float, float, float] | None = None,
    ) -> FramePacket:
        image_path = None
        if isinstance(image, (str, Path)):
            source_path = Path(image)
            loaded_image = cv2.imread(str(source_path))
            if loaded_image is None:
                raise FileNotFoundError(f"Could not read image: {source_path}")
            image_path = str(source_path)
            image_name = image_name or source_path.name
        else:
            loaded_image = image
            image_name = image_name or f"frame_{self.frames_processed:06d}.png"

        prepared_image = self._prepare_input_image(loaded_image)
        gps_row = self._gps_row_for_image(image_name)

        return FramePacket(
            image=prepared_image,
            frame_index=self.frames_processed if frame_index is None else frame_index,
            image_name=image_name,
            image_path=image_path,
            easting=(
                self._row_value(gps_row, "easting", default=None)
                if easting is None
                else easting
            ),
            northing=(
                self._row_value(gps_row, "northing", default=None)
                if northing is None
                else northing
            ),
            altitude=(
                self._row_value(gps_row, "altitude", default=None)
                if altitude is None
                else altitude
            ),
            orientation=orientation or self._orientation_from_row(gps_row),
        )

    def _localize_segmentations(self, segmentations, frame, image, flow_transform):
        curr_pos = (frame.easting, frame.northing, frame.altitude)
        context = LocalizationContext(
            frame=frame,
            image=image,
            curr_pos=curr_pos,
            flow_transform=flow_transform,
        )
        has_gps = (
            frame.easting is not None
            and frame.northing is not None
            and frame.altitude is not None
        )
        localizer = self.gps_localizer if has_gps else self.flow_localizer

        for segmentation in segmentations:
            geo_pos = localizer.localize(segmentation, context)

            if geo_pos is not None:
                segmentation.geo_pos = geo_pos

        return segmentations

    def _draw_debug_directions(self, image, direction, came_from):
        if not self.debug or image is None:
            return image

        main_arrows = [
            (vector, color)
            for vector, color in [
                (direction, (255, 255, 255)),
                (came_from, (255, 0, 255)),
            ]
            if vector is not None and np.linalg.norm(vector) > 0
        ]
        if not main_arrows:
            return image

        output = image.copy()
        height, width = output.shape[:2]
        center = (width // 2, height // 2)
        arrow_length = max(1, int(min(width, height) * 0.25))
        thickness = max(2, int(round(min(width, height) / 240)))
        for vector, color in main_arrows:
            endpoint = (
                int(round(center[0] + vector[0] * arrow_length)),
                int(round(center[1] - vector[1] * arrow_length)),
            )
            cv2.arrowedLine(
                output,
                center,
                endpoint,
                color,
                thickness,
                cv2.LINE_AA,
                tipLength=0.25,
            )
        return output

    def _smooth_vector(self, vector, state_attribute):
        if vector is None:
            setattr(self, state_attribute, None)
            return None

        current = np.asarray(vector, dtype=np.float32)
        if current.shape != (2,) or not np.all(np.isfinite(current)):
            raise ValueError("EMA vectors must contain two finite values.")

        magnitude = np.linalg.norm(current)
        if magnitude == 0:
            ema = np.zeros(2, dtype=np.float32)
        else:
            current = current / magnitude
            previous = getattr(self, state_attribute, None)
            if previous is None:
                ema = current
            else:
                alpha = self.vector_ema_alpha
                ema = alpha * current + (1.0 - alpha) * previous

        ema = np.asarray(ema, dtype=np.float32)
        ema_magnitude = np.linalg.norm(ema)
        if ema_magnitude <= np.finfo(np.float32).eps and magnitude > 0:
            ema = current
            ema_magnitude = np.linalg.norm(ema)

        setattr(self, state_attribute, ema)
        if ema_magnitude == 0:
            return np.zeros(2, dtype=np.float32)
        return np.asarray(ema / ema_magnitude, dtype=np.float32)

    def _coverage_directions(self, frame, image):
        has_gps = (
            frame.easting is not None
            and frame.northing is not None
            and frame.altitude is not None
        )
        if has_gps:
            current_position = np.array(
                [frame.easting, frame.northing],
                dtype=np.float32,
            )
        else:
            height, width = image.shape[:2]
            current_position = np.array(
                [
                    width / 2.0 + self.flow_localizer.cumulative_transform_dx,
                    height / 2.0 + self.flow_localizer.cumulative_transform_dy,
                ],
                dtype=np.float32,
            )

        node_positions = self.graph_builder.get_nearby_node_positions(
            current_position,
            radius=200,
            limit=10,
        )
        directions = []
        for node_position in node_positions:
            delta = np.asarray(node_position, dtype=np.float32) - current_position
            if not has_gps:
                delta[1] *= -1
            distance = np.linalg.norm(delta)
            if distance > 0:
                directions.append(delta / distance)
        return directions
    
    def close(self):
        if self._closed:
            return

        self._closed = True

        # Release video writers/windows before slower shutdown work so Ctrl+C
        # cannot leave the output file waiting on unrelated cleanup.
        cleanup_steps = [
            ("main video output", self.video_output.close),
            ("heatmap video output", self.heatmap_video_output.close),
            ("SAM preview output", self.segmentation.close),
            ("graph builder", self.graph_builder.close),
        ]
        for name, close in cleanup_steps:
            close()

    def run_frame(
        self,
        image: np.ndarray | str | Path | None = None,
        *,
        image_name: str | None = None,
        frame_index: int | None = None,
        easting: float | None = None,
        northing: float | None = None,
        altitude: float | None = None,
        orientation: tuple[float, float, float, float] | None = None,
    ):
        frame_t0 = time.perf_counter()
        if image is None:
            frame = self.get_next_frame()
        else:
            frame = self._frame_from_input(
                image,
                image_name=image_name,
                frame_index=frame_index,
                easting=easting,
                northing=northing,
                altitude=altitude,
                orientation=orientation,
            )
        if frame is None:
            total_seconds = time.perf_counter() - frame_t0
            return PriorityFrameResult(
                image_name=None,
                image_path=None,
                frame_index=None,
                heatmap_only=None,
                numerical_heatmap=None,
                direction=None,
                came_from=None,
                output_frame=None,
                latency_seconds={
                    "total": total_seconds,
                    "vlm": 0.0,
                    "sam3": 0.0,
                    "other": total_seconds,
                },
                keep_running=False,
            )

        image = frame.image
        out = image
        vlm_seconds = 0.0
        sam3_seconds = 0.0

        # Position/geolocation is disabled for now so a plain folder of images just works.
        # position = (
        #     frame.easting,
        #     frame.northing,
        #     frame.altitude
        # )

        scene_dict = None
        scene_edge_intents = []
        recent_graph_context = {"nodes": [], "spatial_edges": [], "model_edges": []}
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
        came_from = self._smooth_vector(
            self.drone_motion.get_came_from(
                frame,
                segmentation_result.flow_transform,
            ),
            "_came_from_ema",
        )

        segmentations = self._localize_segmentations(
            segmentations,
            frame,
            image,
            segmentation_result.flow_transform,
        )

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
                    frame.image_name,
                    cluster.label,
                    cx,
                    cy,
                ])

        heatmap_images_dir = self.output_dir / "heatmap_imgs"
        heatmap_images_dir.mkdir(parents=True, exist_ok=True)

        # Set Segmentation Types to Display MASKS
        mask_frame = None
        if self.masks:
            mask_frame = label_mask(self.masks, image, segmentations)

        # Create Heatmap
        heatmap_text, heatmap_only, numerical_heatmap = self.heatmap.draw_heatmap(
            image,
            clustered,
        )
        direction = self._smooth_vector(
            self.direction.get_direction(
                numerical_heatmap,
                came_from,
            ),
            "_direction_ema",
        )
        heatmap_keep_running = True
        if heatmap_text is not None and heatmap_only is not None:
            out = heatmap_text
            heatmap_keep_running = self.heatmap_video_output.handle_frame(
                heatmap_text,
                header=f"Task: {self.task}",
            )
        out = self._draw_debug_directions(out, direction, came_from)

        # Save Heatmap Individual Heatmap Images
        safe_imwrite(str(heatmap_images_dir / frame.image_name), heatmap_only)

        if scene_dict is not None:
            add_result = self.graph_builder.add_nodes(clustered)
            self.graph_builder.resolve_scene_edge_intents(
                scene_edge_intents,
                add_result,
                recent_graph_context,
            )
            graph_frame = self.graph_builder.render_2d_graph_frame(view=self.graph_view)
            if graph_frame is not None:
                self.last_graph_frame = graph_frame

        else:
            self.graph_builder.assign_existing_node_ids(clustered)

        # Save the DB node ID matched to each clustered mask for every frame.
        assigned_node_ids = [
            cluster.node_id
            for cluster in clustered
            if getattr(cluster, "node_id", None)
        ]
        max_node_id_length = max(map(len, assigned_node_ids), default=1)
        node_ids = np.full(image.shape[:2], "", dtype=f"<U{max_node_id_length}")
        for cluster in sorted(clustered, key=lambda cluster: cluster.score, reverse=True):
            node_id = getattr(cluster, "node_id", None)
            if not node_id:
                continue
            mask = cluster.mask.astype(bool)
            node_ids[mask & (node_ids == "")] = node_id
        node_ids_path = heatmap_images_dir / f"{Path(frame.image_name).stem}.nodes.npz"
        np.savez_compressed(node_ids_path, node_ids=node_ids)

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

        main_keep_running = self.video_output.handle_frame(
            out,
            header=f"Task: {self.task} | Graph: {self.graph_view.title()}",
            side_image=mask_frame,
            side_header=f"Mask(s): {', '.join(sorted(self.masks)) or 'None'}",
        )
        keep_running = heatmap_keep_running and main_keep_running
        self._handle_graph_view_key()
        self.frames_processed += 1
        total_seconds = time.perf_counter() - frame_t0
        return PriorityFrameResult(
            image_name=frame.image_name,
            image_path=frame.image_path,
            frame_index=frame.frame_index,
            heatmap_only=heatmap_only,
            numerical_heatmap=numerical_heatmap,
            direction=direction,
            came_from=came_from,
            output_frame=out,
            latency_seconds={
                "total": total_seconds,
                "vlm": vlm_seconds,
                "sam3": sam3_seconds,
                "other": max(total_seconds - vlm_seconds - sam3_seconds, 0.0),
            },
            keep_running=keep_running,
        )

    def _handle_graph_view_key(self):
        key = self.video_output.last_key
        if key == ord("1"):
            self._set_graph_view("spatial")
        elif key == ord("2"):
            self._set_graph_view("model")

    def _set_graph_view(self, view):
        if view not in {"model", "spatial"}:
            raise ValueError(f"Unknown graph view: {view}")
        if self.graph_view == view:
            return

        self.graph_view = view
        graph_frame = self.graph_builder.render_2d_graph_frame(view=view)
        if graph_frame is not None:
            self.last_graph_frame = graph_frame
        if self.debug:
            print(f"Graph view: {view}")

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
