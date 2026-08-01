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
from priority_map.scripts.geospatial import (
    CesiumFrameMetadata,
    lonlat_to_local,
)

load_dotenv()


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
    goal_found: bool
    georeference_valid: bool
    heatmap_only: np.ndarray | None
    numerical_heatmap: np.ndarray | None
    direction: np.ndarray | None
    direction_mode: str
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
    cesium_metadata: CesiumFrameMetadata | None = None


class PriorityMapRunner:
    DIRECTION_MINIMUM_HEAT = 1.0

    def __init__(
        self,
        image_folder: str | Path | None,
        output_dir: str | Path | None = None,
        task="Find cars",
        debrief: str | None = None,
        mask=None,
        sam_step=config.SAM_STEP,
        sam_thresh=config.SAM_TRESH,
        blur_spread=config.BLUR_SPREAD,
        dilation_scale=config.DILATION_SCALE,
        max_image_edge=config.MAX_IMAGE_EDGE,
        *,
        sam_model_path: str | Path,
        debug=False,
        record=True,
        panoramic=False,
        gps_csv: str | Path | None = None,
        camera_intrinsics: str | Path | None = None,
        scene_model: str | None = None,
        vector_ema_alpha: float = config.VECTOR_EMA_ALPHA,
        max_direction_turn_degrees: float = config.MAX_DIRECTION_TURN_DEGREES,
        exclusion_angle_degrees: float = config.EXCLUSION_ANGLE_DEGREES,
        max_observed_coverage_ratio: float = config.MAX_OBSERVED_COVERAGE_RATIO,
        coverage_lookahead_seconds: float = config.COVERAGE_LOOKAHEAD_SECONDS,
        persist_artifacts: bool = True,
        require_cesium_georeference: bool = False,
    ):
        self.dataset_root = Path(image_folder) if image_folder is not None else None
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
        self.persist_artifacts = persist_artifacts
        self.require_cesium_georeference = require_cesium_georeference
        self._geographic_origin = None
        if not 0 < vector_ema_alpha <= 1:
            raise ValueError("vector_ema_alpha must be greater than 0 and at most 1.")
        if not 0 < max_direction_turn_degrees <= 180:
            raise ValueError(
                "max_direction_turn_degrees must be greater than 0 "
                "and at most 180."
            )
        if not 0 <= max_observed_coverage_ratio <= 1:
            raise ValueError(
                "max_observed_coverage_ratio must be between 0 and 1."
            )
        if (
            not np.isfinite(coverage_lookahead_seconds)
            or coverage_lookahead_seconds <= 0
        ):
            raise ValueError(
                "coverage_lookahead_seconds must be finite and greater than 0."
            )
        self.vector_ema_alpha = vector_ema_alpha
        self.max_direction_turn_degrees = max_direction_turn_degrees
        self.max_observed_coverage_ratio = float(
            max_observed_coverage_ratio
        )
        self.coverage_lookahead_seconds = float(
            coverage_lookahead_seconds
        )
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
        self.direction = Direction(
            exclusion_angle_degrees=exclusion_angle_degrees,
        )
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
        cesium_metadata: CesiumFrameMetadata | None = None,
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
            cesium_metadata=cesium_metadata,
        )

    def _localize_segmentations(self, segmentations, frame, image, flow_transform):
        cesium_metadata = frame.cesium_metadata
        if (
            cesium_metadata is not None
            and cesium_metadata.is_valid()
        ):
            if self._geographic_origin is None:
                self._geographic_origin = (
                    cesium_metadata.longitude,
                    cesium_metadata.latitude,
                )
            origin_longitude, origin_latitude = self._geographic_origin
            coverage_radius_m = cesium_metadata.coverage_radius_m(
                image.shape,
                grid_size=5,
            )
            for segmentation in segmentations:
                if segmentation.centroid is None:
                    continue
                longitude, latitude = cesium_metadata.project_pixel(
                    segmentation.centroid,
                    image.shape,
                )
                local_position = lonlat_to_local(
                    longitude,
                    latitude,
                    origin_longitude,
                    origin_latitude,
                )
                segmentation.geo_pos = tuple(local_position)
                segmentation.longitude = longitude
                segmentation.latitude = latitude
                segmentation.ground_height_m = (
                    cesium_metadata.ground_height_m
                )
                segmentation.coverage_radius_m = coverage_radius_m
                segmentation.observed_frame = frame.frame_index
            return segmentations

        if getattr(self, "require_cesium_georeference", False):
            for segmentation in segmentations:
                segmentation.geo_pos = None
            return segmentations

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

    @staticmethod
    def _validated_speed(speed_mps):
        if speed_mps is not None:
            speed_mps = float(speed_mps)
            if not np.isfinite(speed_mps) or speed_mps < 0.0:
                raise ValueError("speed_mps must be finite and nonnegative.")
        return speed_mps

    def _camera_center_en(self, cesium_metadata):
        if self._geographic_origin is None:
            return None
        return lonlat_to_local(
            cesium_metadata.longitude,
            cesium_metadata.latitude,
            self._geographic_origin[0],
            self._geographic_origin[1],
        )

    def _update_observed_nodes_for_frame(self, frame, image):
        metadata = frame.cesium_metadata
        if metadata is None or not metadata.is_valid():
            return
        camera_center = self._camera_center_en(metadata)
        if camera_center is None:
            return

        visible_node_ids = [
            node["id"]
            for node in self.graph_builder.get_georeferenced_nodes()
            if metadata.local_point_is_visible(
                node["position"],
                camera_center,
                image.shape,
            )
        ]
        self.graph_builder.update_observed_nodes(visible_node_ids)

    def _observed_region_validator(
        self,
        frame,
        image,
        speed_mps,
    ):
        metadata = frame.cesium_metadata
        if (
            speed_mps is None
            or metadata is None
            or not metadata.is_valid()
        ):
            return None

        camera_center = self._camera_center_en(metadata)
        if camera_center is None:
            return None
        observed_regions = [
            (node["position"], node["coverage_radius_m"])
            for node in self.graph_builder.get_georeferenced_nodes(
                observed_only=True,
            )
            if node["coverage_radius_m"] is not None
            and node["coverage_radius_m"] > 0.0
        ]
        if not observed_regions:
            return None

        sample_times = np.linspace(
            self.coverage_lookahead_seconds / 5.0,
            self.coverage_lookahead_seconds,
            5,
        )

        def candidate_is_feasible(direction):
            movement_direction = metadata.image_direction_to_en(direction)
            if np.linalg.norm(movement_direction) == 0.0:
                return True
            for sample_time in sample_times:
                predicted_center = (
                    camera_center
                    + movement_direction * speed_mps * float(sample_time)
                )
                coverage_ratio = metadata.observed_circle_coverage_ratio(
                    predicted_center,
                    image.shape,
                    observed_regions,
                )
                if coverage_ratio > self.max_observed_coverage_ratio:
                    return False
            return True

        return candidate_is_feasible

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

    def _smooth_vector(
        self,
        vector,
        state_attribute,
        max_turn_degrees=None,
    ):
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

        previous = getattr(self, state_attribute, None)
        if (
            max_turn_degrees is not None
            and previous is not None
            and ema_magnitude > 0
        ):
            ema = self._limit_vector_turn(
                previous,
                ema,
                max_turn_degrees,
            )
            ema_magnitude = np.linalg.norm(ema)

        setattr(self, state_attribute, ema)
        if ema_magnitude == 0:
            return np.zeros(2, dtype=np.float32)
        return np.asarray(ema / ema_magnitude, dtype=np.float32)

    def _limit_vector_turn(self, previous, candidate, max_turn_degrees):
        previous = np.asarray(previous, dtype=np.float32)
        candidate = np.asarray(candidate, dtype=np.float32)
        previous_magnitude = np.linalg.norm(previous)
        candidate_magnitude = np.linalg.norm(candidate)
        if previous_magnitude == 0 or candidate_magnitude == 0:
            return candidate

        previous = previous / previous_magnitude
        candidate = candidate / candidate_magnitude
        dot = float(np.clip(np.dot(previous, candidate), -1.0, 1.0))
        cross = float(
            previous[0] * candidate[1] -
            previous[1] * candidate[0]
        )
        signed_angle = float(np.arctan2(cross, dot))
        maximum_angle = float(np.deg2rad(max_turn_degrees))
        if abs(signed_angle) <= maximum_angle:
            return candidate

        limited_angle = np.copysign(maximum_angle, signed_angle)
        cosine = np.cos(limited_angle)
        sine = np.sin(limited_angle)
        return np.array(
            [
                previous[0] * cosine - previous[1] * sine,
                previous[0] * sine + previous[1] * cosine,
            ],
            dtype=np.float32,
        )

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
        speed_mps: float | None = None,
        cesium_metadata: CesiumFrameMetadata | None = None,
    ):
        frame_t0 = time.perf_counter()
        speed_mps = self._validated_speed(speed_mps)
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
                cesium_metadata=cesium_metadata,
            )
        if frame is None:
            total_seconds = time.perf_counter() - frame_t0
            return PriorityFrameResult(
                image_name=None,
                image_path=None,
                frame_index=None,
                goal_found=False,
                georeference_valid=False,
                heatmap_only=None,
                numerical_heatmap=None,
                direction=None,
                direction_mode="hold",
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
        georeference_valid = bool(
            frame.cesium_metadata is not None
            and frame.cesium_metadata.is_valid()
        )
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
        goal_found = False
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
                goal_found = self._scene_contains_goal(scene_dict)
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
        self._update_observed_nodes_for_frame(frame, image)

        heatmap_images_dir = None
        if self.persist_artifacts:
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
        numerical_heatmap = np.asarray(
            numerical_heatmap
            if numerical_heatmap is not None
            else np.zeros(image.shape[:2], dtype=np.float32),
            dtype=np.float32,
        )
        maximum_heat = float(
            np.max(numerical_heatmap)
            if numerical_heatmap.size > 0
            else 0.0
        )
        navigation_heatmap = (
            numerical_heatmap
            if maximum_heat >= self.DIRECTION_MINIMUM_HEAT
            else np.zeros_like(numerical_heatmap)
        )
        candidate_validator = self._observed_region_validator(
            frame,
            image,
            speed_mps,
        )
        direction_decision = self.direction.get_decision(
            navigation_heatmap,
            came_from=came_from,
            candidate_validator=candidate_validator,
        )
        raw_direction = direction_decision.direction
        raw_is_coverage_feasible = bool(
            candidate_validator is None
            or candidate_validator(raw_direction)
        )
        direction = self._smooth_vector(
            raw_direction,
            "_direction_ema",
            max_turn_degrees=self.max_direction_turn_degrees,
        )
        smoothed_is_blocked = (
            direction is not None
            and np.linalg.norm(direction) > 0
            and (
                self.direction.is_blocked_by_came_from(
                    direction,
                    came_from,
                )
                or (
                    candidate_validator is not None
                    and raw_is_coverage_feasible
                    and not candidate_validator(direction)
                )
            )
        )
        if smoothed_is_blocked:
            direction = np.asarray(raw_direction, dtype=np.float32)
            self._direction_ema = direction.copy()

        if direction is None or np.linalg.norm(direction) == 0:
            direction_mode = "hold"
        elif direction_decision.patch_heat > 0:
            direction_mode = "target"
        else:
            direction_mode = "explore"
        heatmap_keep_running = True
        if heatmap_text is not None and heatmap_only is not None:
            out = heatmap_text
            heatmap_keep_running = self.heatmap_video_output.handle_frame(
                heatmap_text,
                header=f"Task: {self.task}",
            )
        out = self._draw_debug_directions(out, direction, came_from)

        # Save Heatmap Individual Heatmap Images
        if self.persist_artifacts:
            safe_imwrite(str(heatmap_images_dir / frame.image_name), heatmap_only)

        graph_location_is_valid = (
            georeference_valid
            or not getattr(
                self,
                "require_cesium_georeference",
                False,
            )
        )
        if scene_dict is not None and graph_location_is_valid:
            add_result = self.graph_builder.add_nodes(clustered)
            self.graph_builder.resolve_scene_edge_intents(
                scene_edge_intents,
                add_result,
                recent_graph_context,
            )
            if self.persist_artifacts or self.debug:
                graph_frame = self.graph_builder.render_2d_graph_frame(view=self.graph_view)
                if graph_frame is not None:
                    self.last_graph_frame = graph_frame

        elif graph_location_is_valid:
            self.graph_builder.assign_existing_node_ids(clustered)

        if self.persist_artifacts:
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
            goal_found=goal_found,
            georeference_valid=georeference_valid,
            heatmap_only=heatmap_only,
            numerical_heatmap=numerical_heatmap,
            direction=direction,
            direction_mode=direction_mode,
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

    @staticmethod
    def _scene_contains_goal(scene_dict):
        if not scene_dict:
            return False

        for label_info in scene_dict.values():
            if not isinstance(label_info, dict):
                continue
            try:
                if float(label_info.get("score")) == 100.0:
                    return True
            except (TypeError, ValueError):
                continue

        return False

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
