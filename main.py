import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import math
import time
from pathlib import Path
import cv2
import pandas as pd
import traceback
from dotenv import load_dotenv
import numpy as np

from scripts.video_helper import label_mask, safe_imwrite

from modules.SceneUnderstanding import SceneUnderstanding
from modules.Heatmap import Heatmap
from modules.Segment import Segment, Segmentation, _resize_to_screen
from modules.GraphBuilder import GraphBuilder
from modules.GraphChat import ChatWithGraph
from modules.GeoLocalizer import GeoLocalizer
from modules.PanoramaBuilder import PanoramaBuilder

from scripts.video_helper import (
    compose_video_frame,
    get_video_writer,
    release_video_writer,
)

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_folder",
        nargs="?",
        help="Folder of images to process.",
    )
    parser.add_argument(
        "--dataset-root",
        default=r"C:\Users\jletobar3\Downloads\UAV_VisLoc_example\03\drone",
        help="Dataset root. Supports plain image folders, old query.csv/query_images, or UAV_VisLoc csv/drone layouts.",
    )
    parser.add_argument("--task", default="Find cars")
    parser.add_argument("--mask", nargs="*", default=[])
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--record-ui", action="store_true")
    parser.add_argument("--record-graph", action="store_true")
    return parser.parse_args()


class DroneHeatmap: 
    def __init__(self, dataset_root: str, task="Find cars", mask=None, sam_step=3):
        self.dataset_root = Path(dataset_root)
        self.task = task
        self.masks = mask or []
        self.sam_step = sam_step
        
        self.query_csv, self.query_images_dir = self._load_dataset_index()

        self.index = 0
        self.output_dir = Path("examples") / time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scene_understanding = SceneUnderstanding()
        self.segmentation = Segment()
        self.graph_builder = GraphBuilder(graph_path=self.output_dir / "graph.json")
        self.heatmap = Heatmap()
        self.geo_localizer = GeoLocalizer()
        self.graph_chat = ChatWithGraph(self.graph_builder.G)

        self.masks = {m.lower() for m in self.masks}

        self.heat_panoramic_builder = PanoramaBuilder(alpha=0.9)
        self.panoramic_builder = PanoramaBuilder(alpha=0.5)

        self.video_writer = None
        self.video_path = None

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
                f"No images, query.csv, or UAV_VisLoc csv file found in {self.dataset_root}"
            )

        drone_dir = self.dataset_root / "drone"
        if not drone_dir.exists():
            raise FileNotFoundError(f"Expected drone image folder at {drone_dir}")

        query_csv = pd.read_csv(csv_files[0])
        if "filename" not in query_csv.columns:
            raise ValueError(f"{csv_files[0]} must contain a 'filename' column.")
        required_columns = {"lat", "lon", "height"}
        missing_columns = required_columns - set(query_csv.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_files[0]} is missing required column(s): {missing}")

        query_csv = self._normalize_uav_visloc_csv(query_csv)

        return query_csv, drone_dir

    def _normalize_uav_visloc_csv(self, query_csv):
        query_csv = query_csv.copy()
        origin_lat = float(query_csv.iloc[0]["lat"])
        origin_lon = float(query_csv.iloc[0]["lon"])

        meters_per_degree_lat = 111_320.0
        meters_per_degree_lon = (
            meters_per_degree_lat * math.cos(math.radians(origin_lat))
        )

        query_csv["name"] = query_csv["filename"]
        query_csv["easting"] = (
            query_csv["lon"].astype(float) - origin_lon
        ) * meters_per_degree_lon
        query_csv["northing"] = (
            query_csv["lat"].astype(float) - origin_lat
        ) * meters_per_degree_lat
        query_csv["altitude"] = query_csv["height"].astype(float)

        orientation_columns = {
            "Omega": "orient_x",
            "Kappa": "orient_y",
            "Phi1": "orient_z",
            "Phi2": "orient_w",
        }
        for source_column, target_column in orientation_columns.items():
            query_csv[target_column] = self._series_or_default(
                query_csv,
                source_column,
                0.0,
            )

        return query_csv

    def _series_or_default(self, dataframe, column, default):
        if column in dataframe.columns:
            return dataframe[column].astype(float)
        return default

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

            image_name = self._row_value(row, "name", "filename", default="")
            image_path = (self.query_images_dir / str(image_name))
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
    
    def close_video(self):
        release_video_writer(self.video_writer)
        self.video_writer = None

    def show_video(self, image, header, side_image=None, side_header=None):
        image = compose_video_frame(
            image,
            header,
            side_image=side_image,
            side_header=side_header,
        )

        self.video_writer, video_path = get_video_writer(
            self.video_writer,
            image,
            self.output_dir,
        )
        if video_path is not None:
            self.video_path = video_path

        self.video_writer.write(image)

        return image
    
    def set_mask(self, text):
        print(text)
        return f"Received: {text}"

    def set_masks(self, masks):
        self.masks = {mask.lower() for mask in masks}

    def ask_graph(self, text):
        graph = self.graph_builder._build_topology().copy()
        return self.graph_chat.ask(text, graph=graph)
    

    def run(self):

        if self.has_next():

            # Get Frame Info
            frame = self.get_next_frame()
            if frame is None:
                return

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
                scene_dict = self.scene_understanding.get_labels(image, self.task)
            # print(scene_dict)

            # Get Segmentations From Image
            segmentations = self.segmentation.get_segmentations(image, scene_dict)
            if segmentations is None:
                segmentations = []

            for segmentation in segmentations:
                segmentation.geo_pos = None
                # segmentation.geo_pos = self.geo_localizer.get_location(
                #     image,
                #     segmentation.mask,
                #     position
                # )
                # cv2.imshow("Segmentation", segmentation.mask.astype(np.uint8) * 255)

            # Set Segmentation Types to Display
            mask_frame = None
            if self.masks:
                mask_frame = label_mask(self.masks, image, segmentations)

            # curr_nodes = self.graph_builder.build_graph(segmentations)
            # graph_frame = self.graph_builder.render_2d_graph_frame()
            graph_frame = None
    
            # Create Heatmap
            heatmap = self.heatmap.draw_heatmap(image, segmentations)
            # print(f"Frame {frame['frame_index']}:", end=" ")
            # for node_id in self.graph_builder.G.nodes:
            #     print(node_id, end=" ")
            # print()
            if heatmap is not None: out = heatmap

            # Create Panoramic Images
            os.makedirs(f"{self.output_dir}/heat_panorama", exist_ok=True)
            os.makedirs(f"{self.output_dir}/panorama", exist_ok=True)
            transform = self.segmentation.transform_dx, self.segmentation.transform_dy

            heat_panorama = self.heat_panoramic_builder.create_panorama(transform, heatmap)
            # cv2.imshow("Heat Panorama", self.heat_panoramic_builder.panorama)
            # cv2.waitKey(1)
            if self.index % 10 == 0: safe_imwrite(f"{self.output_dir}/heat_panorama/heat_panorama_{self.index}.png", heat_panorama)

            panorama = self.panoramic_builder.create_panorama(transform, image)
            # cv2.imshow("Panorama", self.panoramic_builder.panorama)
            # cv2.waitKey(1)
            if self.index % 10 == 0: safe_imwrite(f"{self.output_dir}/panorama/panorama_{self.index}.png", panorama)


            video_frame = self.show_video(
                out,
                header=f"Task: {self.task}",
                side_image=mask_frame,
                side_header=f"Mask(s): {', '.join(sorted(self.masks)) or 'None'}",
            )
            # node_labels = {
            #     attrs["label"]
            #     for _, attrs in self.graph_builder.G.nodes(data=True)
            # }

            node_labels = self.scene_understanding.vocabulary

            return video_frame, graph_frame, node_labels

        return None
        

def run_cv2_loop(drone):
    window_name = "Drone Heatmap"

    while drone.has_next():
        result = drone.run()
        if result is None:
            break

        if isinstance(result, tuple):
            frame = result[0]
        else:
            frame = result

        if frame is None:
            break

        cv2.imshow(window_name, _resize_to_screen(frame))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break


def run_app_ui(drone, args):
    from modules.AppUI import AppUI

    ui = AppUI(
        on_submit=drone.ask_graph,
        on_mask_change=drone.set_masks,
    )
    if args.record_ui:
        ui.start_ui_recording(drone.output_dir / "ui_demo.mp4")
    if args.record_graph:
        ui.start_graph_recording(drone.output_dir / "graph_demo.mp4")

    def next_frame():
        if not drone.has_next():
            ui.stop()
            return None

        return drone.run()

    try:
        ui.run_frame_loop(next_frame)

    except Exception:
        traceback.print_exc()

    finally:
        drone.close_video()
        ui.close()

def main():
    args = parse_args()
    dataset_root = args.image_folder or args.dataset_root
    drone = DroneHeatmap(
        dataset_root,
        task=args.task,
        mask=args.mask,
    )

    try:
        if args.ui:
            run_app_ui(drone, args)
        else:
            if args.record_ui or args.record_graph:
                print("--record-ui and --record-graph only apply when --ui is set.")
            run_cv2_loop(drone)

    except Exception:
        traceback.print_exc()

    finally:
        drone.close_video()
        cv2.destroyAllWindows()
        drone.graph_builder.draw_3d_graph()


if __name__ == "__main__":
    main()
