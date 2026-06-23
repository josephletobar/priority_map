import os

from attr import dataclass
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
from pathlib import Path
import cv2
import pandas as pd
import traceback
from dotenv import load_dotenv

from scripts.video_helper import (
    label_mask,
    safe_imwrite,
    VideoOutput,
)
from scripts.cluster_segmentations import cluster_segmentations, ClusteredSegmentation

from modules.SceneUnderstanding import SceneUnderstanding
from modules.Heatmap import Heatmap
from modules.Segment import Segment
from modules.GraphBuilder import GraphBuilder
# from modules.GraphChat import ChatWithGraph
from modules.GeoLocalizer import GeoLocalizer
from modules.PanoramaBuilder import PanoramaBuilder

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
        # default=r"D:\Train\Train\query_images",
        default = r"D:\dronevid2",
        help="Dataset root. Supports plain image folder.",
    )
    parser.add_argument("--task", default="Find cars")
    parser.add_argument("--mask", nargs="*", default=[])
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--record", action="store_true")
    return parser.parse_args()


class DroneHeatmap:
    def __init__(
        self,
        dataset_root: str,
        task="Find cars",
        mask=None,
        sam_step=15,
        show=False,
        record=False,
    ):
        self.dataset_root = Path(dataset_root)
        self.task = task
        self.masks = mask or []
        self.sam_step = sam_step
        
        self.query_csv, self.query_images_dir = self._load_dataset_index()

        self.index = 0
        self.output_dir = Path("examples") / time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scene_understanding = SceneUnderstanding()
        self.segmentation = Segment(show_preview=show)
        self.graph_builder = GraphBuilder(output_dir=self.output_dir)
        self.heatmap = Heatmap()
        self.geo_localizer = GeoLocalizer()
        self.masks = {m.lower() for m in self.masks}
        self.video_output = VideoOutput(
            output_dir=self.output_dir,
            show=show,
            record=record,
        )

        self.heat_panoramic_builder = PanoramaBuilder(alpha=0.9)
        self.panoramic_builder = PanoramaBuilder(alpha=0.5)
        self.show = show

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
        self.video_output.close()
        self.graph_builder.close()

    def run_frame(self):
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
            scene_dict = self.scene_understanding.get_labels(image, self.task)
        # print(scene_dict)

        # Get Segmentations From Image
        segmentations = self.segmentation.get_segmentations(image, scene_dict)
        if segmentations is None:
            segmentations = []

        clustered = cluster_segmentations(segmentations)

        # Set Segmentation Types to Display MASKS
        mask_frame = None
        if self.masks:
            mask_frame = label_mask(self.masks, image, segmentations)

        # Create Heatmap
        heatmap = self.heatmap.draw_heatmap(image, clustered)
        if heatmap is not None:
            out = heatmap

        if scene_dict is not None:
            self.graph_builder.add_nodes(clustered)
            if self.show:
                self.graph_builder.render_2d_graph_frame()
                if self.graph_builder.last_2d_frame is not None:
                    cv2.imshow("2D Graph", self.graph_builder.last_2d_frame)
                    cv2.waitKey(1)

        # Commented out Panorama block for debugging now 
        # # Create Panoramic Images
        # os.makedirs(f"{self.output_dir}/heat_panorama", exist_ok=True)
        # os.makedirs(f"{self.output_dir}/panorama", exist_ok=True)
        # transform = self.segmentation.transform_dx, self.segmentation.transform_dy

        # heat_panorama = self.heat_panoramic_builder.create_panorama(transform, heatmap)
        # # cv2.imshow("Heat Panorama", self.heat_panoramic_builder.panorama)
        # # cv2.waitKey(1)
        # if self.index % 10 == 0:
        #     safe_imwrite(
        #         f"{self.output_dir}/heat_panorama/heat_panorama_{self.index}.png",
        #         heat_panorama,
        #     )

        # panorama = self.panoramic_builder.create_panorama(transform, image)
        # # cv2.imshow("Panorama", self.panoramic_builder.panorama)
        # # cv2.waitKey(1)
        # if self.index % 10 == 0:
        #     safe_imwrite(f"{self.output_dir}/panorama/panorama_{self.index}.png", panorama)

        return self.video_output.handle_frame(
            out,
            header=f"Task: {self.task}",
            side_image=mask_frame,
            side_header=f"Mask(s): {', '.join(sorted(self.masks)) or 'None'}",
        )

    def run(self):
        while self.has_next():
            if not self.run_frame():
                break

# def run_graph_chat(drone):
#     graph = drone.graph_builder._build_topology().copy()
#     chat = ChatWithGraph(graph)
#     chat.run()


def main():
    args = parse_args()
    dataset_root = args.image_folder or args.dataset_root
    drone = DroneHeatmap(
        dataset_root,
        task=args.task,
        mask=args.mask,
        show=args.show,
        record=args.record,
    )

    try:
        drone.run()

    except Exception:
        traceback.print_exc()

    finally:
        drone.close_video()
        cv2.destroyAllWindows()
        # if args.chat:
        #     run_graph_chat(drone)
        # drone.graph_builder.draw_3d_graph()


if __name__ == "__main__":
    main()
