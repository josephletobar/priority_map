import os

from attr import dataclass
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import atexit
import argparse
import signal
import time
from pathlib import Path
import cv2
import pandas as pd
import traceback
import csv
from dotenv import load_dotenv
import config.params as config
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
from modules.GraphAgent import GraphAgent
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
        # default = r"C:\Users\jletobar3\Projects\dronevid2",
        default = r"D:\UAV_VisLoc_dataset\05\drone",
        help="Dataset root. Supports plain image folder.",
    )
    parser.add_argument("--task", default="Find cars")
    parser.add_argument("--debrief", default = "debrief.txt")
    parser.add_argument("--mask", nargs="*", default=[])
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--panoramic", action="store_true")
    parser.add_argument(
        "--graph_agent",
        "--graph-agent",
        action="store_true",
        help="Enable the asynchronous graph agent. Off by default.",
    )
    return parser.parse_args()


class DroneHeatmap:
    def __init__(
        self,
        dataset_root: str,
        task="Find cars",
        debrief="debrief.txt",
        mask=None,
        sam_step=config.SAM_STEP,
        show=False,
        record=False,
        panoramic=False,
        graph_agent=False,
    ):
        self.dataset_root = Path(dataset_root)
        self.task = task
        self.debrief = debrief
        self.masks = mask or []
        self.sam_step = sam_step
        self.panoramic = panoramic
        
        self.query_csv, self.query_images_dir = self._load_dataset_index()

        self.index = 0
        self.output_dir = Path("examples") / time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scene_understanding = SceneUnderstanding()
        self.segmentation = Segment(show_preview=show)
        self.graph_builder = GraphBuilder(output_dir=self.output_dir)
        self.graph_agent = (
            GraphAgent(self.graph_builder, f"{self.task}: {self.debrief}")
            if graph_agent
            else None
        )
        self.heatmap = Heatmap()
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
    
    def close_video(self):
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
            scene_dict = self.scene_understanding.get_labels(image, f"{self.task}: {self.debrief}")
        # print(scene_dict)

        # Get Segmentations From Image
        segmentations = self.segmentation.get_segmentations(image, scene_dict)
        if segmentations is None:
            segmentations = []

        clustered = cluster_segmentations(segmentations)

        # Set Where to Save Observations CSV
        csv_path = Path(f"{self.output_dir}/observations.csv")
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
        os.makedirs(f"{self.output_dir}/heatmap_imgs", exist_ok=True)
        safe_imwrite(f"{self.output_dir}/heatmap_imgs/{self.image_name}", heatmap_only)

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
            os.makedirs(f"{self.output_dir}/heat_panorama", exist_ok=True)
            os.makedirs(f"{self.output_dir}/panorama", exist_ok=True)
            transform = self.segmentation.transform_dx, self.segmentation.transform_dy

            # Base image panorama
            panorama = self.panoramic_builder.create_panorama(transform, image)
            if self.index % panorama_save == 0:
                safe_imwrite(f"{self.output_dir}/panorama/panorama_{self.index}.png", panorama)

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
                    safe_imwrite(f"{self.output_dir}/heat_panorama/heat_panorama_{self.index}.png", heat_panorama)


        # Write Output Frame To Video

        keep_running = self.video_output.handle_frame(
            out,
            header=f"Task: {self.task} | Graph Level {1 if self.graph_view == 'semantic' else 2}",
            side_image=mask_frame,
            side_header=f"Mask(s): {', '.join(sorted(self.masks)) or 'None'}",
        )
        self._handle_graph_view_key()
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

# def run_graph_chat(drone):
#     graph = drone.graph_builder._build_topology().copy()
#     chat = ChatWithGraph(graph)
#     chat.run()


def main():

    t0 = time.perf_counter()
    drone = None
    cleaned_up = False

    def cleanup():
        nonlocal cleaned_up
        if cleaned_up:
            return

        cleaned_up = True
        if drone is not None:
            drone.close_video()
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except cv2.error as exc:
            print(f"Cleanup warning (OpenCV windows): {exc}")

    atexit.register(cleanup)
    previous_sigint = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum, frame):
        print("\nInterrupted. Cleaning up video resources...")
        cleanup()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)
    
    try:
        args = parse_args()
        dataset_root = args.image_folder or args.dataset_root
        drone = DroneHeatmap(
            dataset_root,
            task=args.task,
            debrief=args.debrief,
            mask=args.mask,
            show=args.show,
            record=args.record,
            panoramic=args.panoramic,
            graph_agent=args.graph_agent,
        )

        drone.run()

    except KeyboardInterrupt:
        pass

    except Exception:
        traceback.print_exc()

    finally:
        cleanup()
        signal.signal(signal.SIGINT, previous_sigint)
        try:
            atexit.unregister(cleanup)
        except ValueError:
            pass
        # if args.chat:
        #     run_graph_chat(drone)
        # drone.graph_builder.draw_3d_graph()

        print(f"Total time: {(time.perf_counter() - t0):.2f} seconds")


if __name__ == "__main__":
    main()
