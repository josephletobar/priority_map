import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import cv2
import pandas as pd
import traceback
from dotenv import load_dotenv

from modules.SceneUnderstanding import SceneUnderstanding
from modules.Heatmap import Heatmap
from modules.Segment import Segment
from modules.GraphBuilder import GraphBuilder
from modules.GraphChat import ChatWithGraph
from scripts.video_output import create_video_writer, release_video_writer

load_dotenv()

class DroneHeatmap: 
    def __init__(self, dataset_root: str, task="Find cars", sam_step=15):
        self.dataset_root = Path(dataset_root)
        self.task = task
        self.sam_step = sam_step
        
        self.query_csv = pd.read_csv(self.dataset_root / "query.csv")

        self.query_images_dir = (self.dataset_root / "query_images")

        self.index = 0

        self.scene_understanding = SceneUnderstanding()
        self.segmentation = Segment()
        self.graph_builder = GraphBuilder()
        self.heatmap = Heatmap()
        self.video_writer = None
        self.video_path = None

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

            image_path = (self.query_images_dir / row["name"])
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            image[:, :, 1] = (image[:, :, 1] * 0.65).astype(image.dtype)
            image[:, :, 2:3] = (image[:, :, 2:3] * 0.8).astype(image.dtype)

            return {
                "image": image,
                "image_path": str(image_path),
                "easting": row["easting"],
                "northing": row["northing"],
                "altitude": row["altitude"],
                "orientation": (
                    row["orient_x"],
                    row["orient_y"],
                    row["orient_z"],
                    row["orient_w"],
                ),
                "frame_index": frame_index,
            }

        return None
    
    def _get_video_writer(self, image):
        if self.video_writer is not None:
            return self.video_writer

        self.video_writer, self.video_path = create_video_writer(image)
        return self.video_writer

    def close_video(self):
        release_video_writer(self.video_writer)
        self.video_writer = None

    def _draw_task_header(self, image):
        output = image.copy()
        text = f"Task: {self.task}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        thickness = 2
        padding_x = 18
        padding_y = 12

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            scale,
            thickness
        )

        header_height = text_height + baseline + padding_y * 2
        overlay = output.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (output.shape[1], header_height),
            (0, 0, 0),
            -1
        )
        output = cv2.addWeighted(overlay, 0.45, output, 0.55, 0)

        cv2.putText(
            output,
            text,
            (padding_x, padding_y + text_height),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        return output

    def show_video(self, image):
        image = self._draw_task_header(image)

        self._get_video_writer(image).write(image)

        cv2.imshow("Frame", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    def run(self):

        if self.has_next():

            frame = self.get_next_frame()
            if frame is None:
                return

            image = frame["image"]
            out = image

            gps = (
                frame["easting"],
                frame["northing"]
            )
            # print(
            #     frame["frame_index"],
            #     gps,
            #     frame["altitude"]
            # )

            scene_dict = None
            if self.should_run_sam(frame):
                scene_dict = self.scene_understanding.get_labels(image, self.task)
            # print(scene_dict)

            segmentations = self.segmentation.get_segmentations(image, scene_dict)

            curr_nodes = self.graph_builder.build_graph(segmentations)

            heatmap = self.heatmap.draw_heatmap(image, segmentations)

            print(f"Frame {frame['frame_index']}:", end=" ")
            for node_id in self.graph_builder.G.nodes:
                print(node_id, end=" ")
            print()

            if heatmap is not None: out = heatmap

            # self.graph_builder.build_graph(self.heatmap.nodes)
            self.graph_builder.draw_2d_graph()

            self.show_video(out)
        

if __name__ == "__main__":

    drone = DroneHeatmap(r"D:\Train\Train")

    try:
        while drone.has_next():
            drone.run()

    except Exception:
        traceback.print_exc()

    finally:
        drone.close_video()
        cv2.destroyAllWindows()

        final_graph = drone.graph_builder.draw_3d_graph()
        
        chat = ChatWithGraph(final_graph)
        while True:
            chat.chat()


