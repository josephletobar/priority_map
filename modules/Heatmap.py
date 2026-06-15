from ultralytics.models.sam import SAM3SemanticPredictor
import numpy as np
import cv2
from dataclasses import dataclass
from scripts.heatmap_helper import merge_similar
from ultralytics import YOLOWorld

@dataclass
class Region:
    mask: np.ndarray
    label: str
    id: str
    score: float   

Node = Region


class Heatmap:
    def __init__(self):

        self.regions = []
        self.heat_gamma = 10.0

        overrides = dict(
            conf=0.5,
            task="segment",
            mode="predict",
            model="models/sam3.pt",
            half=True,  # Use FP16 for faster inference
            save=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.prev_gray = None

        # self.model = YOLOWorld("yolov8x-worldv2.pt")
        # self.model.set_classes(["forest", "trees", "field", "grass"])

        self.nodes = []

    def _parse_dict(self, scene_dict):
        prompts = list(scene_dict.keys())

        return prompts

    def _get_flow_map(self, curr_image):
        curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

        flow = self.dis.calc(self.prev_gray, curr_gray, None)

        h, w = curr_image.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        map_x = (x - flow[..., 0]).astype(np.float32)
        map_y = (y - flow[..., 1]).astype(np.float32)

        self.prev_gray = curr_gray

        return map_x, map_y
    
    
    def _create_region(self, mask: np.ndarray, label, score):
        self.regions.append(
            Region(
                mask=mask,
                label=label,
                score=score,
                id=""
            )
        )

    def _create_nodes(self):
        for region in self.regions:
            self.nodes.append(
                Node(
                    mask=region.mask > 0,
                    label=region.label,
                    score=region.score,
                    id=""
                )
            )

        self.nodes = merge_similar(self.nodes)

        counts = {}
        for node in self.nodes:
            idx = counts.get(node.label, 0)
            counts[node.label] = idx + 1
            node.id = f"{node.label}_{idx}"


    def _create_heatmap(self, image):
        heatmap = np.zeros(image.shape[:2], dtype=np.float32)
        valid = np.zeros(image.shape[:2], dtype=np.float32)

        for region in self.regions:
            mask = region.mask.astype(np.float32)
            score = region.score

            boosted_score = (score / 100) ** self.heat_gamma

            heatmap = np.maximum(
                heatmap,
                mask * boosted_score
            )

            valid = np.maximum(
                valid,
                mask
            )

        spread = (301, 301)
        sigma = 0
        heatmap = cv2.GaussianBlur(heatmap, spread, sigma)
        valid = cv2.GaussianBlur(valid, spread, sigma)
        heatmap = heatmap / (valid + 1e-6)
        heatmap = np.power(heatmap, 1 / self.heat_gamma) * 100

        heatmap = np.clip(heatmap, 0, 100)
        heatmap = (heatmap * 2.55).astype(np.uint8)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        output = cv2.addWeighted(
            image,    # base image
            0.6,      # weight of base image
            heatmap,  # heatmap overlay image
            0.4,      # weight of heatmap
            0         # constant brightness offset added to every pixel
        )

        return output

    def _draw_node_labels(self, image):
        output = image.copy()

        for node in self.nodes:
            mask = node.mask > 0
            if not np.any(mask):
                continue

            ys, xs = np.nonzero(mask)
            x = int(xs.mean())
            y = int(ys.mean())

            cv2.putText(
                output,
                node.label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 100, 100),
                2,
                cv2.LINE_AA,
            )

        return output

    def get(self, image, scene_dict):
 
        # results = self.model.predict(image)
        # annotated = results[0].plot()
        # return annotated

        if scene_dict is None: # Not SAM step

            if self.prev_gray is None: return None

            map_x, map_y = self._get_flow_map(image)

            for region in self.regions:
                region.mask = cv2.remap(
                    region.mask.astype(np.uint8),  # mask being tracked
                    map_x,                         # x-coordinate lookup table from optical flow
                    map_y,                         # y-coordinate lookup table from optical flow
                    interpolation=cv2.INTER_NEAREST,  # preserve binary mask values (0/1)
                    borderMode=cv2.BORDER_CONSTANT,   # pixels outside image become a constant value
                    borderValue=0                    # outside-image pixels become background
                )

            for node in self.nodes:
                node.mask = cv2.remap(
                    node.mask.astype(np.uint8),
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

        else: # SAM step
            self.prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if scene_dict is None:
                return None

            prompts = self._parse_dict(scene_dict)

            if len(prompts) < 1:
                return None
        
            results = self.predictor(image, text=prompts)
            if not results: return None
            result = results[0]

            if result.masks is None: return None
            masks = result.masks.data.cpu().numpy()  # (N, H, W)

            self.regions = []
            for i in range(len(result.boxes)):
                prompt = result.names[int(result.boxes.cls[i])]
                mask = masks[i]
                score = scene_dict[prompt]["score"]
                label = scene_dict[prompt]["label"]

                self._create_region(mask, label, score)

                # annotated = result.plot()
                # return annotated

            self._create_nodes()

        heatmap = self._create_heatmap(image)
        heatmap = self._draw_node_labels(heatmap)

        return heatmap
