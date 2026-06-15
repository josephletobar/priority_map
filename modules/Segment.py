import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
import cv2
from dataclasses import dataclass

@dataclass
class Region:
    mask: np.ndarray
    label: str
    id: str
    score: float   
    
class Segment():

    def __init__(self):

        self.regions = []

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

    def get_segmentations(self, image, scene_dict):
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

            # self._create_nodes()

        return self.regions