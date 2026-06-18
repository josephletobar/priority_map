import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
import cv2
from dataclasses import dataclass
import time 

FLOW_SCALE = 0.05

@dataclass
class Segmentation:
    mask: np.ndarray
    label: str
    id: str
    score: float   
    geo_pos: tuple[float, float, float] | None = None
    
class Segment():

    def __init__(self):

        self.segmentations = []

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
        prompts = []
        prompt_to_label = {}

        for label, label_info in scene_dict.items():
            label_prompts = label_info.get("prompt", [])
            if isinstance(label_prompts, str):
                label_prompts = [label_prompts]

            for prompt in label_prompts:
                prompt = str(prompt).strip()
                if not prompt:
                    continue

                prompts.append(prompt)
                prompt_to_label[prompt] = label

        return prompts, prompt_to_label

    def _get_flow_map(self, curr_image):
        curr_gray_full = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

        curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        h, w = curr_gray.shape[:2]
        flow_size = (
            max(1, int(w * FLOW_SCALE)),
            max(1, int(h * FLOW_SCALE)),
        )

        curr_gray = cv2.resize(curr_gray, flow_size, interpolation=cv2.INTER_AREA)
        prev_gray = cv2.resize(self.prev_gray, flow_size, interpolation=cv2.INTER_AREA)

        t0 = time.perf_counter()
        flow = self.dis.calc(curr_gray, prev_gray, None)
        # print(f"DIS flow: {(time.perf_counter() - t0) * 1000:.2f} ms")

        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] /= FLOW_SCALE
        flow[..., 1] /= FLOW_SCALE

        x, y = np.meshgrid(np.arange(w), np.arange(h))

        map_x = (x + flow[..., 0]).astype(np.float32)
        map_y = (y + flow[..., 1]).astype(np.float32)

        self.prev_gray = curr_gray_full

        return map_x, map_y
    
    
    def _create_segmentation(self, mask: np.ndarray, label, score):
        self.segmentations.append(
            Segmentation(
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

            for segmentation in self.segmentations:
                segmentation.mask = cv2.remap(
                    segmentation.mask.astype(np.uint8),  # mask being tracked
                    map_x,                            # x-coordinate lookup table from optical flow
                    map_y,                            # y-coordinate lookup table from optical flow
                    interpolation=cv2.INTER_NEAREST,  # preserve binary mask values (0/1)
                    borderMode=cv2.BORDER_CONSTANT,   # pixels outside image become a constant value
                    borderValue=0                    # outside-image pixels become background
                )


        else: # SAM step
            self.prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if scene_dict is None:
                return None

            prompts, prompt_to_label = self._parse_dict(scene_dict)

            if len(prompts) < 1:
                return None
        
            results = self.predictor(image, text=prompts)
            if not results: return None
            result = results[0]

            if result.masks is None: return None
            masks = result.masks.data.cpu().numpy()  # (N, H, W)

            self.segmentations = []
            for i in range(len(result.boxes)):
                prompt = result.names[int(result.boxes.cls[i])]
                if prompt not in prompt_to_label:
                    continue

                mask = masks[i]
                label = prompt_to_label[prompt]
                score = scene_dict[label]["score"]

                self._create_segmentation(mask, label, score)

                # annotated = result.plot()
                # return annotated

            # self._create_nodes()


        return self.segmentations
