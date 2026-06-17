import numpy as np
import cv2
from dataclasses import dataclass
from scripts.get_masks import get_masks

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

        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.prev_gray = None

    def _prompt_key(self, prompt):
        return str(prompt).strip().casefold()

    def _parse_dict(self, scene_dict):
        prompt_info = {}

        for label, info in scene_dict.items():
            if not isinstance(info, dict):
                continue

            prompts = info.get("prompt", [])
            if isinstance(prompts, str):
                prompts = [prompts]

            for prompt in prompts:
                prompt = str(prompt).strip()
                if not prompt:
                    continue

                prompt_info[prompt] = {
                    "label": str(label),
                    "score": float(info.get("score", 0)),
                }

        return prompt_info

    def _get_flow_map(self, curr_image):
        curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

        flow = self.dis.calc(self.prev_gray, curr_gray, None)

        h, w = curr_image.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        map_x = (x - flow[..., 0]).astype(np.float32)
        map_y = (y - flow[..., 1]).astype(np.float32)

        self.prev_gray = curr_gray

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

        print(f"segmenting {scene_dict}")

        if scene_dict is None: # Not SAM step

            if self.prev_gray is None: return None

            map_x, map_y = self._get_flow_map(image)

            for segmentation in self.segmentations:
                segmentation.mask = cv2.remap(
                    segmentation.mask.astype(np.uint8),  # mask being tracked
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

            prompt_info = self._parse_dict(scene_dict)
            prompts = list(prompt_info.keys())
            prompt_lookup = {
                self._prompt_key(prompt): info
                for prompt, info in prompt_info.items()
            }

            if len(prompts) < 1:
                return None
            
            masks_by_prompt = get_masks(image, prompts)

            self.segmentations = []
            for prompt, masks in masks_by_prompt.items():
                info = (
                    prompt_info.get(prompt)
                    or prompt_lookup.get(self._prompt_key(prompt))
                )
                if info is None:
                    continue

                for mask in masks:
                    self._create_segmentation(
                        mask,
                        info["label"],
                        info["score"],
                    )
        
    

        return self.segmentations
