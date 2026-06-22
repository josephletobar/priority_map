import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
import cv2
import ctypes
from dataclasses import dataclass
import time 
from config.prompts import PROMPT_TEMPLATES
from modules.PanoramaBuilder import PanoramaBuilder

FLOW_SCALE = 0.05
SAM3_PREVIEW_MARGIN = 120
SAM3_INFERENCE_SIZE = (720, 480)

PROMPT_TEMPLATE = "{prompt}"


def _screen_size(default=(1280, 720)):
    try:
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return default


def _resize_to_screen(image, margin=SAM3_PREVIEW_MARGIN):
    if image is None:
        return None

    screen_width, screen_height = _screen_size()
    max_width = max(1, screen_width - margin)
    max_height = max(1, screen_height - margin)
    height, width = image.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)

    if scale >= 1.0:
        return image

    return cv2.resize(
        image,
        (
            max(1, int(width * scale)),
            max(1, int(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )


def _resize_for_sam(image):
    height, width = image.shape[:2]
    if (width, height) == SAM3_INFERENCE_SIZE:
        return image

    interpolation = cv2.INTER_AREA if width > SAM3_INFERENCE_SIZE[0] or height > SAM3_INFERENCE_SIZE[1] else cv2.INTER_LINEAR
    return cv2.resize(image, SAM3_INFERENCE_SIZE, interpolation=interpolation)

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
            conf=0.3,
            task="segment",
            mode="predict",
            model="models/sam3.pt",
            half=True,  # Use FP16 for faster inference
            save=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.prev_gray = None

        self.transform_dx = 0
        self.transform_dy = 0

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

                templated_prompt = PROMPT_TEMPLATE.format(prompt=prompt)
                prompts.append(templated_prompt)
                prompt_to_label[templated_prompt] = label

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

        self.transform_dx = float(np.median(flow[..., 0]))
        self.transform_dy = float(np.median(flow[..., 1]))

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
        image_height, image_width = image.shape[:2]
        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute optical flow on every frame (if we have previous frame)
        if self.prev_gray is not None:
            map_x, map_y = self._get_flow_map(image)

            for segmentation in self.segmentations:
                segmentation.mask = cv2.remap(
                    segmentation.mask.astype(np.uint8),
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

        # Run SAM if this is a SAM step
        if scene_dict is not None:
            prompts, prompt_to_label = self._parse_dict(scene_dict)

            if len(prompts) >= 1:
                sam_image = _resize_for_sam(image)
                results = self.predictor(sam_image, text=prompts)
                if results:
                    result = results[0]

                    annotated = result.plot()
                    cv2.imshow("SAM3", _resize_to_screen(annotated))
                    cv2.waitKey(1)

                    if result.masks is not None:
                        masks = result.masks.data.cpu().numpy()  # (N, H, W)

                        self.segmentations = []
                        for i in range(len(result.boxes)):
                            prompt = result.names[int(result.boxes.cls[i])]
                            if prompt not in prompt_to_label:
                                continue

                            mask = masks[i]
                            mask = mask.astype(np.uint8)
                            if mask.shape[:2] != (image_height, image_width):
                                mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

                            label = prompt_to_label[prompt]
                            score = scene_dict[label]["score"]

                            self._create_segmentation(mask, label, score)

        self.prev_gray = curr_gray
        return self.segmentations
