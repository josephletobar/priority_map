import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.transforms import InterpolationMode
from ultralytics import SAM

from config.prompts import PROMPT_TEMPLATES


ROOT = Path(__file__).resolve().parents[1]
CLIP_SURGERY_ROOT = ROOT / "external" / "CLIP_Surgery"
if CLIP_SURGERY_ROOT.exists():
    sys.path.insert(0, str(CLIP_SURGERY_ROOT))

import clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = os.getenv("CLIP_MODEL", "CS-ViT-B/16")
SAM_MODEL = os.getenv("SAM_MODEL", str(ROOT / "models" / "mobile_sam.pt"))

clip_model, _ = clip.load(CLIP_MODEL, device=DEVICE)
clip_model.eval()
clip_input_size = clip_model.visual.input_resolution

preprocess = Compose([
    Resize((clip_input_size, clip_input_size), interpolation=InterpolationMode.BICUBIC),
    ToTensor(),
    Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
])

sam = SAM(SAM_MODEL)


def _to_clip_tensor(image: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    return preprocess(pil_image).unsqueeze(0).to(DEVICE)


def _predict_sam_masks(
    image: np.ndarray,
    point_coords,
    point_labels,
) -> list[np.ndarray]:
    point_coords = np.asarray(point_coords, dtype=np.float32)
    point_labels = np.asarray(point_labels, dtype=np.int32)

    if point_coords.size == 0:
        return []

    sam_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = sam.predict(
        source=sam_image,
        points=[point_coords.tolist()],
        labels=[point_labels.tolist()],
        retina_masks=True,
        device=DEVICE,
        verbose=False,
    )

    masks = results[0].masks
    if masks is None or len(masks.data) == 0:
        return []

    return [
        mask.detach().cpu().numpy().astype(np.uint8)
        for mask in masks.data
    ]


def _points_to_hull_mask(image_shape, points) -> np.ndarray | None:
    points = np.asarray(points, dtype=np.int32)
    if len(points) < 3:
        return None

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


def get_masks(
    image: np.ndarray,
    texts: list[str],
    threshold: float = 0.4,
) -> dict[str, list[np.ndarray]]:
    texts = [text.strip() for text in texts if text and text.strip()]
    if image is None or not texts:
        return {}

    image_tensor = _to_clip_tensor(image)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = clip.encode_text_with_prompt_ensemble(
            clip_model,
            texts,
            DEVICE,
            prompt_templates=PROMPT_TEMPLATES,
        )
        redundant_features = clip.encode_text_with_prompt_ensemble(
            clip_model,
            [""],
            DEVICE,
            prompt_templates=PROMPT_TEMPLATES,
        )
        similarity = clip.clip_feature_surgery(
            image_features,
            text_features,
            redundant_features,
        )[0]

    masks_by_text = {}
    for idx, text in enumerate(texts):
        points, labels = clip.similarity_map_to_points(
            similarity[1:, idx],
            image.shape[:2],
            t=threshold,
            down_sample=1,
        )

        labels = np.asarray(labels)
        pos_points = np.asarray(
            [point for point, label in zip(points, labels) if label == 1],
            dtype=np.int32,
        )

        if len(pos_points) > 50:
            hull_mask = _points_to_hull_mask(image.shape, pos_points)
            masks = [] if hull_mask is None else [hull_mask]
        else:
            masks = _predict_sam_masks(image, points, labels)

        masks_by_text[text] = masks

    return masks_by_text
