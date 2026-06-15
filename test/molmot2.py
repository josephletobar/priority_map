import os
from pathlib import Path
import re
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

try:
    import bitsandbytes as bnb
except ImportError as exc:
    raise RuntimeError(
        "bitsandbytes is required for 4-bit Molmo loading. "
        "Install it with: pip install bitsandbytes"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "001024.png"

HF_HOME = Path("D:/huggingface")
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
os.environ["HF_MODULES_CACHE"] = str(HF_HOME / "modules")

MODEL_ID = r"D:\models\Molmo2-4B"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

DEVICE = "cuda:0"

START_TIME = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - START_TIME:7.2f}s] {msg}", flush=True)


def print_mem(label):
    free_b, total_b = torch.cuda.mem_get_info()
    gib = 1024**3
    print(
        f"{label}: free={free_b/gib:.2f} GiB, "
        f"total={total_b/gib:.2f} GiB"
    )


def extract_points(text):
    match = re.search(r'<points\s+coords="([^"]+)"', text)
    if match is None:
        return []

    values = [
        float(value)
        for value in re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
    ]

    if len(values) % 2 == 1:
        values = values[:-1]

    return list(zip(values[0::2], values[1::2]))


def show_points(image_rgb, points):
    output = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
    height, width = output.shape[:2]

    for index, (x, y) in enumerate(points, start=1):
        px = int(round((x / 1000.0) * width))
        py = int(round((y / 1000.0) * height))
        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))

        cv2.circle(output, (px, py), 8, (0, 0, 255), -1)
        cv2.circle(output, (px, py), 11, (255, 255, 255), 2)
        cv2.putText(
            output,
            str(index),
            (px + 12, py - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Molmo points", output)
    cv2.waitKey(0)
    cv2.destroyWindow("Molmo points")


log(f"CUDA device: {torch.cuda.get_device_name(0)}")
print_mem("Before load")

log("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

log("Loading model...")
log(f"bitsandbytes version: {bnb.__version__}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_skip_modules=["vision_backbone"],
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)

model.eval()

log("Model loaded.")
print_mem("After load")

image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": "Point to the roads in this aerial image."
            },
        ],
    }
]

log("Preparing inputs...")

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

for key, value in inputs.items():
    if isinstance(value, torch.Tensor):
        print(f"Input {key}: shape={tuple(value.shape)}, dtype={value.dtype}")

inputs = {
    k: (
        v.to(DEVICE, dtype=torch.float16)
        if isinstance(v, torch.Tensor) and k.startswith("pixel")
        else v.to(DEVICE)
        if isinstance(v, torch.Tensor)
        else v
    )
    for k, v in inputs.items()
}

log("Generating...")

infer_start = time.perf_counter()

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

infer_time = time.perf_counter() - infer_start

generated_tokens = output[0, inputs["input_ids"].shape[1]:]
num_tokens = len(generated_tokens)

print(f"\nInference time: {infer_time:.2f}s")
print(f"Generated tokens: {num_tokens}")
print(f"Tokens/sec: {num_tokens / infer_time:.2f}")

generated_text = processor.decode(
    generated_tokens,
    skip_special_tokens=True,
)

print(f"Total runtime: {time.perf_counter() - START_TIME:.2f}s")
print("\n=== RESPONSE ===")
print(generated_text)

points = extract_points(generated_text)
print(f"Parsed points: {points}")
show_points(image, points)
