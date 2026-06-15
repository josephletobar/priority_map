import os
from pathlib import Path
import time

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "001024.png"

HF_HOME = Path("D:/huggingface")
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
os.environ["HF_MODULES_CACHE"] = str(HF_HOME / "modules")

MODEL_ID = "allenai/Molmo2-4B"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

DEVICE = "cuda:0"
DTYPE = torch.float16

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


log(f"CUDA device: {torch.cuda.get_device_name(0)}")
print_mem("Before load")

log("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

log("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=DTYPE,
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

inputs = {
    k: v.to(model.device) if hasattr(v, "to") else v
    for k, v in inputs.items()
}

log("Generating...")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

generated_tokens = output[0, inputs["input_ids"].shape[1]:]

generated_text = processor.decode(
    generated_tokens,
    skip_special_tokens=True,
)

print("\n=== RESPONSE ===")
print(generated_text)