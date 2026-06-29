import argparse
import csv
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


CHECKPOINT_DIR = "allenai/MolmoPoint-8B"
DEFAULT_DATASET_ROOT = Path(os.environ.get("DRONE_DATASET_ROOT", r"C:\Users\jleto\Downloads\Train\Train"))
PROMPT = "Point to the roads, buildings, vehicles, and other mission-relevant regions."
OUTPUT_DIR = Path("outputs/molmo")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MODEL_DTYPE = torch.bfloat16
MIN_RECOMMENDED_RAM_GB = 24.0


def get_total_ram_gb():
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size / (1024**3)
        except (ValueError, OSError, AttributeError):
            pass

    return None


def print_system_info():
    cpu = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "Unknown"
    ram_gb = get_total_ram_gb()
    ram_text = f"{ram_gb:.2f} GB" if ram_gb is not None else "Unknown"

    print("System information:")
    print(f"  CPU: {cpu}")
    print(f"  RAM: {ram_text}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MolmoPoint on randomly sampled drone dataset images using CPU only."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root containing query.csv and query_images/. Can also be set with DRONE_DATASET_ROOT.",
    )
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_DIR,
        help="MolmoPoint checkpoint name or local checkpoint directory.",
    )
    parser.add_argument(
        "--allow-low-ram",
        action="store_true",
        help="Attempt model loading even if RAM is below the recommended threshold.",
    )
    return parser.parse_args()


def has_enough_ram_for_cpu_load(checkpoint_dir, allow_low_ram=False):
    ram_gb = get_total_ram_gb()
    if ram_gb is None:
        return True

    if ram_gb >= MIN_RECOMMENDED_RAM_GB:
        return True

    if allow_low_ram:
        print(
            f"Only {ram_gb:.2f} GB RAM detected, below the recommended "
            f"{MIN_RECOMMENDED_RAM_GB:.0f} GB. Attempting load anyway because "
            "--allow-low-ram was set."
        )
        return True

    print(
        f"Only {ram_gb:.2f} GB RAM detected. {checkpoint_dir} is likely too large "
        f"for CPU-only loading without quantization; even {MODEL_DTYPE} weights need "
        "substantial memory during and after checkpoint load."
    )
    print(
        "Skipping model load to avoid the operating system terminating Python. "
        "Run on a machine with more RAM, use a smaller MolmoPoint checkpoint if available, "
        "or explicitly re-enable quantization/offload for this experiment."
    )
    return False


def discover_images_recursively(root_dir):
    images = []

    for path in root_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        images.append(path)

    return images


def discover_dataset_images(dataset_root):
    query_csv = dataset_root / "query.csv"
    query_images_dir = dataset_root / "query_images"
    images = []

    if query_csv.exists():
        with query_csv.open(newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            if "name" not in (reader.fieldnames or []):
                raise ValueError(f"{query_csv} does not contain a 'name' column.")

            for row in reader:
                image_name = row.get("name")
                if not image_name:
                    continue

                image_path = query_images_dir / image_name
                if image_path.suffix.lower() in IMAGE_EXTENSIONS and image_path.exists():
                    images.append(image_path)

    if images:
        return images

    if query_images_dir.exists():
        images = discover_images_recursively(query_images_dir)

    if images:
        return images

    if dataset_root.exists():
        return discover_images_recursively(dataset_root)

    return []


def select_images(images):
    if len(images) <= 5:
        selected_count = len(images)
    else:
        selected_count = random.randint(3, 5)

    if selected_count < 3:
        print(f"Found only {selected_count} image(s); processing all available images.")

    return random.sample(images, selected_count)


def points_to_array(points):
    if points is None:
        return np.empty((0, 4))

    point_array = np.array(points)
    if point_array.size == 0:
        return np.empty((0, 4))

    return np.atleast_2d(point_array)


def point_to_pixels(x, y, width, height):
    x = float(x)
    y = float(y)

    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return int(round(x * width)), int(round(y * height))

    if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0:
        return int(round(x / 100.0 * width)), int(round(y / 100.0 * height))

    return int(round(x)), int(round(y))


def save_visualization(image, points, image_path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    radius = max(4, min(width, height) // 160)

    for point in points_to_array(points):
        if len(point) < 4:
            continue

        object_id, _image_num, x, y = point[:4]
        px, py = point_to_pixels(x, y, width, height)
        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))
        label = str(object_id)

        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill="red",
            outline="white",
            width=max(1, radius // 2),
        )
        draw.text((px + radius + 2, py + radius + 2), label, fill="yellow")

    output_path = OUTPUT_DIR / f"{image_path.stem}_molmo_points.png"
    annotated.save(output_path)
    return output_path


def remove_quantization_config(config):
    for target in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "vision_config", None),
    ):
        if target is None:
            continue

        for attr in ("quantization_config", "_pre_quantization_dtype"):
            if hasattr(target, attr):
                delattr(target, attr)


def load_cpu_model(checkpoint_dir):
    config = AutoConfig.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
    )
    remove_quantization_config(config)

    return AutoModelForImageTextToText.from_pretrained(
        checkpoint_dir,
        config=config,
        trust_remote_code=True,
        device_map="cpu",
        dtype=MODEL_DTYPE,
        low_cpu_mem_usage=True,
        quantization_config=None,
    )


def run_inference_on_image(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image", "image": image},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        image_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
        return_pointing_metadata=True,
    )

    metadata = inputs.pop("metadata")
    inputs = {
        k: v.to("cpu") if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    start_time = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            logits_processor=model.build_logit_processor_from_inputs(inputs),
            max_new_tokens=200,
        )
    inference_time = time.perf_counter() - start_time

    generated_tokens = output[:, inputs["input_ids"].size(1):]
    generated_text = processor.post_process_image_text_to_text(
        generated_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]

    points = model.extract_image_points(
        generated_text,
        metadata["token_pooling"],
        metadata["subpatch_mapping"],
        metadata["image_sizes"],
    )

    visualization_path = save_visualization(image, points, image_path)

    return generated_text, points, inference_time, visualization_path


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser()
    checkpoint_dir = args.checkpoint

    images = discover_dataset_images(dataset_root)

    if not images:
        print(f"No jpg, jpeg, or png images were found from the dataset: {dataset_root}")
        return

    selected_images = select_images(images)

    print_system_info()
    if not has_enough_ram_for_cpu_load(checkpoint_dir, args.allow_low_ram):
        return

    print(f"Loading {checkpoint_dir} on CPU...")

    model = load_cpu_model(checkpoint_dir)

    processor = AutoProcessor.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        padding_side="left",
    )

    print(f"Dataset root: {dataset_root}")
    print(f"Processing {len(selected_images)} image(s) sampled from the query image source:")
    for image_path in selected_images:
        print(f"  - {image_path}")
    print()

    for image_path in selected_images:
        print(f"Image filename: {image_path.name}")
        try:
            generated_text, points, inference_time, visualization_path = run_inference_on_image(
                model,
                processor,
                image_path,
            )

            print("Generated text:")
            print(generated_text)
            print("Extracted points:")
            print(points_to_array(points))
            print(f"Inference time: {inference_time:.2f} seconds")
            print(f"Visualization: {visualization_path}")
        except Exception as exc:
            print(f"Failed to process {image_path}: {exc}")
        finally:
            print()


if __name__ == "__main__":
    main()
