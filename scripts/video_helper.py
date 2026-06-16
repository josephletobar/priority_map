import cv2
import numpy as np
from pathlib import Path


def video_path(output_dir="example", filename="video.mp4"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / filename


def create_video_writer(image, output_dir="examples", filename="video.mp4", fps=30):
    output_path = video_path(output_dir, filename)

    height, width = image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not video_writer.isOpened():
        video_writer.release()
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    return video_writer, output_path


def release_video_writer(video_writer):
    if video_writer is not None:
        video_writer.release()


def get_video_writer(video_writer, image, output_dir, filename="video.mp4"):
    if video_writer is not None:
        return video_writer, None

    return create_video_writer(
        image,
        output_dir=output_dir,
        filename=filename,
    )


def draw_header(image, text):
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2
    padding_x = 18
    padding_y = 12

    (_, text_height), baseline = cv2.getTextSize(
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


def stack_side_by_side(image, side_image):
    if side_image.shape[0] != image.shape[0]:
        scale = image.shape[0] / side_image.shape[0]
        side_image = cv2.resize(
            side_image,
            (int(side_image.shape[1] * scale), image.shape[0])
        )

    return np.hstack([image, side_image])


def compose_video_frame(image, header, side_image=None, side_header=None):
    image = draw_header(image, header)

    if side_image is not None:
        side_image = draw_header(side_image, side_header or header)
        image = stack_side_by_side(image, side_image)

    return image
