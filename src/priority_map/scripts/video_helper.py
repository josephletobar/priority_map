import cv2
import ctypes
import numpy as np
from pathlib import Path

MASK_EDGE_BLUR = 51
PREVIEW_MARGIN = 120

def video_path(output_dir="example", filename="video.avi"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / filename


def _video_writer_candidates(filename):
    path = Path(filename)
    suffix = path.suffix.lower()

    if suffix == ".mp4":
        avi_filename = f"{path.stem}.avi"
        return [
            (filename, "avc1"),
            (filename, "H264"),
            (avi_filename, "MJPG"),
        ]

    if suffix == ".avi":
        return [
            (filename, "MJPG"),
            (filename, "XVID"),
        ]

    return [
        ("video.avi", "MJPG"),
        ("video.avi", "XVID"),
    ]


def _prepare_video_frame(image, frame_size=None):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if frame_size is None:
        height, width = image.shape[:2]
        pad_bottom = height % 2
        pad_right = width % 2
        if pad_bottom or pad_right:
            image = cv2.copyMakeBorder(
                image,
                0,
                pad_bottom,
                0,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
    else:
        width, height = frame_size
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(image)


def create_video_writer(image, output_dir="examples", filename="video.avi", fps=30, debug=False):
    image = _prepare_video_frame(image)
    height, width = image.shape[:2]
    frame_size = (width, height)
    failed = []

    for candidate_filename, codec in _video_writer_candidates(filename):
        output_path = video_path(output_dir, candidate_filename)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

        if video_writer.isOpened():
            if debug:
                print(f"Recording video to {output_path} using {codec}")
            return video_writer, output_path, frame_size

        video_writer.release()
        failed.append(f"{output_path} ({codec})")
        output_path.unlink(missing_ok=True)

    raise RuntimeError(f"Failed to open video writer. Tried: {', '.join(failed)}")


def release_video_writer(video_writer):
    if video_writer is not None:
        video_writer.release()


def get_video_writer(video_writer, image, output_dir, filename="video.avi", frame_size=None, debug=False):
    if video_writer is not None:
        return (
            video_writer,
            None,
            frame_size,
            _prepare_video_frame(image, frame_size=frame_size),
        )

    video_writer, output_path, frame_size = create_video_writer(
        image,
        output_dir=output_dir,
        filename=filename,
        debug=debug,
    )
    return (
        video_writer,
        output_path,
        frame_size,
        _prepare_video_frame(image, frame_size=frame_size),
    )


def screen_size(default=(1280, 720)):
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def resize_to_screen(image, margin=PREVIEW_MARGIN):
    if image is None:
        return None

    screen_width, screen_height = screen_size()
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


def _handle_video_frame(
    image,
    header,
    *,
    side_image=None,
    side_header=None,
    show=False,
    record=False,
    output_dir="examples",
    filename="video.avi",
    video_writer=None,
    video_frame_size=None,
    window_name=None,
    margin=PREVIEW_MARGIN,
    debug=False,
):
    key = -1
    image = compose_video_frame(
        image,
        header,
        side_image=side_image,
        side_header=side_header,
    )

    video_path = None
    if record:
        video_writer, video_path, video_frame_size, video_frame = get_video_writer(
            video_writer,
            image,
            output_dir,
            filename=filename,
            frame_size=video_frame_size,
            debug=debug,
        )
        video_writer.write(video_frame)

    if show:
        cv2.imshow(window_name or header, resize_to_screen(image, margin=margin))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return image, video_writer, video_path, video_frame_size, False, key

    return image, video_writer, video_path, video_frame_size, True, key


class VideoOutput:
    def __init__(
        self,
        output_dir="examples",
        show=False,
        record=False,
        filename="video.avi",
        window_name=None,
        margin=PREVIEW_MARGIN,
        debug=False,
    ):
        self.output_dir = output_dir
        self.show = show
        self.record = record
        self.filename = filename
        self.window_name = window_name
        self.margin = margin
        self.debug = debug
        self.video_writer = None
        self.video_path = None
        self.video_frame_size = None
        self.last_key = -1

    def handle_frame(self, image, header, side_image=None, side_header=None):
        image, self.video_writer, video_path, self.video_frame_size, keep_running, key = _handle_video_frame(
            image,
            header,
            side_image=side_image,
            side_header=side_header,
            show=self.show,
            record=self.record,
            output_dir=self.output_dir,
            filename=self.filename,
            video_writer=self.video_writer,
            video_frame_size=self.video_frame_size,
            window_name=self.window_name,
            margin=self.margin,
            debug=self.debug,
        )
        self.last_key = key
        if video_path is not None:
            self.video_path = video_path
        return keep_running

    def close(self):
        release_video_writer(self.video_writer)
        self.video_writer = None
        self.video_frame_size = None
        if self.show and self.window_name:
            cv2.destroyWindow(self.window_name)


def label_mask(masks: list[str], image: np.ndarray, segmentations: list):
    if not masks: return None

    mask_frame = np.zeros_like(image)
    blur_size = MASK_EDGE_BLUR
    if blur_size % 2 == 0:
        blur_size += 1
    pad = blur_size // 2

    for segmentation in segmentations:
        if segmentation.label.lower() not in masks:
            continue

        mask = segmentation.mask.astype(np.uint8)
        x, y, width, height = cv2.boundingRect(mask)
        if width == 0 or height == 0:
            continue

        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(image.shape[1], x + width + pad)
        y1 = min(image.shape[0], y + height + pad)

        mask_roi = mask[y0:y1, x0:x1]
        image_roi = image[y0:y1, x0:x1]
        output_roi = mask_frame[y0:y1, x0:x1]

        alpha = cv2.GaussianBlur(mask_roi * 255, (blur_size, blur_size), 0)
        alpha = (alpha.astype(np.float32) / 255.0)[..., None]

        blended = (
            output_roi.astype(np.float32) * (1.0 - alpha)
            + image_roi.astype(np.float32) * alpha
        ).astype(np.uint8)

        mask_bool = mask_roi.astype(bool)
        output_roi[:] = blended
        output_roi[mask_bool] = image_roi[mask_bool]
                    
    return mask_frame

def safe_imwrite(path: str, image: np.ndarray, max_dim: int = 8000) -> bool:
    if image is None:
        raise ValueError(f"Failed to save {path}: image is None")

    h, w = image.shape[:2]

    scale = min(
        1.0,
        max_dim / max(h, w)
    )

    if scale < 1.0:
        image = cv2.resize(
            image,
            (
                int(w * scale),
                int(h * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    saved = cv2.imwrite(path, image)
    if not saved:
        raise OSError(f"Failed to save {path}: cv2.imwrite returned False")
    return saved
