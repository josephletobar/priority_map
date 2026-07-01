import numpy as np
import cv2
from priority_map.modules.PanoramaBuilder import PanoramaBuilder
from priority_map.config import params as config

HEATMAP_PROCESS_SCALE = 1
LABEL_REFERENCE_HEIGHT = 720
LABEL_FONT_MIN_SCALE = 0.3
LABEL_FONT_GAIN = 1.35
LABEL_FONT_GAMMA = 2.0
LABEL_COLOR_MAX_DIVERGENCE = 80.0
DILATION_EDGE_DIVISOR = 33
BLUR_REFERENCE_EDGE = 640
BLUR_EDGE_SCALE = 0.0627


class Heatmap:
    def __init__(self, blur_spread=config.BLUR_SPREAD):
        self.heat_gamma = 1.0

        self.BLUR_SPREAD = blur_spread

        self.transform_dx = 0
        self.transform_dy = 0

    def _annotation_resolution_scale(self, image_shape):
        height = image_shape[0]
        return max(1, height) / LABEL_REFERENCE_HEIGHT

    def _label_font_scale(self, score, resolution_scale):
        score_norm = np.clip(float(score) / 100.0, 0.0, 1.0)
        return resolution_scale * (
            LABEL_FONT_MIN_SCALE + LABEL_FONT_GAIN * (score_norm ** LABEL_FONT_GAMMA)
        )

    def _score_color(self, score):
        heat_value = np.uint8([[np.clip(float(score), 0.0, 100.0) * 2.55]])
        return cv2.applyColorMap(heat_value, cv2.COLORMAP_JET)[0, 0]

    def _label_color(self, heatmap, x, y, score):
        score_color = self._score_color(score)
        if heatmap is None:
            return tuple(int(c) for c in score_color)

        pulled_color = heatmap[y, x]
        color_distance = np.linalg.norm(
            pulled_color.astype(np.float32) - score_color.astype(np.float32)
        )
        if color_distance > LABEL_COLOR_MAX_DIVERGENCE:
            return tuple(int(c) for c in score_color)

        return tuple(int(c) for c in pulled_color)

    def _odd_kernel_size(self, value, max_size=None):
        kernel_size = max(3, int(round(value)))
        if kernel_size % 2 == 0:
            kernel_size += 1

        if max_size is not None:
            max_size = max(3, int(max_size))
            if max_size % 2 == 0:
                max_size -= 1
            kernel_size = min(kernel_size, max_size)

        return kernel_size

    def _dilation_iterations(self, image_shape):
        height, width = image_shape[:2]
        final_edge = max(height, width)
        return max(1, int(round(final_edge / DILATION_EDGE_DIVISOR)))

    def _scaled_blur_spread(self, image_shape):
        height, width = image_shape[:2]
        final_edge = max(height, width)
        scaled_spread = self.BLUR_SPREAD + (
            (final_edge - BLUR_REFERENCE_EDGE) * BLUR_EDGE_SCALE
        )
        return max(1.0, scaled_spread)

    def _create_heatmap(self, image, regions):
        if not regions: return image, None

        h, w = image.shape[:2]
        small_size = (
            max(1, int(w * HEATMAP_PROCESS_SCALE)),
            max(1, int(h * HEATMAP_PROCESS_SCALE)),
        )
        small_image = cv2.resize(
            image,
            small_size,
            interpolation=cv2.INTER_AREA,
        )
        heatmap = np.zeros(small_image.shape[:2], dtype=np.float32)
        valid = np.zeros(small_image.shape[:2], dtype=np.float32)

        for region in regions:
            mask = cv2.resize(
                region.mask.astype(np.float32),
                small_size,
                interpolation=cv2.INTER_AREA,
            )
            score = region.score

            # boosted_score = (score / 100) ** self.heat_gamma

            # heatmap = np.maximum(
            #     heatmap,
            #     mask * boosted_score
            # )

            # valid = np.maximum(
            #     valid,
            #     mask
            # )

            heatmap = np.maximum(
                heatmap,
                mask * score
            )

        kernel_scale = HEATMAP_PROCESS_SCALE
        blur_spread = self._scaled_blur_spread(image.shape)
        spread_size = self._odd_kernel_size(blur_spread * kernel_scale / 15)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (spread_size, spread_size)
        )

        heatmap = cv2.dilate(
            heatmap,
            kernel,
            iterations=self._dilation_iterations(image.shape),
        )

        gaussian_size = self._odd_kernel_size(blur_spread * kernel_scale, max_size=min(small_size))
        heatmap = cv2.GaussianBlur(heatmap, (gaussian_size, gaussian_size), 0)

        heatmap = np.clip(heatmap, 0, 100)
        heatmap = (heatmap * 2.55).astype(np.uint8)
        

        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap_colored, (w, h), interpolation=cv2.INTER_LINEAR)

        output = cv2.addWeighted(
            small_image,    # base image
            0.6,      # weight of base image
            heatmap_colored,  # heatmap overlay image
            0.4,      # weight of heatmap
            0         # constant brightness offset added to every pixel
        )

        return cv2.resize(output, (w, h), interpolation=cv2.INTER_LINEAR), heatmap_resized

    def _draw_segmentation_labels(self, image, heatmap, segmentations):
        output = image.copy()
        height, width = output.shape[:2]
        resolution_scale = self._annotation_resolution_scale(output.shape)

        for segmentation in segmentations:
            if segmentation.centroid is None:
                continue

            x, y = segmentation.centroid
            x = max(0, min(width - 1, int(x)))
            y = max(0, min(height - 1, int(y)))

            color = self._label_color(heatmap, x, y, segmentation.score)

            segmentation.color = color  # Store the color in the segmentation object

            score_norm = np.clip(float(segmentation.score) / 100.0, 0.0, 1.0)
            dot_radius = max(1, int(round(4 * score_norm * resolution_scale)))
            cv2.circle(
                output,  # image
                (x, y),  # center
                dot_radius,  # scaled according to relevance and frame size
                color,  # BGR color from heatmap
                -1,  # filled
                cv2.LINE_AA,  # antialiased
            )

            font_scale = self._label_font_scale(segmentation.score, resolution_scale)
            thickness = max(1, int(round((2 if segmentation.score > 50 else 1) * resolution_scale)))
            text_size, baseline = cv2.getTextSize(segmentation.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            pad_x = max(2, int(round(4 * resolution_scale)))
            pad_y = max(2, int(round(4 * resolution_scale)))
            text_x = x + max(3, int(round(6 * resolution_scale)))
            text_y = y - max(3, int(round(6 * resolution_scale)))
            rect_top_left = (text_x - pad_x, text_y - text_size[1] - pad_y)
            rect_bottom_right = (text_x + text_size[0] + pad_x, text_y + baseline + pad_y)

            overlay = output.copy()
            cv2.rectangle(overlay, rect_top_left, rect_bottom_right, (60, 60, 60), -1)  # gray filled
            output = cv2.addWeighted(overlay, 0.6, output, 0.4, 0)  # blend

            cv2.putText(
                output,  # image
                segmentation.label,  # text
                (text_x, text_y),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                font_scale,  # scaled according to relevance
                color,  # BGR color from heatmap
                thickness,  # thickness
                cv2.LINE_AA,  # antialiased
            )

        return output

    def draw_heatmap(self, image, segmentations):

        heatmap_overlay, heatmap_only = self._create_heatmap(image, segmentations)
        heatmap_text = self._draw_segmentation_labels(heatmap_overlay, heatmap_only, segmentations)

        # transform = self.panoramic_transform.transform_dx, self.panoramic_transform.transform_dy
        # self.panorama_builder.create_panorama(transform, result)
        # # cv2.imshow("Panorama Heat", self.panorama_builder.panorama)
        # # cv2.waitKey(1)

        self.prev_heat = heatmap_text

        return heatmap_text, heatmap_only
