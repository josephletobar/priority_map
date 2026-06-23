import numpy as np
import cv2
from modules.PanoramaBuilder import PanoramaBuilder

HEATMAP_PROCESS_SCALE = 1


class Heatmap:
    def __init__(self):
        self.heat_gamma = 10.0

        self.transform_dx = 0
        self.transform_dy = 0

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

            boosted_score = (score / 100) ** self.heat_gamma

            heatmap = np.maximum(
                heatmap,
                mask * boosted_score
            )

            valid = np.maximum(
                valid,
                mask
            )

        spread_size = max(3, int(401 * HEATMAP_PROCESS_SCALE))
        if spread_size % 2 == 0:
            spread_size += 1
        spread = (spread_size, spread_size)
        sigma = 0
        heatmap = cv2.GaussianBlur(heatmap, spread, sigma)
        valid = cv2.GaussianBlur(valid, spread, sigma)
        heatmap = heatmap / (valid + 1e-6)
        heatmap = np.power(heatmap, 1 / self.heat_gamma) * 100

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

        for segmentation in segmentations:
            if segmentation.centroid is None:
                continue

            x, y = segmentation.centroid
            x = max(0, min(width - 1, int(x)))
            y = max(0, min(height - 1, int(y)))

            color = tuple(int(c) for c in heatmap[y, x])

            cv2.circle(
                output,  # image
                (x, y),  # center
                int(4 * float(segmentation.score/100)),  # scaled according to relevance
                color,  # BGR color from heatmap
                -1,  # filled
                cv2.LINE_AA,  # antialiased
            )

            text_size = cv2.getTextSize(segmentation.label, cv2.FONT_HERSHEY_SIMPLEX, 1.25 * float(segmentation.score/100), 2 if segmentation.score > 50 else 1)[0]
            rect_top_left = (x + 4, y - text_size[1] - 10)
            rect_bottom_right = (x + 8 + text_size[0], y + 2)

            overlay = output.copy()
            cv2.rectangle(overlay, rect_top_left, rect_bottom_right, (60, 60, 60), -1)  # gray filled
            output = cv2.addWeighted(overlay, 0.6, output, 0.4, 0)  # blend

            cv2.putText(
                output,  # image
                segmentation.label,  # text
                (x + 6, y - 6),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1.25 * float(segmentation.score/100),  # scaled according to relevance
                color,  # BGR color from heatmap
                2 if segmentation.score > 50 else 1,  # thickness
                cv2.LINE_AA,  # antialiased
            )

        return output

    def draw_heatmap(self, image, segmentations):

        heatmap_overlaid, heatmap_colored = self._create_heatmap(image, segmentations)
        result = self._draw_segmentation_labels(heatmap_overlaid, heatmap_colored, segmentations)

        # transform = self.panoramic_transform.transform_dx, self.panoramic_transform.transform_dy
        # self.panorama_builder.create_panorama(transform, result)
        # # cv2.imshow("Panorama Heat", self.panorama_builder.panorama)
        # # cv2.waitKey(1)

        self.prev_heat = result

        return result
