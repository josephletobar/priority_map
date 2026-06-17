import numpy as np
import cv2

HEATMAP_PROCESS_SCALE = 1

class Heatmap:
    def __init__(self):
        self.heat_gamma = 10.0

    def _create_heatmap(self, image, regions):
        if not regions: return image

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

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        output = cv2.addWeighted(
            small_image,    # base image
            0.6,      # weight of base image
            heatmap,  # heatmap overlay image
            0.4,      # weight of heatmap
            0         # constant brightness offset added to every pixel
        )

        return cv2.resize(output, (w, h), interpolation=cv2.INTER_LINEAR)

    # def _draw_node_labels(self, image):
    #     output = image.copy()

    #     for node in self.nodes:
    #         mask = node.mask > 0
    #         if not np.any(mask):
    #             continue

    #         ys, xs = np.nonzero(mask)
    #         x = int(xs.mean())
    #         y = int(ys.mean())

    #         cv2.putText(
    #             output,
    #             node.label,
    #             (x, y),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.7,
    #             (100, 100, 100),
    #             2,
    #             cv2.LINE_AA,
    #         )

    #     return output

    def draw_heatmap(self, image, segmentations):

        heatmap = self._create_heatmap(image, segmentations)
        # heatmap = self._draw_node_labels(heatmap)

        return heatmap
