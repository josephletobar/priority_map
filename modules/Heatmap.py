import numpy as np
import cv2

class Heatmap:
    def __init__(self):
        self.heat_gamma = 10.0

    def _create_heatmap(self, image, regions):
        heatmap = np.zeros(image.shape[:2], dtype=np.float32)
        valid = np.zeros(image.shape[:2], dtype=np.float32)

        if not regions: return image

        for region in regions:
            mask = region.mask.astype(np.float32)
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

        spread = (301, 301)
        sigma = 0
        heatmap = cv2.GaussianBlur(heatmap, spread, sigma)
        valid = cv2.GaussianBlur(valid, spread, sigma)
        heatmap = heatmap / (valid + 1e-6)
        heatmap = np.power(heatmap, 1 / self.heat_gamma) * 100

        heatmap = np.clip(heatmap, 0, 100)
        heatmap = (heatmap * 2.55).astype(np.uint8)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        output = cv2.addWeighted(
            image,    # base image
            0.6,      # weight of base image
            heatmap,  # heatmap overlay image
            0.4,      # weight of heatmap
            0         # constant brightness offset added to every pixel
        )

        return output

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
