import numpy as np

class PanoramaBuilder:
    def __init__(self, alpha=0.9):
        self.alpha = alpha

        self.panorama = None
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.prev_heat = None

    def create_panorama(self, transform, image):

        transform_dx = transform[0] 
        transform_dy = transform[1] 

        h, w = image.shape[:2]

        if self.panorama is None:
            self.panorama = image.copy()
            self.pan_x = 0
            self.pan_y = 0
            return
        else:
            self.pan_x -= transform_dx
            self.pan_y -= transform_dy

            x = int(self.pan_x)
            y = int(self.pan_y)

            min_x = min(0, x)
            min_y = min(0, y)

            max_x = max(self.panorama.shape[1], x + w)
            max_y = max(self.panorama.shape[0], y + h)

            # allocate necessary space
            new_panorama = np.zeros(
                (max_y - min_y, max_x - min_x, 3),
                dtype=np.uint8
            )

            # copy existing panorama
            new_panorama[
                -min_y:-min_y + self.panorama.shape[0],
                -min_x:-min_x + self.panorama.shape[1]
            ] = self.panorama

            # region where new image should go
            roi = new_panorama[
                y - min_y:y - min_y + h,
                x - min_x:x - min_x + w
            ]

            mask = np.all(roi == 0, axis=2)

            # fill empty pixels
            roi[mask] = image[mask]

            # average overlapping pixels (commented out for testing)
            # roi[~mask] = (
            #     self.alpha * roi[~mask].astype(np.float32)
            #     + (1 - self.alpha) * image[~mask].astype(np.float32)
            # ).astype(np.uint8)

            self.panorama = new_panorama

        return self.panorama