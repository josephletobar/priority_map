import numpy as np


class GeoLocalizer:
    def __init__(self):
        pass

    def get_location(self, image, mask, curr_pos):
        easting, northing, altitude = curr_pos

        if image is None or mask is None:
            return (easting, northing, altitude)

        h, w = image.shape[:2]
        binary_mask = mask > 0

        if not np.any(binary_mask):
            return (easting, northing, altitude)

        ys, xs = np.nonzero(binary_mask)
        cx = float(xs.mean())
        cy = float(ys.mean())

        meters_per_pixel_x = float(altitude) / max(w, 1)
        meters_per_pixel_y = float(altitude) / max(h, 1)

        dx = (cx - (w / 2.0)) * meters_per_pixel_x
        dy = (cy - (h / 2.0)) * meters_per_pixel_y

        return (
            float(easting) + dx,
            float(northing) - dy,
            float(altitude),
        )
