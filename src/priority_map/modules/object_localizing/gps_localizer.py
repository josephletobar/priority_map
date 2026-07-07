import numpy as np

from priority_map.modules.Segment import Segmentation
from priority_map.modules.object_localizing.localizer import ObjectLocalizer, LocalizationContext

class GpsLocalizer(ObjectLocalizer):
    def __init__(self):
        super().__init__()

    def localize(self, segmentation: Segmentation, context: LocalizationContext):
        easting, northing, altitude = context.curr_pos
        image = context.image

        mask = segmentation.mask
        if easting is None or northing is None or altitude is None:
            return None

        if image is None or mask is None:
            return (float(easting), float(northing))

        h, w = image.shape[:2]
        binary_mask = mask > 0

        if not np.any(binary_mask):
            return (float(easting), float(northing))

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
        )
