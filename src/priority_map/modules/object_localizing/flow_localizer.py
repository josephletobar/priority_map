from priority_map.modules.object_localizing.localizer import ObjectLocalizer, LocalizationContext
from priority_map.modules.Segment import Segmentation

class FlowLocalizer(ObjectLocalizer):
    def __init__(self):
        self.cumulative_transform_dx = 0
        self.cumulative_transform_dy = 0
        self.last_frame_index = None
        super().__init__()

    def localize(self, segmentation: Segmentation, context: LocalizationContext):
        if segmentation.centroid is None or context.flow_transform is None:
            return None

        centroid_x, centroid_y = segmentation.centroid[0], segmentation.centroid[1]
        transform_dx, transform_dy = context.flow_transform[0], context.flow_transform[1]

        frame_index = getattr(context.frame, "frame_index", None)
        if frame_index != self.last_frame_index:
            self.cumulative_transform_dx += transform_dx
            self.cumulative_transform_dy += transform_dy
            self.last_frame_index = frame_index

        geo_pos = (centroid_x + self.cumulative_transform_dx, centroid_y + self.cumulative_transform_dy)

        return geo_pos
