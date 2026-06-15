import numpy as np
import cv2

MERGE_DIST = 9

def merge(a, b):
    merged_score = (a.score + b.score) / 2
    merged_label = a.label if a.score >= b.score else b.label
    merged_id = a.id or b.id
    pos_a = getattr(a, "geo_pos", None)
    pos_b = getattr(b, "geo_pos", None)

    if pos_a is not None and pos_b is not None:
        merged_geo_pos = tuple(
            (float(a_coord) + float(b_coord)) / 2
            for a_coord, b_coord in zip(pos_a, pos_b)
        )
    else:
        merged_geo_pos = pos_a or pos_b

    return type(a)(
        mask=np.logical_or(a.mask > 0, b.mask > 0),
        label=merged_label,
        score=merged_score,
        id=merged_id,
        geo_pos=merged_geo_pos
    )

def similar(a, b):
    score_match = abs(a.score - b.score) < 20
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MERGE_DIST, MERGE_DIST)
    )
    touching = np.any(
        cv2.dilate(a.mask.astype(np.uint8), kernel, iterations=1)
        &
        b.mask.astype(np.uint8)
    )
    return score_match and touching

def merge_similar(items):
    items = list(items)

    while True:
        merged = False

        for i, item_a in enumerate(items):
            for item_b in items[i + 1:]:
                if similar(item_a, item_b):
                    merged_item = merge(item_a, item_b)
                    items = [
                        item for item in items
                        if item is not item_a and item is not item_b
                    ]
                    items.append(merged_item)
                    merged = True
                    break

            if merged:
                break

        if not merged:
            return items
