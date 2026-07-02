from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

from priority_map.config import params as config


@dataclass
class ClusteredSegmentation:
    label: str
    centroid: tuple[int, int]
    score: float
    count: int
    mask: np.ndarray
    geo_pos: tuple[float, float]
    reasoning: str = ""
    color: tuple[int, int, int] | None = None


def cluster_segmentations(segmentations, distance_threshold=config.SEGMENTATION_CLUSTER_DISTANCE_THRESHOLD):
    """Cluster segmentations by label and spatial proximity."""
    if not segmentations:
        return []

    by_label = {}
    for seg in segmentations:
        by_label.setdefault(seg.label, []).append(seg)

    clustered = []

    for label, segs in by_label.items():
        if not segs:
            continue

        centroids = np.array([seg.centroid for seg in segs])
        clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(centroids)

        clusters_dict = {}
        for seg, cluster_id in zip(segs, clustering.labels_):
            clusters_dict.setdefault(cluster_id, []).append(seg)

        for cluster_segs in clusters_dict.values():
            avg_centroid = tuple(np.mean([seg.centroid for seg in cluster_segs], axis=0).astype(int))
            valid_geo_positions = [
                seg.geo_pos
                for seg in cluster_segs
                if seg.geo_pos is not None
            ]
            avg_geo_pos = (
                tuple(np.mean(valid_geo_positions, axis=0))
                if valid_geo_positions
                else avg_centroid
            )
            score = cluster_segs[0].score
            count = len(cluster_segs)
            merged_mask = np.logical_or.reduce([seg.mask for seg in cluster_segs]).astype(np.uint8)
            reasonings = []
            seen_reasonings = set()
            for seg in cluster_segs:
                reasoning = str(getattr(seg, "reasoning", "") or "").strip()
                if reasoning and reasoning not in seen_reasonings:
                    seen_reasonings.add(reasoning)
                    reasonings.append(reasoning)

            clustered.append(
                ClusteredSegmentation(
                    label,
                    centroid=avg_centroid,
                    score=score,
                    count=count,
                    mask=merged_mask,
                    geo_pos=avg_geo_pos,
                    reasoning="\n".join(reasonings),
                )
            )

    return clustered
