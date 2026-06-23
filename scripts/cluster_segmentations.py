from dataclasses import dataclass
import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class ClusteredSegmentation:
    label: str  # segmentation label
    centroid: tuple[int, int]  # averaged centroid of clustered segmentations
    score: float  # relevance score (same for all in cluster)
    count: int  # number of segmentations merged
    mask: np.ndarray  # merged mask of all clustered segmentations
    geo_pos: tuple[float, float]  # global position (centroid + accumulated transform)


def cluster_segmentations(segmentations, distance_threshold=50):
    """Cluster segmentations by label and spatial proximity.

    Args:
        segmentations: list of Segmentation objects to cluster
        distance_threshold: max distance in pixels to consider segmentations as nearby (DBSCAN eps)

    Returns:
        list of ClusteredSegmentation objects
    """
    if not segmentations:
        return []

    # Group by label
    by_label = {}
    for seg in segmentations:
        if seg.label not in by_label:
            by_label[seg.label] = []
        by_label[seg.label].append(seg)

    clustered = []

    for label, segs in by_label.items():
        if not segs:
            continue

        # Get centroids for clustering
        centroids = np.array([seg.centroid for seg in segs])

        # DBSCAN clustering by distance
        clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(centroids)
        labels = clustering.labels_

        # Group by cluster
        clusters_dict = {}
        for seg, cluster_id in zip(segs, labels):
            if cluster_id not in clusters_dict:
                clusters_dict[cluster_id] = []
            clusters_dict[cluster_id].append(seg)

        # Create ClusteredSegmentation for each cluster
        for cluster_segs in clusters_dict.values():
            avg_centroid = tuple(np.mean([seg.centroid for seg in cluster_segs], axis=0).astype(int))
            avg_geo_pos = tuple(np.mean([seg.geo_pos for seg in cluster_segs if seg.geo_pos is not None], axis=0))
            score = cluster_segs[0].score
            count = len(cluster_segs)
            merged_mask = np.logical_or.reduce([seg.mask for seg in cluster_segs]).astype(np.uint8)

            clustered.append(ClusteredSegmentation(
                label if count == 1 else f"{label}s",
                centroid=avg_centroid,
                score=score,
                count=count,
                mask=merged_mask,
                geo_pos=avg_geo_pos,
            ))

            

    return clustered
