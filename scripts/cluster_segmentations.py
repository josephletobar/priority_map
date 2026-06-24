from dataclasses import dataclass
import os
import re

import numpy as np
import requests
from sklearn.cluster import DBSCAN


@dataclass
class ClusteredSegmentation:
    label: str  # segmentation label
    centroid: tuple[int, int]  # averaged centroid of clustered segmentations
    score: float  # relevance score (same for all in cluster)
    count: int  # number of segmentations merged
    mask: np.ndarray  # merged mask of all clustered segmentations
    geo_pos: tuple[float, float]  # global position (centroid + accumulated transform)
    color: tuple[int, int, int] | None = None  # color for visualization (optional)

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

LABEL_PROMPT = """
Given nearby labels: {labels}

Name the physical place they form together.
Use 1-2 common words.
Do not use abstract words like pattern, analysis, arrangement.
Do not explain.
"""

_SEMANTIC_LABEL_CACHE = {}


def _fallback_label(cluster_segs):
    return max(cluster_segs, key=lambda seg: seg.score).label


def _sanitize_semantic_label(label):
    label = str(label).strip().lower()
    label = re.sub(r'[^a-z0-9\s-]', '', label)
    words = [word for word in re.split(r'\s+', label) if word]
    if not words or len(words) > 2:
        return None
    return " ".join(words)


def _semantic_label(labels, fallback):
    unique_labels = tuple(sorted({str(label).strip().lower() for label in labels if str(label).strip()}))
    if not unique_labels:
        return fallback

    if unique_labels in _SEMANTIC_LABEL_CACHE:
        return _SEMANTIC_LABEL_CACHE[unique_labels]

    model = os.getenv("SEMANTIC_CLUSTER_MODEL", "gemma3:1b")
    ollama_url = os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
    timeout = int(os.getenv("SEMANTIC_CLUSTER_TIMEOUT", "20"))
    keep_alive = os.getenv("SEMANTIC_CLUSTER_KEEP_ALIVE", "5m")

    try:
        response = requests.post(
            ollama_url,
            json={
                "model": model,
                "prompt": LABEL_PROMPT.format(labels=", ".join(unique_labels)),
                "stream": False,
                "keep_alive": keep_alive,
                "options": {
                    "temperature": 0,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        label = _sanitize_semantic_label(result.get("response", ""))
    except Exception:
        label = None

    label = label or fallback
    _SEMANTIC_LABEL_CACHE[unique_labels] = label
    return label


def _score_weighted_distances(clustered_segmentations, score_weight):
    centroids = np.array([seg.centroid for seg in clustered_segmentations], dtype=float)
    scores = np.array([seg.score for seg in clustered_segmentations], dtype=float)

    spatial_distances = np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :],
        axis=2,
    )
    score_deltas = np.abs(scores[:, None] - scores[None, :]) / 100.0

    return spatial_distances * (1.0 + score_weight * score_deltas)


def semantic_clustering(clustered_segmentations, distance_threshold=600, score_weight=2.0):
    """Spatially merge existing clusters and label each merged cluster semantically.

    Args:
        clustered_segmentations: list of ClusteredSegmentation objects to merge
        distance_threshold: max distance in pixels to consider segmentations as nearby (DBSCAN eps)
        score_weight: how strongly score differences increase effective distance

    Returns:
        list of ClusteredSegmentation objects
    """
    if not clustered_segmentations:
        return []

    valid_segmentations = [
        seg for seg in clustered_segmentations
        if seg.centroid is not None
    ]
    if not valid_segmentations:
        return []

    distances = _score_weighted_distances(valid_segmentations, score_weight)
    clustering = DBSCAN(
        eps=distance_threshold,
        min_samples=2,
        metric="precomputed",
    ).fit(distances)

    clusters_dict = {}
    for seg, cluster_id in zip(valid_segmentations, clustering.labels_):
        clusters_dict.setdefault(cluster_id, []).append(seg)

    semantic_clustered = []
    for cluster_segs in clusters_dict.values():
        avg_centroid = tuple(np.mean([seg.centroid for seg in cluster_segs], axis=0).astype(int))

        valid_geo_positions = [seg.geo_pos for seg in cluster_segs if seg.geo_pos is not None]
        avg_geo_pos = (
            tuple(np.mean(valid_geo_positions, axis=0))
            if valid_geo_positions
            else avg_centroid
        )

        score = max(seg.score for seg in cluster_segs)
        count = sum(seg.count for seg in cluster_segs)
        merged_mask = np.logical_or.reduce([seg.mask for seg in cluster_segs]).astype(np.uint8)
        representative = max(cluster_segs, key=lambda seg: seg.score)
        label = _semantic_label(
            [seg.label for seg in cluster_segs],
            fallback=_fallback_label(cluster_segs),
        )

        semantic_clustered.append(ClusteredSegmentation(
            label=label,
            centroid=avg_centroid,
            score=score,
            count=count,
            mask=merged_mask,
            geo_pos=avg_geo_pos,
            color=representative.color,
        ))

    return semantic_clustered
