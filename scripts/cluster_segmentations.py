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

def cluster_segmentations(segmentations, distance_threshold=600):
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
Observed labels: {labels}

Given only these observed labels, choose the most likely scene type.

Allowed scene types:
residential area, rural area, dense forest, park, road network

Return only one label.
"""

_SEMANTIC_LABEL_CACHE = {}


def _normalize_semantic_label_input(label):
    label = str(label).strip().lower()
    label = re.sub(r'\bx\d+\b', ' ', label)
    label = re.sub(r'(?<=[a-z])(?:x\d+)+\b', ' ', label)
    label = re.sub(r'[^a-z0-9\s-]', ' ', label)
    words = [word for word in re.split(r'\s+', label) if word]
    if not words:
        return None

    normalized_words = []
    for word in words:
        if len(word) > 4 and word.endswith('ess') and not word.endswith('ness'):
            word = word[:-1]

        if len(word) > 4 and word.endswith('ies'):
            word = f"{word[:-3]}y"
        elif len(word) > 4 and (
            word.endswith('ches')
            or word.endswith('shes')
            or word.endswith('xes')
            or word.endswith('zes')
            or word.endswith('ses')
        ):
            word = word[:-2]
        elif len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
            word = word[:-1]

        normalized_words.append(word)

    return " ".join(normalized_words)


def _semantic_label_inputs(labels):
    normalized = []
    seen = set()

    for label in labels:
        normalized_label = _normalize_semantic_label_input(label)
        if not normalized_label or normalized_label in seen:
            continue
        seen.add(normalized_label)
        normalized.append(normalized_label)

    return tuple(sorted(normalized))


def _sanitize_semantic_label(label):
    label = str(label).strip().lower()
    label = re.sub(r'[^a-z0-9\s-]', '', label)
    words = [word for word in re.split(r'\s+', label) if word]
    if not words or len(words) > 2:
        return None
    return " ".join(words)


def _semantic_label(cluster_segs):
    unique_labels = _semantic_label_inputs(seg.label for seg in cluster_segs)
    if not unique_labels:
        return None
    if len(unique_labels) == 1:
        return unique_labels[0]

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

    _SEMANTIC_LABEL_CACHE[unique_labels] = label
    return label


def _score_weighted_distances(clustered_segmentations, score_weight):
    centroids = np.array([seg.centroid for seg in clustered_segmentations], dtype=float)
    scores = np.array([seg.score for seg in clustered_segmentations], dtype=float)
    score_weights = (np.clip(scores, 0, 100) / 100.0) ** 2

    spatial_distances = np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :],
        axis=2,
    )
    score_deltas = np.abs(score_weights[:, None] - score_weights[None, :])

    return spatial_distances * (1.0 + score_weight * score_deltas)


def semantic_cluster_from_members(cluster_segs):
    avg_centroid = tuple(np.mean([seg.centroid for seg in cluster_segs], axis=0).astype(int))

    valid_geo_positions = [seg.geo_pos for seg in cluster_segs if seg.geo_pos is not None]
    avg_geo_pos = (
        tuple(np.mean(valid_geo_positions, axis=0))
        if valid_geo_positions
        else avg_centroid
    )

    count = sum(seg.count for seg in cluster_segs)
    merged_mask = np.logical_or.reduce([seg.mask for seg in cluster_segs]).astype(np.uint8)
    representative = max(cluster_segs, key=lambda seg: seg.score)
    label = representative.label
    if len(cluster_segs) > 1:
        label = _semantic_label(cluster_segs) or label

    return ClusteredSegmentation(
        label=label,
        centroid=avg_centroid,
        score=0,
        count=count,
        mask=merged_mask,
        geo_pos=avg_geo_pos,
        color=None,
    )


def semantic_clustering_with_members(clustered_segmentations, distance_threshold=450, score_weight=3.5):
    """Spatially merge existing clusters and return semantic clusters with members.

    Args:
        clustered_segmentations: list of ClusteredSegmentation objects to merge
        distance_threshold: max distance in pixels to consider segmentations as nearby (DBSCAN eps)
        score_weight: how strongly score differences increase effective distance

    Returns:
        list of (ClusteredSegmentation, list[ClusteredSegmentation]) tuples
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
        if cluster_id == -1:
            cluster_id = f"noise_{id(seg)}"
        clusters_dict.setdefault(cluster_id, []).append(seg)

    semantic_clustered = []
    for cluster_segs in clusters_dict.values():
        prompt_labels = _semantic_label_inputs(seg.label for seg in cluster_segs)
        semantic_cluster = semantic_cluster_from_members(cluster_segs)

        print("Semantic clustering:")
        print(f"labels used for {semantic_cluster.label}:", list(prompt_labels))
        print("----------------------")

        semantic_clustered.append((
            semantic_cluster,
            cluster_segs,
        ))

    return semantic_clustered


def semantic_clustering(clustered_segmentations, distance_threshold=2000, score_weight=2.0):
    """Spatially merge existing clusters and label each merged cluster semantically."""
    return [
        semantic_cluster
        for semantic_cluster, _ in semantic_clustering_with_members(
            clustered_segmentations,
            distance_threshold=distance_threshold,
            score_weight=score_weight,
        )
    ]
