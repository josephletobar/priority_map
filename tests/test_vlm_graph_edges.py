import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from priority_map.modules.GraphBuilder import GraphBuilder
from priority_map.modules.SceneUnderstanding import SceneUnderstanding
from priority_map.scripts.cluster_segmentations import ClusteredSegmentation


def cluster(label, source_label, position):
    return ClusteredSegmentation(
        label=label,
        source_label=source_label,
        centroid=position,
        score=50,
        count=1,
        mask=np.ones((2, 2), dtype=np.uint8),
        geo_pos=position,
        color=(0, 255, 0),
    )


class SceneResponseTests(unittest.TestCase):
    def test_normalizes_compact_edges_and_skips_malformed_entries(self):
        scene = SceneUnderstanding.__new__(SceneUnderstanding)
        result = scene._normalize_scene_response({
            "labels": {
                "road": {
                    "reasoning": "Useful access, but it is not the target.",
                    "score": 80,
                    "edges": [
                        {"to_label": "building", "text": "Supports"},
                        {"to_node_id": "vehicle_0", "text": "Likely Contains"},
                        {"to_label": "building", "text": "this label is too long"},
                        {"to_label": "building", "text": ""},
                        "invalid",
                    ],
                },
                "building": {
                    "reasoning": "Useful context, but indirect.",
                    "score": 45,
                },
            }
        })

        self.assertEqual(set(result.labels), {"road", "building"})
        self.assertEqual(result.edge_intents, [
            {
                "source_label": "road",
                "to_label": "building",
                "text": "supports",
            },
            {
                "source_label": "road",
                "to_node_id": "vehicle_0",
                "text": "likely_contains",
            },
        ])

    def test_requires_labels_wrapper_but_ignores_malformed_edge_list(self):
        scene = SceneUnderstanding.__new__(SceneUnderstanding)
        with self.assertRaises(ValueError):
            scene._normalize_scene_response({"road": {"reasoning": "x", "score": 1}})
        result = scene._normalize_scene_response({
            "labels": {
                "road": {"reasoning": "x", "score": 1, "edges": {}}
            }
        })
        self.assertEqual(result.edge_intents, [])


class GraphBuilderEdgeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.builder = GraphBuilder(Path(self.temp_dir.name))

    def tearDown(self):
        self.builder.close()
        self.temp_dir.cleanup()

    def test_resolves_current_and_prior_edges_and_deduplicates(self):
        prior = self.builder.add_nodes([cluster("vehicle", "vehicle", (900, 900))])
        prior_id = prior["label_to_node_ids"]["vehicle"][0]
        current = self.builder.add_nodes([
            cluster("roads", "road", (0, 0)),
            cluster("building", "building", (100, 0)),
        ])
        intents = [
            {"source_label": "road", "to_label": "building", "text": "provides access to"},
            {"source_label": "road", "to_node_id": prior_id, "text": "leads toward"},
            {"source_label": "road", "to_node_id": "missing_0", "text": "invalid"},
            {"source_label": "road", "to_label": "road", "text": "self"},
        ]
        context = {"nodes": [{"id": prior_id}], "spatial_edges": [], "model_edges": []}

        inserted = self.builder.resolve_scene_edge_intents(intents, current, context)
        repeated = self.builder.resolve_scene_edge_intents(intents, current, context)

        self.assertEqual(len(inserted), 2)
        self.assertEqual(repeated, [])
        self.builder.cursor.execute("SELECT text FROM model_edges ORDER BY text")
        self.assertEqual(
            [row[0] for row in self.builder.cursor.fetchall()],
            ["leads toward", "provides access to"],
        )

    def test_context_distinguishes_spatial_and_model_edges(self):
        added = self.builder.add_nodes([
            cluster("road", "road", (0, 0)),
            cluster("building", "building", (100, 0)),
        ])
        self.builder.resolve_scene_edge_intents(
            [{"source_label": "road", "to_label": "building", "text": "serves"}],
            added,
            {"nodes": []},
        )

        context = self.builder.get_recent_graph_context(limit=10)

        self.assertTrue(context["spatial_edges"])
        self.assertEqual(context["model_edges"][0]["text"], "serves")
        self.assertNotIn("edges", context)

    def test_rendered_graph_excludes_spatial_edges(self):
        self.builder.add_nodes([
            cluster("road", "road", (0, 0)),
            cluster("building", "building", (100, 0)),
        ])
        drawn_edge_counts = []

        def capture_draw(graph, *args, **kwargs):
            drawn_edge_counts.append(graph.number_of_edges())

        with patch("priority_map.modules.GraphBuilder.nx.draw", side_effect=capture_draw):
            frame = self.builder.render_2d_graph_frame()

        self.assertIsNotNone(frame)
        self.assertEqual(drawn_edge_counts, [0])


if __name__ == "__main__":
    unittest.main()
