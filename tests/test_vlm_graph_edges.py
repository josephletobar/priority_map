import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np

from priority_map.modules.GraphBuilder import GraphBuilder
from priority_map.modules.SceneUnderstanding import SceneUnderstanding
from priority_map.runner import PriorityMapRunner
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

    def test_nearby_node_positions_respect_radius_and_limit(self):
        for index, x in enumerate((1.0, 2.0, 3.0, 4.0, 6.0)):
            self.builder.cursor.execute(
                """
                INSERT INTO nodes
                (id, label, score, count, geo_pos_x, geo_pos_y,
                 color_b, color_g, color_r, mask_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"coverage_{index}",
                    f"coverage_{index}",
                    1.0,
                    1,
                    x,
                    0.0,
                    0,
                    0,
                    0,
                    None,
                ),
            )
        self.builder.conn.commit()

        limited = self.builder.get_nearby_node_positions(
            (0.0, 0.0),
            radius=5,
            limit=3,
        )
        all_nearby = self.builder.get_nearby_node_positions(
            (0.0, 0.0),
            radius=5,
            limit=10,
        )

        self.assertEqual(limited, [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])
        self.assertEqual(
            all_nearby,
            [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
        )

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

    def test_spatial_view_draws_only_the_spatial_mst(self):
        self.builder.add_nodes([
            cluster("road", "road", (0, 0)),
            cluster("building", "building", (100, 0)),
            cluster("vehicle", "vehicle", (50, 80)),
        ])
        drawn_graphs = []
        rendered_labels = []

        def capture_draw(graph, *args, **kwargs):
            drawn_graphs.append(graph.copy())

        def capture_labels(graph, pos, edge_labels, **kwargs):
            rendered_labels.append(edge_labels)

        with (
            patch("priority_map.modules.GraphBuilder.nx.draw", side_effect=capture_draw),
            patch(
                "priority_map.modules.GraphBuilder.nx.draw_networkx_edge_labels",
                side_effect=capture_labels,
            ),
        ):
            frame = self.builder.render_2d_graph_frame(view="spatial")

        self.assertIsNotNone(frame)
        self.assertEqual(drawn_graphs[0].number_of_edges(), 2)
        self.assertEqual(len(rendered_labels[0]), 2)
        self.assertTrue(all(label.isdigit() for label in rendered_labels[0].values()))

    def test_longer_model_labels_have_weaker_attraction(self):
        graph = nx.Graph()
        graph.add_edges_from([("a", "b"), ("b", "c")])
        labels = {("a", "b"): "short", ("b", "c"): "much_longer_label"}

        self.builder._apply_model_layout_weights(graph, labels)

        self.assertGreater(
            graph["a"]["b"]["layout_weight"],
            graph["b"]["c"]["layout_weight"],
        )

    def test_model_layout_rerenders_from_scratch_with_fixed_seed(self):
        first_graph = nx.Graph([("a", "b")])
        second_graph = nx.Graph([("a", "b"), ("b", "c")])
        first_positions = {
            "a": np.array([0.0, 0.0]),
            "b": np.array([1.0, 0.0]),
        }
        second_positions = {
            **first_positions,
            "c": np.array([2.0, 0.0]),
        }

        with patch(
            "priority_map.modules.GraphBuilder.nx.spring_layout",
            side_effect=[first_positions, second_positions],
        ) as spring_layout:
            self.builder._model_layout(first_graph, {("a", "b"): "links"})
            self.builder._model_layout(
                second_graph,
                {("a", "b"): "links", ("b", "c"): "relates_to"},
            )

        self.assertNotIn("pos", spring_layout.call_args_list[0].kwargs)
        self.assertNotIn("pos", spring_layout.call_args_list[1].kwargs)
        self.assertEqual(
            spring_layout.call_args_list[1].kwargs["seed"],
            self.builder.MODEL_LAYOUT_SEED,
        )
        self.assertEqual(spring_layout.call_args_list[1].kwargs["method"], "energy")
        self.assertEqual(
            spring_layout.call_args_list[1].kwargs["gravity"],
            self.builder.MODEL_LAYOUT_GRAVITY,
        )

    def test_edge_label_length_directly_controls_minimum_distance(self):
        positions = {
            "short_a": np.array([0.0, 0.0]),
            "short_b": np.array([0.01, 0.0]),
            "long_a": np.array([0.0, 1.0]),
            "long_b": np.array([0.01, 1.0]),
        }
        labels = {
            ("short_a", "short_b"): "brief",
            ("long_a", "long_b"): "substantially_longer",
        }

        separated = self.builder._separate_model_edge_labels(positions, labels)
        short_distance = np.linalg.norm(separated["short_b"] - separated["short_a"])
        long_distance = np.linalg.norm(separated["long_b"] - separated["long_a"])

        self.assertGreater(long_distance, short_distance)


class RunnerGraphViewTests(unittest.TestCase):
    def test_graph_view_keys_switch_and_rerender(self):
        runner = PriorityMapRunner.__new__(PriorityMapRunner)
        runner.graph_view = "model"
        runner.graph_builder = MagicMock()
        runner.graph_builder.render_2d_graph_frame.return_value = np.ones((2, 2, 3))
        runner.video_output = SimpleNamespace(last_key=ord("1"))
        runner.last_graph_frame = None
        runner.debug = False

        runner._handle_graph_view_key()

        self.assertEqual(runner.graph_view, "spatial")
        runner.graph_builder.render_2d_graph_frame.assert_called_once_with(view="spatial")
        self.assertIsNotNone(runner.last_graph_frame)

        runner.video_output.last_key = ord("2")
        runner._handle_graph_view_key()
        self.assertEqual(runner.graph_view, "model")


if __name__ == "__main__":
    unittest.main()
