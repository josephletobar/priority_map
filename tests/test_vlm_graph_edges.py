import tempfile
import unittest
import sqlite3
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import networkx as nx
import cv2
import numpy as np

from priority_map.modules.GraphBuilder import GraphBuilder
from priority_map.modules.SceneUnderstanding import SceneUnderstanding
from priority_map.runner import PriorityMapRunner
from priority_map.scripts.cluster_segmentations import ClusteredSegmentation


def cluster(label, source_label, position, score=50):
    return ClusteredSegmentation(
        label=label,
        source_label=source_label,
        centroid=position,
        score=score,
        count=1,
        mask=np.ones((2, 2), dtype=np.uint8),
        geo_pos=position,
        color=(0, 255, 0),
    )


class SceneResponseTests(unittest.TestCase):
    def test_normalizes_compact_and_longer_edges_and_skips_malformed_entries(self):
        scene = SceneUnderstanding.__new__(SceneUnderstanding)
        result = scene._normalize_scene_response({
            "labels": {
                "road": {
                    "reasoning": "Useful access, but it is not the target.",
                    "score": 80,
                    "edges": [
                        {"to_label": "building", "text": "Supports"},
                        {"to_node_id": "vehicle_0", "text": "Likely Contains"},
                        {"to_label": "building", "text": "this longer relationship label is allowed"},
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
            {
                "source_label": "road",
                "to_label": "building",
                "text": "this_longer_relationship_label_is_allowed",
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

    def test_observed_state_tracks_visible_georeferenced_nodes(self):
        rows = [
            ("visible_0", 0.0, 0.0),
            ("outside_0", 100.0, 0.0),
        ]
        self.builder.cursor.executemany(
            '''
            INSERT INTO nodes
            (id, label, score, count, geo_pos_x, geo_pos_y,
             coordinate_mode, coverage_radius_m)
            VALUES (?, ?, 50.0, 1, ?, ?, 'wgs84', 10.0)
            ''',
            [(node_id, node_id, x, y) for node_id, x, y in rows],
        )
        self.builder.conn.commit()

        self.builder.update_observed_nodes(["visible_0"])
        nodes = {
            node["id"]: node
            for node in self.builder.get_georeferenced_nodes()
        }

        self.assertFalse(nodes["visible_0"]["observed"])
        self.assertTrue(nodes["outside_0"]["observed"])

        self.builder.update_observed_nodes(["outside_0"])
        nodes = {
            node["id"]: node
            for node in self.builder.get_georeferenced_nodes()
        }
        self.assertTrue(nodes["visible_0"]["observed"])
        self.assertFalse(nodes["outside_0"]["observed"])

    def test_existing_database_is_migrated_with_observed_default_false(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "graph.db"
            connection = sqlite3.connect(db_path)
            connection.execute(
                '''
                CREATE TABLE nodes (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    score REAL NOT NULL,
                    count INTEGER NOT NULL,
                    geo_pos_x REAL NOT NULL,
                    geo_pos_y REAL NOT NULL,
                    color_b INTEGER,
                    color_g INTEGER,
                    color_r INTEGER
                )
                '''
            )
            connection.commit()
            connection.close()

            migrated = GraphBuilder(Path(temp_dir))
            try:
                columns = {
                    row[1]
                    for row in migrated.cursor.execute(
                        "PRAGMA table_info(nodes)"
                    ).fetchall()
                }
                self.assertIn("observed", columns)
                self.assertIn("frame_blob", columns)
                self.assertIn("visual_encoding", columns)
            finally:
                migrated.close()

    def test_priority_score_controls_visual_storage_encoding(self):
        mask = np.zeros((120, 160), dtype=np.uint8)
        mask[20:100, 40:120] = 1
        frame = np.full((120, 160, 3), 127, dtype=np.uint8)
        clusters = []
        for index, score in enumerate((0, 25, 50, 75, 100)):
            item = cluster(
                f"priority_{score}",
                f"priority_{score}",
                (index * 1000, 0),
                score=score,
            )
            item.mask = mask.copy()
            clusters.append(item)

        self.builder.add_nodes(clusters, frame_image=frame)
        rows = {
            score: (mask_blob, frame_blob, encoding)
            for score, mask_blob, frame_blob, encoding in self.builder.cursor.execute(
                '''
                SELECT score, mask_blob, frame_blob, visual_encoding
                FROM nodes
                WHERE label LIKE 'priority_%'
                '''
            ).fetchall()
        }

        self.assertEqual(rows[0.0], (None, None, "metadata"))
        self.assertEqual(rows[25.0], (None, None, "metadata"))

        low_mask_blob, low_frame_blob, low_encoding = rows[50.0]
        self.assertEqual(low_encoding, "mask_low")
        self.assertIsNone(low_frame_blob)
        with np.load(BytesIO(low_mask_blob), allow_pickle=False) as data:
            self.assertEqual(data["mask"].shape, (48, 64))

        full_mask_blob, full_frame_blob, full_encoding = rows[75.0]
        self.assertEqual(full_encoding, "mask_full")
        self.assertIsNone(full_frame_blob)
        with np.load(BytesIO(full_mask_blob), allow_pickle=False) as data:
            self.assertEqual(data["mask"].shape, mask.shape)

        frame_mask_blob, frame_blob, frame_encoding = rows[100.0]
        self.assertEqual(frame_encoding, "frame_jpeg")
        self.assertIsNone(frame_mask_blob)
        decoded_frame = cv2.imdecode(
            np.frombuffer(frame_blob, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(decoded_frame.shape, frame.shape)

    def test_visual_storage_rejects_non_enum_priority_score(self):
        invalid = cluster("invalid", "invalid", (0, 0), score=60)
        with self.assertRaisesRegex(ValueError, "0, 25, 50, 75, or 100"):
            self.builder.add_nodes([invalid], frame_image=np.zeros((2, 2, 3)))

    def test_priority_100_requires_its_source_frame(self):
        highest = cluster("highest", "highest", (0, 0), score=100)
        with self.assertRaisesRegex(ValueError, "frame_image is required"):
            self.builder.add_nodes([highest])

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
