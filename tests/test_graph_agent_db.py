import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from priority_map.modules.GraphAgent import review_priority_map_db
from priority_map.modules.GraphBuilder import GraphBuilder


class StandaloneGraphAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.builder = GraphBuilder(self.output_dir)
        self.builder.cursor.execute(
            """
            INSERT INTO nodes
            (id, label, score, count, geo_pos_x, geo_pos_y, color_b, color_g, color_r, mask_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("target_0", "target", 50.0, 1, 0.0, 0.0, 0, 0, 0, None),
        )
        self.builder.conn.commit()

    def tearDown(self):
        self.builder.close()
        self.temp_dir.cleanup()

    def test_preserves_original_task_and_updates_scores_and_colors(self):
        self.builder.set_original_task("original task")
        with patch(
            "priority_map.modules.GraphAgent.GraphAgent._call_model",
            return_value=({"reasoning": "updated", "updates": [{"node_id": "target_0", "delta": 20}]}, "{}"),
        ):
            result = review_priority_map_db(self.builder.db_path, "new information")

        self.assertEqual(result["original_task"], "original task")
        self.assertEqual(result["changes"], [("target_0", 50.0, 70.0)])
        row = self.builder.cursor.execute(
            "SELECT score, color_b, color_g, color_r FROM nodes WHERE id = ?", ("target_0",)
        ).fetchone()
        self.assertEqual(row[0], 70.0)
        self.assertNotEqual(tuple(row[1:]), (0, 0, 0))

    def test_backfills_missing_original_task(self):
        with patch(
            "priority_map.modules.GraphAgent.GraphAgent._call_model",
            return_value=({"reasoning": "none", "updates": []}, "{}"),
        ):
            result = review_priority_map_db(
                self.builder.db_path,
                "new information",
                original_task="stored task",
            )

        self.assertEqual(result["original_task"], "stored task")
        row = self.builder.cursor.execute(
            "SELECT value FROM metadata WHERE key = 'original_task'"
        ).fetchone()
        self.assertEqual(row[0], "stored task")

    def test_rejects_unrelated_sqlite_database(self):
        unrelated_path = self.output_dir / "unrelated.db"
        sqlite3.connect(unrelated_path).close()
        with self.assertRaisesRegex(ValueError, "Not a PriorityMap graph database"):
            review_priority_map_db(unrelated_path, "new information")
