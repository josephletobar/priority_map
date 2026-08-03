import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from priority_map.agent_cli import parse_args
from priority_map.modules.GraphAgent import ask_priority_map_db
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

    def test_answers_question_without_updating_scores(self):
        self.builder.set_original_task("original task")
        with patch(
            "priority_map.modules.GraphAgent.GraphAgent._call_model",
            return_value="The target is near target_0.",
        ):
            result = ask_priority_map_db(self.builder.db_path, "Where is the target?")

        self.assertEqual(result["original_task"], "original task")
        self.assertEqual(result["answer"], "The target is near target_0.")
        row = self.builder.cursor.execute(
            "SELECT score, color_b, color_g, color_r FROM nodes WHERE id = ?", ("target_0",)
        ).fetchone()
        self.assertEqual(row[0], 50.0)
        self.assertEqual(tuple(row[1:]), (0, 0, 0))

    def test_backfills_missing_original_task(self):
        with patch(
            "priority_map.modules.GraphAgent.GraphAgent._call_model",
            return_value="There is not enough information.",
        ):
            result = ask_priority_map_db(
                self.builder.db_path,
                "What is visible?",
                original_task="stored task",
            )

        self.assertEqual(result["original_task"], "stored task")
        row = self.builder.cursor.execute(
            "SELECT value FROM metadata WHERE key = 'original_task'"
        ).fetchone()
        self.assertEqual(row[0], "stored task")

    def test_sends_model_edges_to_agent(self):
        self.builder.cursor.execute(
            """
            INSERT INTO nodes
            (id, label, score, count, geo_pos_x, geo_pos_y, color_b, color_g, color_r)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("context_0", "context", 25.0, 1, 1.0, 1.0, 0, 0, 0),
        )
        self.builder.conn.commit()
        self.builder.insert_model_edges(
            [
                {
                    "source_id": "target_0",
                    "target_id": "context_0",
                    "text": "near entrance",
                }
            ]
        )

        captured = {}

        def fake_call(prompt):
            captured["prompt"] = prompt
            return "The target is near an entrance."

        with patch(
            "priority_map.modules.GraphAgent.GraphAgent._call_model",
            side_effect=fake_call,
        ):
            ask_priority_map_db(self.builder.db_path, "What is near the target?")

        self.assertIn('"model_edges": [', captured["prompt"])
        self.assertIn('"text": "near entrance"', captured["prompt"])
        self.assertIn('"created_by": "scene_vlm"', captured["prompt"])

    def test_agent_cli_accepts_scene_model_provider_argument(self):
        args = parse_args(
            [
                str(self.builder.db_path),
                "--scene-model",
                "openai:gpt-5.4",
                "--question",
                "What is visible?",
            ]
        )
        self.assertEqual(args.scene_model, "openai:gpt-5.4")
        self.assertEqual(args.question, "What is visible?")

    def test_rejects_unrelated_sqlite_database(self):
        unrelated_path = self.output_dir / "unrelated.db"
        sqlite3.connect(unrelated_path).close()
        with self.assertRaisesRegex(ValueError, "Not a PriorityMap graph database"):
            ask_priority_map_db(unrelated_path, "What is visible?")
