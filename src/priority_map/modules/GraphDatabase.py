import sqlite3
from pathlib import Path

import cv2
import numpy as np


class PriorityMapDatabase:
    """Direct SQLite access for the standalone PriorityMap database agent."""

    REQUIRED_TABLES = {"nodes", "edges"}

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        if not self.db_path.is_file():
            raise FileNotFoundError(f"PriorityMap database not found: {self.db_path}")
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._validate()
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.commit()

    def _validate(self):
        tables = {
            row[0]
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing = self.REQUIRED_TABLES - tables
        if missing:
            self.conn.close()
            raise ValueError(
                f"Not a PriorityMap graph database; missing table(s): {', '.join(sorted(missing))}"
            )

        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(nodes)")}
        required_columns = {"id", "label", "score", "color_b", "color_g", "color_r"}
        missing_columns = required_columns - columns
        if missing_columns:
            self.conn.close()
            raise ValueError(
                "Not a PriorityMap graph database; nodes table is missing column(s): "
                f"{', '.join(sorted(missing_columns))}"
            )

    def get_original_task(self):
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key = ?", ("original_task",)
        ).fetchone()
        return row[0] if row else None

    def set_original_task_if_missing(self, task):
        if not task:
            return self.get_original_task()
        self.conn.execute(
            "INSERT OR IGNORE INTO metadata (key, value) VALUES (?, ?)",
            ("original_task", str(task)),
        )
        self.conn.commit()
        return self.get_original_task()

    def graph_context(self):
        node_columns = {
            row[1] for row in self.conn.execute("PRAGMA table_info(nodes)")
        }
        observed_column = (
            "observed" if "observed" in node_columns else "0 AS observed"
        )
        nodes = [
            dict(row)
            for row in self.conn.execute(
                f"SELECT id, label, score, {observed_column} "
                "FROM nodes ORDER BY id"
            )
        ]
        edges = [
            dict(row)
            for row in self.conn.execute(
                "SELECT source_id, target_id, weight FROM edges ORDER BY source_id, target_id"
            )
        ]
        return nodes, edges

    @staticmethod
    def _score_color(score):
        heat_value = np.uint8([[np.clip(score, 0, 100) * 2.55]])
        return tuple(int(channel) for channel in cv2.applyColorMap(heat_value, cv2.COLORMAP_JET)[0, 0])

    def apply_score_deltas(self, updates):
        changes = []
        with self.conn:
            for update in updates or []:
                if not isinstance(update, dict):
                    continue
                node_id = str(update.get("node_id", "")).strip()
                try:
                    delta = int(update.get("delta"))
                except (TypeError, ValueError):
                    continue
                if not node_id:
                    continue

                row = self.conn.execute(
                    "SELECT score FROM nodes WHERE id = ?", (node_id,)
                ).fetchone()
                if row is None:
                    continue

                old_score = float(row[0])
                delta = max(-20, min(20, delta))
                new_score = max(0.0, min(100.0, old_score + delta))
                color_b, color_g, color_r = self._score_color(new_score)
                self.conn.execute(
                    """
                    UPDATE nodes
                    SET score = ?, color_b = ?, color_g = ?, color_r = ?
                    WHERE id = ?
                    """,
                    (new_score, color_b, color_g, color_r, node_id),
                )
                changes.append((node_id, old_score, new_score))
        return changes

    def close(self):
        self.conn.close()
