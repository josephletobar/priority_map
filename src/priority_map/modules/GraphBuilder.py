import sqlite3
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib
import networkx as nx
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from priority_map.scripts.cluster_segmentations import ClusteredSegmentation


class GraphBuilder:
    MATCH_DISTANCE_THRESHOLD = 200  # pixels
    EDGE_THRESHOLD = 600  # pixels

    def __init__(self, output_dir, debug=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / "graph.db"
        self.debug = debug

        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()
        self._init_db()

        self.last_2d_frame = None

    def _init_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                score REAL NOT NULL,
                count INTEGER NOT NULL,
                geo_pos_x REAL NOT NULL,
                geo_pos_y REAL NOT NULL,
                color_b INTEGER,
                color_g INTEGER,
                color_r INTEGER,
                mask_blob BLOB,
                reasoning TEXT,
                agent_reviewed INTEGER NOT NULL DEFAULT 0
            )
        ''')

        self.cursor.execute('PRAGMA table_info(nodes)')
        node_columns = {row[1] for row in self.cursor.fetchall()}
        if 'agent_reviewed' not in node_columns:
            self.cursor.execute(
                'ALTER TABLE nodes ADD COLUMN agent_reviewed INTEGER NOT NULL DEFAULT 0'
            )
        if 'mask_blob' not in node_columns:
            self.cursor.execute('ALTER TABLE nodes ADD COLUMN mask_blob BLOB')
        if 'reasoning' not in node_columns:
            self.cursor.execute('ALTER TABLE nodes ADD COLUMN reasoning TEXT')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                text TEXT NOT NULL,
                created_by TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, text),
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            )
        ''')

        self.cursor.execute('DELETE FROM model_edges')
        self.cursor.execute('DELETE FROM edges')
        self.cursor.execute('DELETE FROM nodes')
        self.conn.commit()

    def _encode_mask(self, mask):
        if mask is None:
            return None

        buffer = BytesIO()
        np.savez_compressed(buffer, mask=mask.astype(np.uint8))
        return sqlite3.Binary(buffer.getvalue())

    def _decode_mask(self, mask_blob):
        if not mask_blob:
            return np.ones((1, 1), dtype=np.uint8)

        try:
            with np.load(BytesIO(mask_blob), allow_pickle=False) as data:
                return data["mask"].astype(np.uint8)
        except Exception:
            return np.ones((1, 1), dtype=np.uint8)

    def _score_to_jet_color(self, score):
        heat_value = np.uint8([[np.clip(score, 0, 100) * 2.55]])
        color = cv2.applyColorMap(heat_value, cv2.COLORMAP_JET)[0, 0]
        return tuple(int(channel) for channel in color)

    def _to_float(self, value, default=0.0):
        if value is None:
            return default

        if isinstance(value, bytes):
            for dtype in (np.float64, np.float32, np.int64, np.int32):
                try:
                    decoded = np.frombuffer(value, dtype=dtype)
                    if decoded.size:
                        return float(decoded[0])
                except Exception:
                    pass
            return default

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _next_node_id(self, base_label):
        self.cursor.execute(
            '''
            SELECT MAX(CAST(SUBSTR(id, LENGTH(?) + 2) AS INTEGER))
            FROM nodes WHERE id LIKE ?
            ''',
            (base_label, f'{base_label}_%'),
        )
        row = self.cursor.fetchone()
        max_idx = row[0] if row[0] is not None else -1
        return f"{base_label}_{max_idx + 1}"

    def _find_matching_node(self, base_label, x, y):
        self.cursor.execute(
            'SELECT id, geo_pos_x, geo_pos_y FROM nodes WHERE label LIKE ?',
            (f'{base_label}%',),
        )
        for node_id, node_x, node_y in self.cursor.fetchall():
            distance = float(np.linalg.norm(np.array([x, y]) - np.array([node_x, node_y])))
            if distance <= self.MATCH_DISTANCE_THRESHOLD:
                return node_id
        return None

    def _node_exists(self, node_id):
        self.cursor.execute('SELECT 1 FROM nodes WHERE id = ?', (node_id,))
        return self.cursor.fetchone() is not None

    def _append_node_reasoning(self, node_id, reasoning):
        reasoning = str(reasoning or "").strip()
        if not reasoning:
            return

        self.cursor.execute('SELECT reasoning FROM nodes WHERE id = ?', (node_id,))
        row = self.cursor.fetchone()
        if not row:
            return

        existing = str(row[0] or "").strip()
        if not existing:
            updated = reasoning
        else:
            parts = [part.strip() for part in existing.split("\n") if part.strip()]
            if reasoning in parts:
                return
            updated = "\n".join(parts + [reasoning])

        self.cursor.execute(
            'UPDATE nodes SET reasoning = ? WHERE id = ?',
            (updated, node_id),
        )

    def add_nodes(self, clustered_segmentations):
        new_node_ids = []
        result = {
            "label_to_node_ids": {},
            "cluster_to_node_id": {},
        }

        for index, seg in enumerate(clustered_segmentations):
            base_label = seg.label
            x, y = seg.geo_pos

            match = self._find_matching_node(base_label, x, y)
            if match:
                self._append_node_reasoning(match, getattr(seg, "reasoning", ""))
                result["label_to_node_ids"].setdefault(base_label, []).append(match)
                result["cluster_to_node_id"][index] = match
                continue

            node_id = self._next_node_id(base_label)
            color = getattr(seg, 'color', None) or self._score_to_jet_color(seg.score)
            mask_blob = self._encode_mask(seg.mask)
            reasoning = str(getattr(seg, "reasoning", "") or "").strip()

            self.cursor.execute(
                '''
                INSERT OR REPLACE INTO nodes
                (id, label, score, count, geo_pos_x, geo_pos_y, color_b, color_g, color_r, mask_blob, reasoning, agent_reviewed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ''',
                (
                    node_id,
                    seg.label,
                    self._to_float(seg.score),
                    int(seg.count),
                    float(x),
                    float(y),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                    mask_blob,
                    reasoning,
                ),
            )
            new_node_ids.append((node_id, x, y))
            result["label_to_node_ids"].setdefault(base_label, []).append(node_id)
            result["cluster_to_node_id"][index] = node_id

        self.conn.commit()

        for node_id, x, y in new_node_ids:
            self.cursor.execute('SELECT id, geo_pos_x, geo_pos_y FROM nodes')
            for row_id, row_x, row_y in self.cursor.fetchall():
                if row_id == node_id:
                    continue

                distance = float(np.linalg.norm(np.array([x, y]) - np.array([row_x, row_y])))
                if distance <= self.EDGE_THRESHOLD:
                    src, dst = sorted([node_id, row_id])
                    self.cursor.execute(
                        '''
                        INSERT OR REPLACE INTO edges (source_id, target_id, weight)
                        VALUES (?, ?, ?)
                        ''',
                        (src, dst, distance),
                    )

        self.conn.commit()
        return result

    def _base_rows_to_clustered(self, node_ids):
        if not node_ids:
            return []

        placeholders = ','.join('?' for _ in node_ids)
        self.cursor.execute(
            f'''
            SELECT id, label, score, count, geo_pos_x, geo_pos_y, color_b, color_g, color_r, mask_blob, reasoning
            FROM nodes WHERE id IN ({placeholders})
            ''',
            tuple(node_ids),
        )

        clustered = []
        for row in self.cursor.fetchall():
            node_id, label, score, count, x, y, color_b, color_g, color_r, mask_blob, reasoning = row
            cluster = ClusteredSegmentation(
                label=label,
                centroid=(int(round(x)), int(round(y))),
                score=self._to_float(score),
                count=count,
                mask=self._decode_mask(mask_blob),
                geo_pos=(x, y),
                reasoning=reasoning or "",
                color=(color_b, color_g, color_r),
            )
            cluster.base_node_id = node_id
            clustered.append(cluster)

        return clustered

    def _get_nodes_and_edges(self, view=None):
        self.cursor.execute('''
            SELECT id, label, geo_pos_x, geo_pos_y, score, color_b, color_g, color_r, reasoning
            FROM nodes
        ''')
        nodes = {}
        for row in self.cursor.fetchall():
            nodes[row[0]] = {
                'label': row[1],
                'pos': (row[2], row[3]),
                'score': self._to_float(row[4]),
                'color': (
                    int(row[5] or 0),
                    int(row[6] or 0),
                    int(row[7] or 0),
                ),
                'reasoning': row[8] or "",
            }

        self.cursor.execute('SELECT source_id, target_id, weight FROM edges')
        edges = [(row[0], row[1], row[2]) for row in self.cursor.fetchall()]

        return nodes, edges, "base"

    def _get_model_edges(self):
        self.cursor.execute('SELECT source_id, target_id, text FROM model_edges')
        return [
            (source_id, target_id, text)
            for source_id, target_id, text in self.cursor.fetchall()
        ]

    def get_agent_graph_data(self, view=None):
        self.cursor.execute('SELECT id, label, score, reasoning, agent_reviewed FROM nodes')
        rows = [
            (node_id, label, self._to_float(score), reasoning or "", agent_reviewed)
            for node_id, label, score, reasoning, agent_reviewed in self.cursor.fetchall()
        ]
        self.cursor.execute('SELECT source_id, target_id, weight FROM edges')
        spatial_edges = self.cursor.fetchall()
        self.cursor.execute('SELECT source_id, target_id, text, created_by FROM model_edges')
        model_edges = self.cursor.fetchall()
        return rows, spatial_edges, model_edges, "base"

    def count_unreviewed_nodes(self, view=None):
        self.cursor.execute('SELECT COUNT(*) FROM nodes WHERE agent_reviewed = 0')
        return self.cursor.fetchone()[0]

    def mark_agent_reviewed(self, node_ids, view=None):
        if not node_ids:
            return

        self.cursor.executemany(
            'UPDATE nodes SET agent_reviewed = 1 WHERE id = ?',
            [(node_id,) for node_id in node_ids],
        )
        self.conn.commit()

    def apply_score_delta(self, node_id, delta, view=None):
        self.cursor.execute(
            'SELECT score, color_b, color_g, color_r FROM nodes WHERE id = ?',
            (node_id,),
        )
        row = self.cursor.fetchone()
        if not row:
            return None

        old_score, old_b, old_g, old_r = row
        old_score = self._to_float(old_score)
        try:
            delta = max(-20, min(20, int(delta)))
        except (TypeError, ValueError):
            return None

        new_score = max(0, min(100, old_score + delta))
        score_ratio = new_score / old_score if old_score > 0 else 1.0
        new_b = max(0, min(255, int((old_b or 0) * score_ratio)))
        new_g = max(0, min(255, int((old_g or 0) * score_ratio)))
        new_r = max(0, min(255, int((old_r or 0) * score_ratio)))

        self.cursor.execute(
            '''
            UPDATE nodes
            SET score = ?, color_b = ?, color_g = ?, color_r = ?
            WHERE id = ?
            ''',
            (new_score, new_b, new_g, new_r, node_id),
        )
        self.conn.commit()
        return old_score, new_score

    def _get_all_nodes_and_edges(self):
        nodes, edges, _ = self._get_nodes_and_edges()
        return nodes, edges

    def insert_model_edges(self, edge_specs, created_by):
        if not edge_specs:
            return []

        inserted = []
        for edge in edge_specs:
            if not isinstance(edge, dict):
                continue

            source_id = str(edge.get("source_id", "")).strip()
            target_id = str(edge.get("target_id", "")).strip()
            text = str(edge.get("text", "")).strip()
            if not source_id or not target_id or not text or source_id == target_id:
                continue
            if not self._node_exists(source_id) or not self._node_exists(target_id):
                continue

            src, dst = sorted([source_id, target_id])
            self.cursor.execute(
                '''
                INSERT OR IGNORE INTO model_edges (source_id, target_id, text, created_by)
                VALUES (?, ?, ?, ?)
                ''',
                (src, dst, text, created_by),
            )
            if self.cursor.rowcount:
                inserted.append((src, dst, text))

        self.conn.commit()
        return inserted

    def resolve_scene_edge_intents(self, edge_intents, add_result, recent_graph_context):
        if not edge_intents:
            return []

        recent_node_ids = {
            str(node.get("id", "")).strip()
            for node in (recent_graph_context or {}).get("nodes", [])
            if str(node.get("id", "")).strip()
        }
        label_to_node_ids = (add_result or {}).get("label_to_node_ids", {})
        resolved_edges = []

        for intent in edge_intents:
            source_label = str(intent.get("source_label", "")).strip()
            source_ids = label_to_node_ids.get(source_label, [])
            if not source_ids:
                continue

            text = str(intent.get("text", "")).strip()
            if not text:
                continue

            target_ids = []
            to_label = str(intent.get("to_label", "")).strip()
            if to_label:
                target_ids.extend(label_to_node_ids.get(to_label, []))

            to_node_id = str(intent.get("to_node_id", "")).strip()
            if to_node_id and to_node_id in recent_node_ids:
                target_ids.append(to_node_id)

            for source_id in source_ids:
                for target_id in target_ids:
                    if source_id == target_id:
                        continue
                    resolved_edges.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "text": text,
                    })

        return self.insert_model_edges(resolved_edges, created_by="scene_vlm")

    def get_recent_graph_context(self, limit=10):
        limit = max(0, int(limit))
        if limit == 0:
            return {"nodes": [], "edges": []}

        self.cursor.execute(
            '''
            SELECT id
            FROM nodes
            ORDER BY rowid DESC
            LIMIT ?
            ''',
            (limit,),
        )
        recent_ids = [row[0] for row in self.cursor.fetchall()]
        if not recent_ids:
            return {"nodes": [], "edges": []}

        recent_placeholders = ','.join('?' for _ in recent_ids)
        self.cursor.execute(
            f'''
            SELECT source_id, target_id, weight
            FROM edges
            WHERE source_id IN ({recent_placeholders})
               OR target_id IN ({recent_placeholders})
            ''',
            tuple(recent_ids + recent_ids),
        )
        edge_rows = self.cursor.fetchall()

        self.cursor.execute(
            f'''
            SELECT source_id, target_id, text, created_by
            FROM model_edges
            WHERE source_id IN ({recent_placeholders})
               OR target_id IN ({recent_placeholders})
            ''',
            tuple(recent_ids + recent_ids),
        )
        model_edge_rows = self.cursor.fetchall()

        node_ids = set(recent_ids)
        for source_id, target_id, _ in edge_rows:
            node_ids.add(source_id)
            node_ids.add(target_id)
        for source_id, target_id, _, _ in model_edge_rows:
            node_ids.add(source_id)
            node_ids.add(target_id)

        node_placeholders = ','.join('?' for _ in node_ids)
        self.cursor.execute(
            f'''
            SELECT id, label, score, reasoning
            FROM nodes
            WHERE id IN ({node_placeholders})
            ''',
            tuple(node_ids),
        )
        node_rows = self.cursor.fetchall()

        return {
            "nodes": [
                {
                    "id": node_id,
                    "label": label,
                    "score": self._to_float(score),
                    "reasoning": reasoning or "",
                }
                for node_id, label, score, reasoning in node_rows
            ],
            "edges": [
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "weight": self._to_float(weight),
                }
                for source_id, target_id, weight in edge_rows
            ],
            "model_edges": [
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "text": text,
                    "created_by": created_by,
                }
                for source_id, target_id, text, created_by in model_edge_rows
            ],
        }

    def render_2d_graph_frame(self, view=None):
        nodes_data, edges, _ = self._get_nodes_and_edges()
        model_edges = self._get_model_edges()

        if not nodes_data:
            return None

        G = nx.Graph()
        G.add_nodes_from(nodes_data.keys())
        G.add_weighted_edges_from([(src, dst, w) for src, dst, w in edges])
        model_edge_pairs = [
            (src, dst)
            for src, dst, _ in model_edges
            if src in nodes_data and dst in nodes_data
        ]
        G.add_edges_from(model_edge_pairs)

        spatial_graph = nx.Graph()
        spatial_graph.add_nodes_from(nodes_data.keys())
        spatial_graph.add_weighted_edges_from([(src, dst, w) for src, dst, w in edges])
        if len(spatial_graph.edges()) > 0:
            spatial_graph = nx.minimum_spanning_tree(spatial_graph, weight='weight')

        fig, ax = plt.subplots(figsize=(8, 5))

        pos = {node_id: data['pos'] for node_id, data in nodes_data.items()}
        node_sizes = [
            100 + (nodes_data[node_id]['score'] / 100.0) * 1000
            for node_id in G.nodes()
        ]
        node_colors = [
            tuple(channel / 255.0 for channel in nodes_data[node_id]['color'][::-1])
            for node_id in G.nodes()
        ]
        node_labels = {node_id: nodes_data[node_id]['label'] for node_id in G.nodes()}

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
        )
        nx.draw_networkx_labels(
            G,
            pos,
            labels=node_labels,
            ax=ax,
        )
        nx.draw_networkx_edges(
            spatial_graph,
            pos,
            ax=ax,
            edge_color="black",
            width=1.4,
        )
        if model_edge_pairs:
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                edgelist=model_edge_pairs,
                edge_color="#b000b8",
                style="dashed",
                width=1.8,
            )

        edge_labels = {
            (src, dst): int(round(data.get('weight', 0)))
            for src, dst, data in spatial_graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            spatial_graph,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=8,
        )
        model_edge_labels = {}
        for src, dst, text in model_edges:
            if src not in nodes_data or dst not in nodes_data:
                continue
            key = (src, dst)
            if key in model_edge_labels:
                model_edge_labels[key] = f"{model_edge_labels[key]}\n{text}"
            else:
                model_edge_labels[key] = text
        if model_edge_labels:
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=model_edge_labels,
                ax=ax,
                font_size=7,
                font_color="#8a008f",
                rotate=False,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": "#b000b8",
                    "alpha": 0.75,
                },
            )
        ax.axis("off")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3]
        self.last_2d_frame = rgb[:, :, ::-1].copy()
        plt.close(fig)

        return self.last_2d_frame

    def draw_2d_graph(self):
        self.render_2d_graph_frame()
        if self.last_2d_frame is not None:
            plt.imshow(self.last_2d_frame[:, :, ::-1])
            plt.axis("off")
            plt.show(block=False)

    def draw_3d_graph(self):
        return self.draw_2d_graph()

    def close(self):
        if self.conn:
            self.conn.close()
