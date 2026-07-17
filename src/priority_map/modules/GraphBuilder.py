import sqlite3
from io import BytesIO
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from priority_map.scripts.cluster_segmentations import (
    ClusteredSegmentation,
    semantic_clustering_with_members,
)


class GraphBuilder:
    MATCH_DISTANCE_THRESHOLD = 200 # pixels
    EDGE_THRESHOLD = 600 # pixels
    SEMANTIC_K_NEAREST = 2
    SEMANTIC_SCORE_WEIGHT_GAMMA = 2.0
    MODEL_LAYOUT_K_SCALE = 3.0
    MODEL_LAYOUT_ATTRACTION_SCALE = 0.25
    MODEL_LAYOUT_ITERATIONS = 200
    MODEL_LAYOUT_SEED = 42
    MODEL_LAYOUT_GRAVITY = 5.0
    MODEL_LAYOUT_MIN_WEIGHT = 0.03
    MODEL_LAYOUT_MAX_WEIGHT = 1.0
    MODEL_EDGE_BASE_LENGTH = 0.12
    MODEL_EDGE_LENGTH_PER_CHARACTER = 0.012
    MODEL_EDGE_SEPARATION_PASSES = 40

    def __init__(self, output_dir, graph_view="base", debug=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / "graph.db"
        self.graph_view = graph_view
        self.debug = debug

        # print(f"GraphBuilder DB path: {self.db_path}")
        # print(f"DB file exists before connect: {self.db_path.exists()}")

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
                mask_blob BLOB
            )
        ''')

        self.cursor.execute('PRAGMA table_info(nodes)')
        node_columns = {row[1] for row in self.cursor.fetchall()}
        if 'mask_blob' not in node_columns:
            self.cursor.execute(
                'ALTER TABLE nodes ADD COLUMN mask_blob BLOB'
            )

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

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_nodes (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                score REAL NOT NULL,
                count INTEGER NOT NULL,
                geo_pos_x REAL NOT NULL,
                geo_pos_y REAL NOT NULL,
                color_b INTEGER,
                color_g INTEGER,
                color_r INTEGER,
                mask_blob BLOB
            )
        ''')

        self.cursor.execute('PRAGMA table_info(semantic_nodes)')
        semantic_node_columns = {row[1] for row in self.cursor.fetchall()}
        if 'mask_blob' not in semantic_node_columns:
            self.cursor.execute(
                'ALTER TABLE semantic_nodes ADD COLUMN mask_blob BLOB'
            )

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES semantic_nodes(id),
                FOREIGN KEY (target_id) REFERENCES semantic_nodes(id)
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_node_members (
                semantic_node_id TEXT NOT NULL,
                base_node_id TEXT NOT NULL,
                PRIMARY KEY (semantic_node_id, base_node_id),
                FOREIGN KEY (semantic_node_id) REFERENCES semantic_nodes(id),
                FOREIGN KEY (base_node_id) REFERENCES nodes(id)
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')
        self.conn.commit()

    def set_original_task(self, task):
        """Persist the immutable task that created this PriorityMap DB."""
        self.cursor.execute(
            'INSERT OR IGNORE INTO metadata (key, value) VALUES (?, ?)',
            ('original_task', str(task)),
        )
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

        with np.load(BytesIO(mask_blob), allow_pickle=False) as data:
            return data["mask"].astype(np.uint8)

    def _score_to_jet_color(self, score):
        heat_value = np.uint8([[np.clip(score, 0, 100) * 2.55]])
        color = cv2.applyColorMap(heat_value, cv2.COLORMAP_JET)[0, 0]
        return tuple(int(channel) for channel in color)

    def _semantic_score_weight(self, score):
        x = np.clip(self._to_float(score), 0.0, 100.0) / 100.0
        return x ** self.SEMANTIC_SCORE_WEIGHT_GAMMA

    def _to_float(self, value, default=0.0):
        if value is None:
            return default

        if isinstance(value, bytes):
            for dtype in (np.float64, np.float32, np.int64, np.int32):
                decoded = np.frombuffer(value, dtype=dtype)
                if decoded.size:
                    return float(decoded[0])
            return default

        return float(value)

    def _get_node_pos(self, node_id):
        self.cursor.execute('SELECT geo_pos_x, geo_pos_y FROM nodes WHERE id = ?', (node_id,))
        row = self.cursor.fetchone()
        if row:
            return row[0], row[1]
        return None, None

    def _next_node_id(self, base_label):
        """Get next unique node_id for label by querying max from DB"""
        self.cursor.execute('''
            SELECT MAX(CAST(SUBSTR(id, LENGTH(?) + 2) AS INTEGER))
            FROM nodes WHERE id LIKE ?
        ''', (base_label, f'{base_label}_%'))
        row = self.cursor.fetchone()
        max_idx = row[0] if row[0] is not None else -1
        node_id = f"{base_label}_{max_idx + 1}"
        # print(f"_next_node_id({base_label}): max_idx={max_idx}, returning {node_id}")
        return node_id

    def _next_semantic_node_id(self):
        self.cursor.execute('''
            SELECT MAX(CAST(SUBSTR(id, LENGTH(?) + 2) AS INTEGER))
            FROM semantic_nodes WHERE id LIKE ?
        ''', ('semantic', 'semantic_%'))
        row = self.cursor.fetchone()
        max_idx = row[0] if row[0] is not None else -1
        return f"semantic_{max_idx + 1}"

    def _find_matching_node(self, base_label, x, y):
        """Find existing node with same base_label within MATCH_DISTANCE_THRESHOLD"""
        self.cursor.execute('SELECT id, geo_pos_x, geo_pos_y FROM nodes WHERE label LIKE ?', (f'{base_label}%',))
        for node_id, node_x, node_y in self.cursor.fetchall():
            distance = float(np.linalg.norm(np.array([x, y]) - np.array([node_x, node_y])))
            if distance <= self.MATCH_DISTANCE_THRESHOLD:
                return node_id
        return None

    def assign_existing_node_ids(self, clustered_segmentations):
        """Attach matching DB node IDs without creating or updating graph nodes."""
        for seg in clustered_segmentations:
            x, y = seg.geo_pos
            seg.node_id = self._find_matching_node(seg.label, x, y)

    def add_nodes(self, clustered_segmentations):
        """Add ClusteredSegmentations as nodes and create edges within 200px distance"""
        # print(f"add_nodes called with {len(clustered_segmentations)} segmentations")
        # for seg in clustered_segmentations:
        #     print(f"  {seg.label} at ({seg.geo_pos[0]:.1f}, {seg.geo_pos[1]:.1f})")

        new_node_ids = []
        result = {"label_to_node_ids": {}}
        

        for seg in clustered_segmentations:
            base_label = seg.label
            x, y = seg.geo_pos

            match = self._find_matching_node(base_label, x, y)
            if match:
                seg.node_id = match
                source_label = getattr(seg, "source_label", None) or base_label
                result["label_to_node_ids"].setdefault(source_label, []).append(match)
                continue

            node_id = self._next_node_id(base_label)

            color = getattr(seg, 'color', None) or (0, 0, 0)
            mask_blob = self._encode_mask(seg.mask)
            self.cursor.execute('''
                INSERT OR REPLACE INTO nodes
                (id, label, score, count, geo_pos_x, geo_pos_y, color_b, color_g, color_r, mask_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (node_id, seg.label, self._to_float(seg.score), int(seg.count), float(x), float(y),
                  int(color[0]), int(color[1]), int(color[2]), mask_blob))

            new_node_ids.append((node_id, x, y))
            seg.node_id = node_id
            source_label = getattr(seg, "source_label", None) or base_label
            result["label_to_node_ids"].setdefault(source_label, []).append(node_id)

        self.conn.commit()

        threshold = self.EDGE_THRESHOLD 
        for node_id, x, y in new_node_ids:
            self.cursor.execute('SELECT id, geo_pos_x, geo_pos_y FROM nodes')
            for row_id, row_x, row_y in self.cursor.fetchall():
                if row_id == node_id:
                    continue

                distance = float(np.linalg.norm(np.array([x, y]) - np.array([row_x, row_y])))
                if distance <= threshold:
                    src, dst = sorted([node_id, row_id])
                    self.cursor.execute('''
                        INSERT OR REPLACE INTO edges (source_id, target_id, weight)
                        VALUES (?, ?, ?)
                    ''', (src, dst, distance))

        self.conn.commit()
        if new_node_ids:
            self.update_semantic_nodes([node_id for node_id, _, _ in new_node_ids])
        return result

    def _node_exists(self, node_id):
        self.cursor.execute('SELECT 1 FROM nodes WHERE id = ?', (node_id,))
        return self.cursor.fetchone() is not None

    def insert_model_edges(self, model_edges, created_by="scene_vlm"):
        inserted = []
        for edge in model_edges or []:
            if not isinstance(edge, dict):
                continue
            source_id = str(edge.get("source_id", "")).strip()
            target_id = str(edge.get("target_id", "")).strip()
            edge_text = str(edge.get("text", "")).strip()
            if not source_id or not target_id or not edge_text or source_id == target_id:
                continue
            if not self._node_exists(source_id) or not self._node_exists(target_id):
                continue

            source_id, target_id = sorted((source_id, target_id))
            self.cursor.execute(
                '''
                INSERT OR IGNORE INTO model_edges (source_id, target_id, text, created_by)
                VALUES (?, ?, ?, ?)
                ''',
                (source_id, target_id, edge_text, created_by),
            )
            if self.cursor.rowcount:
                inserted.append((source_id, target_id, edge_text))
        self.conn.commit()
        return inserted

    def resolve_scene_edge_intents(self, edge_intents, add_result, recent_graph_context):
        label_to_node_ids = (add_result or {}).get("label_to_node_ids", {})
        recent_node_ids = {
            str(node.get("id", "")).strip()
            for node in (recent_graph_context or {}).get("nodes", [])
            if isinstance(node, dict) and str(node.get("id", "")).strip()
        }
        resolved = []
        for intent in edge_intents or []:
            if not isinstance(intent, dict):
                continue
            source_ids = label_to_node_ids.get(str(intent.get("source_label", "")).strip(), [])
            edge_text = str(intent.get("text", "")).strip()
            if not source_ids or not edge_text:
                continue

            target_ids = []
            to_label = str(intent.get("to_label", "")).strip()
            if to_label:
                target_ids.extend(label_to_node_ids.get(to_label, []))
            to_node_id = str(intent.get("to_node_id", "")).strip()
            if to_node_id and to_node_id in recent_node_ids:
                target_ids.append(to_node_id)

            for source_id in source_ids:
                for target_id in dict.fromkeys(target_ids):
                    resolved.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "text": edge_text,
                    })
        return self.insert_model_edges(resolved, created_by="scene_vlm")

    def _get_model_edges(self):
        self.cursor.execute('SELECT source_id, target_id, text FROM model_edges')
        return self.cursor.fetchall()

    def _base_rows_to_clustered(self, node_ids):
        if not node_ids:
            return []

        placeholders = ','.join('?' for _ in node_ids)
        self.cursor.execute(
            f'''
            SELECT id, label, score, count, geo_pos_x, geo_pos_y, color_b, color_g, color_r, mask_blob
            FROM nodes WHERE id IN ({placeholders})
            ''',
            tuple(node_ids),
        )

        clustered = []
        for row in self.cursor.fetchall():
            node_id, label, score, count, x, y, color_b, color_g, color_r, mask_blob = row
            cluster = ClusteredSegmentation(
                label=label,
                centroid=(int(round(x)), int(round(y))),
                score=self._to_float(score),
                count=count,
                mask=self._decode_mask(mask_blob),
                geo_pos=(x, y),
                color=(color_b, color_g, color_r),
            )
            cluster.base_node_id = node_id
            clustered.append(cluster)

        return clustered

    def _semantic_rows_to_clustered(self, semantic_node_ids):
        if not semantic_node_ids:
            return []

        placeholders = ','.join('?' for _ in semantic_node_ids)
        self.cursor.execute(
            f'''
            SELECT id, label, score, count, geo_pos_x, geo_pos_y, color_b, color_g, color_r, mask_blob
            FROM semantic_nodes WHERE id IN ({placeholders})
            ''',
            tuple(semantic_node_ids),
        )

        clustered = []
        for row in self.cursor.fetchall():
            node_id, label, score, count, x, y, color_b, color_g, color_r, mask_blob = row
            cluster = ClusteredSegmentation(
                label=label,
                centroid=(int(round(x)), int(round(y))),
                score=self._to_float(score),
                count=count,
                mask=self._decode_mask(mask_blob),
                geo_pos=(x, y),
                color=(color_b, color_g, color_r),
            )
            cluster.semantic_node_id = node_id
            clustered.append(cluster)

        return clustered

    def _nearest_semantic_node_ids(self, base_node_ids, k=None):
        if not base_node_ids:
            return set()

        k = k or self.SEMANTIC_K_NEAREST
        placeholders = ','.join('?' for _ in base_node_ids)
        self.cursor.execute(
            f'SELECT id, geo_pos_x, geo_pos_y FROM nodes WHERE id IN ({placeholders})',
            tuple(base_node_ids),
        )
        base_rows = self.cursor.fetchall()
        if not base_rows:
            return set()

        self.cursor.execute('SELECT id, geo_pos_x, geo_pos_y FROM semantic_nodes')
        semantic_rows = self.cursor.fetchall()
        if not semantic_rows:
            return set()

        nearest_ids = set()
        for _, base_x, base_y in base_rows:
            distances = []
            for semantic_id, semantic_x, semantic_y in semantic_rows:
                distance = float(np.linalg.norm(
                    np.array([base_x, base_y]) - np.array([semantic_x, semantic_y])
                ))
                distances.append((distance, semantic_id))

            nearest_ids.update(
                semantic_id
                for _, semantic_id in sorted(distances, key=lambda item: item[0])[:k]
            )

        return nearest_ids

    def _insert_semantic_node(self, semantic_cluster, semantic_node_id=None):
        semantic_node_id = semantic_node_id or self._next_semantic_node_id()
        mask_blob = self._encode_mask(semantic_cluster.mask)
        x, y = semantic_cluster.geo_pos

        self.cursor.execute('''
            INSERT INTO semantic_nodes
            (id, label, score, count, geo_pos_x, geo_pos_y, color_b, color_g, color_r, mask_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            semantic_node_id,
            semantic_cluster.label,
            0.0,
            int(semantic_cluster.count),
            float(x),
            float(y),
            0,
            0,
            0,
            mask_blob,
        ))

        return semantic_node_id

    def _update_semantic_node(self, semantic_node_id, semantic_cluster, score=None):
        score = self._to_float(semantic_cluster.score if score is None else score)
        color = self._score_to_jet_color(score)
        mask_blob = self._encode_mask(semantic_cluster.mask)
        x, y = semantic_cluster.geo_pos

        self.cursor.execute('''
            UPDATE semantic_nodes
            SET label = ?,
                score = ?,
                count = ?,
                geo_pos_x = ?,
                geo_pos_y = ?,
                color_b = ?,
                color_g = ?,
                color_r = ?,
                mask_blob = ?
            WHERE id = ?
        ''', (
            semantic_cluster.label,
            score,
            int(semantic_cluster.count),
            float(x),
            float(y),
            int(color[0]),
            int(color[1]),
            int(color[2]),
            mask_blob,
            semantic_node_id,
        ))

    def _semantic_counts(self, semantic_node_ids):
        if not semantic_node_ids:
            return {}

        placeholders = ','.join('?' for _ in semantic_node_ids)
        self.cursor.execute(
            f'SELECT id, count FROM semantic_nodes WHERE id IN ({placeholders})',
            tuple(semantic_node_ids),
        )
        return {row[0]: row[1] for row in self.cursor.fetchall()}

    def _choose_semantic_survivor(self, semantic_node_ids):
        counts = self._semantic_counts(semantic_node_ids)
        return sorted(
            semantic_node_ids,
            key=lambda node_id: (-counts.get(node_id, 0), node_id),
        )[0]

    def _merge_semantic_memberships(self, survivor_id, merged_semantic_ids):
        for semantic_node_id in merged_semantic_ids:
            if semantic_node_id == survivor_id:
                continue

            self.cursor.execute(
                'SELECT base_node_id FROM semantic_node_members WHERE semantic_node_id = ?',
                (semantic_node_id,),
            )
            member_ids = [row[0] for row in self.cursor.fetchall()]
            self.cursor.executemany(
                '''
                INSERT OR IGNORE INTO semantic_node_members
                (semantic_node_id, base_node_id) VALUES (?, ?)
                ''',
                [(survivor_id, member_id) for member_id in member_ids],
            )
            self.cursor.execute(
                'DELETE FROM semantic_node_members WHERE semantic_node_id = ?',
                (semantic_node_id,),
            )
            self.cursor.execute(
                'DELETE FROM semantic_nodes WHERE id = ?',
                (semantic_node_id,),
            )

    def _assign_base_memberships(self, semantic_node_id, base_node_ids):
        if not base_node_ids:
            return

        placeholders = ','.join('?' for _ in base_node_ids)
        self.cursor.execute(
            f'DELETE FROM semantic_node_members WHERE base_node_id IN ({placeholders})',
            tuple(base_node_ids),
        )

        self.cursor.executemany(
            '''
            INSERT OR IGNORE INTO semantic_node_members
            (semantic_node_id, base_node_id) VALUES (?, ?)
            ''',
            [(semantic_node_id, base_node_id) for base_node_id in base_node_ids],
        )

    def _semantic_cluster_from_base_members(self, base_clusters, label=None):
        avg_centroid = tuple(np.mean([cluster.centroid for cluster in base_clusters], axis=0).astype(int))
        valid_geo_positions = [cluster.geo_pos for cluster in base_clusters if cluster.geo_pos is not None]
        avg_geo_pos = (
            tuple(np.mean(valid_geo_positions, axis=0))
            if valid_geo_positions
            else avg_centroid
        )

        merged_mask = np.logical_or.reduce([cluster.mask for cluster in base_clusters]).astype(np.uint8)
        representative = max(base_clusters, key=lambda cluster: cluster.score)

        return ClusteredSegmentation(
            label=label or representative.label,
            centroid=avg_centroid,
            score=0,
            count=sum(cluster.count for cluster in base_clusters),
            mask=merged_mask,
            geo_pos=avg_geo_pos,
            color=None,
        )

    def _recompute_semantic_node_from_members(self, semantic_node_id, label=None):
        self.cursor.execute(
            'SELECT base_node_id FROM semantic_node_members WHERE semantic_node_id = ?',
            (semantic_node_id,),
        )
        base_node_ids = [row[0] for row in self.cursor.fetchall()]
        if not base_node_ids:
            return

        base_clusters = self._base_rows_to_clustered(base_node_ids)
        if not base_clusters:
            return

        weighted_score_sum = 0.0
        total_weight = 0.0
        for cluster in base_clusters:
            score_value = self._to_float(cluster.score)
            weight = self._semantic_score_weight(score_value)
            weighted_score_sum += score_value * weight
            total_weight += weight

        score = (
            weighted_score_sum / total_weight
            if total_weight > 0
            else float(np.mean([cluster.score for cluster in base_clusters]))
        )
        semantic_cluster = self._semantic_cluster_from_base_members(base_clusters, label=label)
        self._update_semantic_node(semantic_node_id, semantic_cluster, score=score)

    def _rebuild_semantic_edges(self):
        self.cursor.execute('DELETE FROM semantic_edges')
        self.cursor.execute('SELECT base_node_id, semantic_node_id FROM semantic_node_members')
        base_to_semantic = {row[0]: row[1] for row in self.cursor.fetchall()}

        self.cursor.execute('SELECT source_id, target_id, weight FROM edges')
        edge_weights = {}
        for source_id, target_id, weight in self.cursor.fetchall():
            semantic_source = base_to_semantic.get(source_id)
            semantic_target = base_to_semantic.get(target_id)
            if not semantic_source or not semantic_target or semantic_source == semantic_target:
                continue

            src, dst = sorted([semantic_source, semantic_target])
            key = (src, dst)
            if key not in edge_weights or weight < edge_weights[key]:
                edge_weights[key] = weight

        self.cursor.executemany(
            '''
            INSERT OR REPLACE INTO semantic_edges (source_id, target_id, weight)
            VALUES (?, ?, ?)
            ''',
            [(src, dst, weight) for (src, dst), weight in edge_weights.items()],
        )

    def update_semantic_nodes(self, new_node_ids):
        if not new_node_ids:
            return

        new_node_ids = list(dict.fromkeys(new_node_ids))
        nearest_semantic_ids = self._nearest_semantic_node_ids(new_node_ids)
        candidate_clusters = (
            self._base_rows_to_clustered(new_node_ids)
            + self._semantic_rows_to_clustered(nearest_semantic_ids)
        )
        if not candidate_clusters:
            return

        semantic_groups = semantic_clustering_with_members(
            candidate_clusters,
            debug=self.debug,
        )
        for semantic_cluster, member_clusters in semantic_groups:
            base_node_ids = {
                member.base_node_id
                for member in member_clusters
                if hasattr(member, 'base_node_id')
            }
            semantic_node_ids = {
                member.semantic_node_id
                for member in member_clusters
                if hasattr(member, 'semantic_node_id')
            }

            if not base_node_ids and len(semantic_node_ids) <= 1:
                continue

            if semantic_node_ids:
                target_semantic_id = self._choose_semantic_survivor(semantic_node_ids)
                self._merge_semantic_memberships(target_semantic_id, semantic_node_ids)
            else:
                target_semantic_id = self._insert_semantic_node(semantic_cluster)

            self._assign_base_memberships(target_semantic_id, base_node_ids)
            self._recompute_semantic_node_from_members(
                target_semantic_id,
                label=semantic_cluster.label,
            )

        self._rebuild_semantic_edges()
        self.conn.commit()

    def _view_tables(self, view=None):
        view = view or self.graph_view
        if view == "semantic":
            self.cursor.execute('SELECT COUNT(*) FROM semantic_nodes')
            if self.cursor.fetchone()[0] > 0:
                return "semantic_nodes", "semantic_edges", "semantic"
        return "nodes", "edges", "base"

    def _get_nodes_and_edges(self, view=None):
        """Query nodes and edges for a graph view."""
        node_table, edge_table, resolved_view = self._view_tables(view)
        self.cursor.execute(f'''
            SELECT id, label, geo_pos_x, geo_pos_y, score, color_b, color_g, color_r
            FROM {node_table}
        ''')
        nodes = {
            row[0]: {
                'label': row[1],
                'pos': (row[2], row[3]),
                'score': self._to_float(row[4]),
                'color': (row[5], row[6], row[7]),
            }
            for row in self.cursor.fetchall()
        }

        self.cursor.execute(f'SELECT source_id, target_id, weight FROM {edge_table}')
        edges = [(row[0], row[1], row[2]) for row in self.cursor.fetchall()]

        return nodes, edges, resolved_view

    def apply_score_delta(self, node_id, delta, view=None):
        node_table, _, _ = self._view_tables(view)
        self.cursor.execute(
            f'SELECT score, color_b, color_g, color_r FROM {node_table} WHERE id = ?',
            (node_id,),
        )
        row = self.cursor.fetchone()
        if not row:
            return None

        old_score, old_b, old_g, old_r = row
        old_score = self._to_float(old_score)
        delta = max(-20, min(20, int(delta)))

        new_score = max(0, min(100, old_score + delta))
        score_ratio = new_score / old_score if old_score > 0 else 1.0
        new_b = max(0, min(255, int(old_b * score_ratio)))
        new_g = max(0, min(255, int(old_g * score_ratio)))
        new_r = max(0, min(255, int(old_r * score_ratio)))

        self.cursor.execute(
            f'''
            UPDATE {node_table}
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

    def get_recent_graph_context(self, limit=10):
        limit = max(0, int(limit))
        if limit == 0:
            return {"nodes": [], "spatial_edges": [], "model_edges": []}

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
            return {"nodes": [], "spatial_edges": [], "model_edges": []}

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
            SELECT id, label, score
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
                }
                for node_id, label, score in node_rows
            ],
            "spatial_edges": [
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

    def _model_edge_labels(self, model_edges, nodes_data):
        grouped = {}
        for source_id, target_id, edge_text in model_edges:
            if source_id not in nodes_data or target_id not in nodes_data:
                continue
            key = tuple(sorted((source_id, target_id)))
            grouped.setdefault(key, []).append(edge_text)
        return {key: "\n".join(labels) for key, labels in grouped.items()}

    def _apply_model_layout_weights(self, graph, edge_labels):
        if not edge_labels:
            return

        lengths = {
            edge: max(1, sum(len(line) for line in label.splitlines()))
            for edge, label in edge_labels.items()
        }
        median_length = max(1.0, float(np.median(list(lengths.values()))))
        for source_id, target_id in graph.edges():
            key = tuple(sorted((source_id, target_id)))
            label_length = lengths.get(key, median_length)
            graph[source_id][target_id]["layout_weight"] = float(np.clip(
                self.MODEL_LAYOUT_ATTRACTION_SCALE * median_length / label_length,
                self.MODEL_LAYOUT_MIN_WEIGHT,
                self.MODEL_LAYOUT_MAX_WEIGHT,
            ))

    def _separate_model_edge_labels(self, positions, edge_labels):
        positions = {
            node_id: np.asarray(position, dtype=float).copy()
            for node_id, position in positions.items()
        }
        for _ in range(self.MODEL_EDGE_SEPARATION_PASSES):
            for (source_id, target_id), label in edge_labels.items():
                if source_id not in positions or target_id not in positions:
                    continue

                rendered_length = max(1, sum(len(line) for line in label.splitlines()))
                desired_length = (
                    self.MODEL_EDGE_BASE_LENGTH
                    + self.MODEL_EDGE_LENGTH_PER_CHARACTER * rendered_length
                )
                delta = positions[target_id] - positions[source_id]
                distance = float(np.linalg.norm(delta))
                if distance >= desired_length:
                    continue

                if distance < 1e-9:
                    delta = np.array([1.0, 0.0])
                    distance = 1.0
                direction = delta / distance
                shift = direction * ((desired_length - distance) / 2.0)
                positions[source_id] -= shift
                positions[target_id] += shift
        return positions

    def _model_layout(self, graph, edge_labels):
        if not graph.nodes:
            return {}

        k = self.MODEL_LAYOUT_K_SCALE / np.sqrt(max(1, graph.number_of_nodes()))
        positions = nx.spring_layout(
            graph,
            k=k,
            iterations=self.MODEL_LAYOUT_ITERATIONS,
            seed=self.MODEL_LAYOUT_SEED,
            weight="layout_weight",
            method="energy",
            gravity=self.MODEL_LAYOUT_GRAVITY,
        )
        return self._separate_model_edge_labels(positions, edge_labels)

    def render_2d_graph_frame(self, view="model"):
        if view not in {"model", "spatial"}:
            raise ValueError(f"Unknown graph view: {view}")

        nodes_data, spatial_edges, _ = self._get_nodes_and_edges("base")

        if not nodes_data:
            return None

        G = nx.Graph()
        G.add_nodes_from(nodes_data.keys())
        edge_labels = {}
        if view == "spatial":
            G.add_weighted_edges_from(
                (source_id, target_id, weight)
                for source_id, target_id, weight in spatial_edges
                if source_id in nodes_data and target_id in nodes_data
            )
            if G.number_of_edges():
                G = nx.minimum_spanning_tree(G, weight="weight")
            pos = {node_id: data["pos"] for node_id, data in nodes_data.items()}
            edge_labels = {
                (source_id, target_id): str(int(round(data.get("weight", 0))))
                for source_id, target_id, data in G.edges(data=True)
            }
        else:
            model_edges = self._get_model_edges()
            edge_labels = self._model_edge_labels(model_edges, nodes_data)
            G.add_edges_from(edge_labels.keys())
            self._apply_model_layout_weights(G, edge_labels)
            pos = self._model_layout(G, edge_labels)

        figure_size = (14, 7) if view == "model" else (8, 5)
        fig, ax = plt.subplots(figsize=figure_size)

        node_sizes = [
            100 + (nodes_data[node_id]['score'] / 100.0) * 1000
            for node_id in G.nodes()
        ]
        node_colors = [tuple(c / 255.0 for c in nodes_data[node_id]['color'][::-1]) for node_id in G.nodes()]
        node_labels = {node_id: nodes_data[node_id]['label'] for node_id in G.nodes()}

        nx.draw(
            G,
            pos,
            ax=ax,
            labels=node_labels,
            with_labels=True,
            node_size=node_sizes,
            node_color=node_colors,
            font_size=14,
        )
        if edge_labels:
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                ax=ax,
                font_size=11 if view == "model" else 10,
                rotate=False,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1},
            )
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
