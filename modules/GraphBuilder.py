import sqlite3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path


class GraphBuilder:
    MATCH_DISTANCE_THRESHOLD = 200 # pixels
    EDGE_THRESHOLD = 200 # pixels

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / "graph.db"

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
                geo_pos_y REAL NOT NULL
            )
        ''')

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

        self.cursor.execute('DELETE FROM nodes')
        self.cursor.execute('DELETE FROM edges')
        self.conn.commit()

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

    def _find_matching_node(self, base_label, x, y):
        """Find existing node with same base_label within MATCH_DISTANCE_THRESHOLD"""
        self.cursor.execute('SELECT id, geo_pos_x, geo_pos_y FROM nodes WHERE label LIKE ?', (f'{base_label}%',))
        for node_id, node_x, node_y in self.cursor.fetchall():
            distance = float(np.linalg.norm(np.array([x, y]) - np.array([node_x, node_y])))
            if distance <= self.MATCH_DISTANCE_THRESHOLD:
                return node_id
        return None

    def add_nodes(self, clustered_segmentations):
        """Add ClusteredSegmentations as nodes and create edges within 200px distance"""
        # print(f"add_nodes called with {len(clustered_segmentations)} segmentations")
        # for seg in clustered_segmentations:
        #     print(f"  {seg.label} at ({seg.geo_pos[0]:.1f}, {seg.geo_pos[1]:.1f})")

        new_node_ids = []
        

        for seg in clustered_segmentations:
            base_label = seg.label.rstrip('s')
            x, y = seg.geo_pos

            match = self._find_matching_node(base_label, x, y)
            if match:
                continue

            node_id = self._next_node_id(base_label)

            self.cursor.execute('''
                INSERT OR REPLACE INTO nodes
                (id, label, score, count, geo_pos_x, geo_pos_y)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (node_id, seg.label, seg.score, seg.count, float(x), float(y)))

            new_node_ids.append((node_id, x, y))

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

    def _get_all_nodes_and_edges(self):
        """Query all nodes and edges for visualization"""
        self.cursor.execute('SELECT id, geo_pos_x, geo_pos_y, score FROM nodes')
        nodes = {row[0]: {'pos': (row[1], row[2]), 'score': row[3]} for row in self.cursor.fetchall()}

        self.cursor.execute('SELECT source_id, target_id, weight FROM edges')
        edges = [(row[0], row[1], row[2]) for row in self.cursor.fetchall()]

        return nodes, edges

    def render_2d_graph_frame(self):
        nodes_data, edges = self._get_all_nodes_and_edges()

        if not nodes_data:
            return None

        # print(f"Drawing graph: {len(nodes_data)} nodes, {len(edges)} edges")
        # for src, dst, weight in edges:
        #     print(f"  Edge: {src}-{dst} weight {weight:.1f}")

        G = nx.Graph()
        G.add_nodes_from(nodes_data.keys())
        G.add_weighted_edges_from([(src, dst, w) for src, dst, w in edges])

        fig, ax = plt.subplots()

        pos = {node_id: data['pos'] for node_id, data in nodes_data.items()}
        node_sizes = [(nodes_data[node_id]['score'] + 5) * 10 for node_id in G.nodes()]
        node_colors = [nodes_data[node_id]['score'] for node_id in G.nodes()]

        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.jet,
        )

        for src, dst, weight in edges:
            if src not in pos or dst not in pos:
                continue

            x1, y1 = pos[src]
            x2, y2 = pos[dst]
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                f"{weight:.1f}",
                fontsize=8,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
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
