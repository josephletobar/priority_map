import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
from pathlib import Path
import json
from types import SimpleNamespace
from scripts.graph_helper import merge_similar
from modules.Segment import Segmentation

class GraphBuilder:
    def __init__(self, recorder=None, graph_path="graph.json"):
        self.G = nx.Graph()
        self.pos = {}
        self.recorder = recorder
        self.graph_path = Path(graph_path)
        self.last_2d_frame = None
        self.node_counts = {}

    # def update_object(self, obj):
    #     if obj.node_id not in self.G:
    #         return

    #     world_x, world_y, world_z = obj.world_pos
    #     self.G.nodes[obj.node_id].update(
    #         world_x=float(world_x),
    #         world_y=float(world_y),
    #         world_z=float(world_z),
    #         txt_embedding=obj.txt_embedding.tolist(),
    #         img_embedding=obj.img_embedding,
    #         confidence=obj.confidence,
    #         first_seen=obj.first_seen,
    #         last_seen=obj.last_seen,
    #     )
    #     self.pos[obj.node_id] = (
    #         world_x,
    #         world_z
    #     )

    # def _cluster(self, final_graph):
    #     nodes = list(final_graph.nodes())

    #     X = np.array([
    #         [
    #             final_graph.nodes[n]["world_x"],
    #             final_graph.nodes[n]["world_y"],
    #             final_graph.nodes[n]["world_z"],
    #         ]
    #         for n in nodes
    #     ])

    #     if len(X) == 0:
    #         return

    #     labels = DBSCAN(
    #         eps=6,
    #         min_samples=2
    #     ).fit_predict(X)

    #     for node, cluster_id in zip(nodes, labels):
    #         final_graph.nodes[node]["cluster"] = int(cluster_id)

    #     return final_graph

    def _build_topology(self):
        if self.G.number_of_edges() == 0:
            return self.G

        return nx.minimum_spanning_tree(self.G, weight="weight")

        # Old project logic below expected weighted world-position edges.
        # if len(self.G.nodes()) == 0:
        #     return self.G

        # threshold_graph = nx.Graph(
        #     (u, v, d)
        #     for u, v, d in self.G.edges(data=True)
        #     if d["weight"] < 1
        # )

        # mst = nx.minimum_spanning_tree(self.G, weight="weight")

        # final_graph = nx.compose(mst, threshold_graph)

        # final_graph = self._cluster(final_graph)

    #     # return final_graph

    # def _strip_embeddings(self, data):
    #     if isinstance(data, dict):
    #         return {
    #             key: self._strip_embeddings(value)
    #             for key, value in data.items()
    #             if key not in {"txt_embedding", "img_embedding"}
    #         }

    #     if isinstance(data, list):
    #         return [self._strip_embeddings(value) for value in data]

    #     return data

    def _figure_to_bgr(self):
        fig = plt.gcf()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3]
        return rgb[:, :, ::-1].copy()

    def write_2d_graph_frame(self):
        if self.recorder is None:
            return

        self.recorder.write(self.get_2d_graph_frame())

    def get_2d_graph_frame(self):
        if self.last_2d_frame is None:
            self.draw_2d_graph()

        return self.last_2d_frame

    def clear_recorder(self):
        self.recorder = None

    # def build_graph(self, nodes):

    #     for node in nodes:
    #         self.G.add_node(node.id, label=node.label, score=node.score)

        # for i, a in enumerate(nodes):
        #     for b in nodes[i + 1:]:
        #         self.G.add_edge(a.id, b.id)
    
    def _next_node_id(self, label):
        idx = self.node_counts.get(label, 0)
        self.node_counts[label] = idx + 1
        return f"{label}_{idx}"

    def _distance(self, a, b):
        pos_a = self.G.nodes[a].get("pos")
        pos_b = self.G.nodes[b].get("pos")

        if pos_a is None or pos_b is None:
            return float("inf")

        return float(np.linalg.norm(np.array(pos_a) - np.array(pos_b)))

    def _graph_nodes_as_segmentations(self):
        return [
            SimpleNamespace(
                id=node_id,
                mask=attrs["mask"],
                label=attrs["label"],
                score=attrs["score"],
                geo_pos=attrs.get("pos"),
            )
            for node_id, attrs in self.G.nodes(data=True)
        ]

    def build_graph(self, segmentations: list[Segmentation]):
        if not segmentations:
            return list(self.G.nodes)

        nodes = merge_similar(
            self._graph_nodes_as_segmentations() + list(segmentations)
        )

        self.G.clear()
        for node in nodes:
            if not node.id:
                node.id = self._next_node_id(node.label)

            self.G.add_node(
                node.id,
                mask=node.mask > 0,
                label=node.label,
                score=node.score,
                pos=node.geo_pos,
            )

        graph_nodes = list(self.G.nodes)
        for i, a in enumerate(graph_nodes):
            for b in graph_nodes[i + 1:]:
                self.G.add_edge(a, b, weight=self._distance(a, b))

        return graph_nodes

    @staticmethod
    def to_serializable_data(graph):
        data = json_graph.node_link_data(graph)
        # data = self._strip_embeddings(data)
        for node in data["nodes"]:
            node.pop("mask", None)
            node.pop("txt_embedding", None)
            node.pop("img_embedding", None)

        return data

    def save_graph(self, filename=None):
        data = self.to_serializable_data(self._build_topology())

        graph_path = Path(filename) if filename is not None else self.graph_path
        graph_path.parent.mkdir(parents=True, exist_ok=True)

        with graph_path.open("w") as f:
            json.dump(data, f, indent=2)

    def render_2d_graph_frame(self):
        final_graph = self._build_topology()
        fig, ax = plt.subplots()

        self.pos = {
            node_id: (attrs["pos"][0], attrs["pos"][1])
            for node_id, attrs in final_graph.nodes(data=True)
            if attrs.get("pos") is not None
        }

        missing_pos = [
            node_id
            for node_id in final_graph.nodes
            if node_id not in self.pos
        ]
        if missing_pos:
            fallback_pos = nx.spring_layout(final_graph.subgraph(missing_pos), seed=0)
            self.pos.update(fallback_pos)

        # node_colors = [
        #     final_graph.nodes[n]["cluster"]
        #     for n in final_graph.nodes()
        # ]

        nx.draw(
            final_graph,
            self.pos,
            ax=ax,
            with_labels=True,
            node_size=1000,
            # node_color=node_colors,
            # cmap=plt.cm.tab10
        )

        # nx.draw(final_graph, self.pos, with_labels=True, node_size=1000)

        for a, b, attrs in final_graph.edges(data=True):
            weight = attrs.get("weight")
            if weight is None or not np.isfinite(weight):
                continue

            x1, y1 = self.pos[a]
            x2, y2 = self.pos[b]
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                f"{weight:.1f}",
                fontsize=8,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
            )

        self.save_graph()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3]
        self.last_2d_frame = rgb[:, :, ::-1].copy()
        plt.close(fig)

        return self.last_2d_frame

    def draw_2d_graph(self):
        self.render_2d_graph_frame()
        plt.imshow(self.last_2d_frame[:, :, ::-1])
        plt.axis("off")
        plt.show(block=False)

        return self._build_topology()
    
    def draw_3d_graph(self):
        return self.draw_2d_graph()

        # Old project logic below expected world_x/world_y/world_z.
        plt.close("all")


        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        final_graph = self._build_topology()

        for node in final_graph.nodes():
            x = final_graph.nodes[node]["world_x"]
            y = final_graph.nodes[node]["world_y"]
            z = final_graph.nodes[node]["world_z"]

            cluster = final_graph.nodes[node].get("cluster", -1)
            
            # color = plt.cm.tab10(cluster % 10)
            # ax.scatter(
            #     x, y, z,
            #     color=color,
            #     s=150
            # )

            print(node, cluster)

            ax.text(x, y, z, node)

        for u, v in final_graph.edges():
            x1 = final_graph.nodes[u]["world_x"]
            y1 = final_graph.nodes[u]["world_y"]
            z1 = final_graph.nodes[u]["world_z"]

            x2 = final_graph.nodes[v]["world_x"]
            y2 = final_graph.nodes[v]["world_y"]
            z2 = final_graph.nodes[v]["world_z"]

            ax.plot([x1, x2], [y1, y2], [z1, z2], color="gray", alpha=0.4)

            weight = final_graph[u][v]["weight"]

            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                (z1 + z2) / 2,
                f"{weight:.2f}"
            )

        self.save_graph()

        plt.show(block=False)

        return final_graph
    
    
    
