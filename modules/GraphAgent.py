import json
import os
import time
import requests
import networkx as nx
from config.prompts import GRAPH_AGENT_PROMPT


class GraphAgent:
    def __init__(
        self,
        graph_builder,
        task_description,
        node_growth_threshold=30,
        model=None,
    ):
        self.graph_builder = graph_builder
        self.task_description = task_description
        self.node_growth_threshold = node_growth_threshold
        self.model = model or os.getenv("GRAPH_AGENT_MODEL", "phi4-mini-reasoning")
        self.ollama_url = os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
        self.keep_alive = os.getenv("GRAPH_AGENT_KEEP_ALIVE", "0")
        self.num_ctx = int(os.getenv("GRAPH_AGENT_NUM_CTX", "4096"))
        self.timeout = int(os.getenv("GRAPH_AGENT_TIMEOUT", "120"))

    def should_run(self):
        """Check if enough unreviewed nodes exist to trigger reasoning."""
        cursor = self.graph_builder.cursor
        cursor.execute('SELECT COUNT(*) FROM nodes WHERE agent_reviewed = 0')
        unreviewed_node_count = cursor.fetchone()[0]

        return unreviewed_node_count >= self.node_growth_threshold

    def _get_context(self):
        """Query eligible nodes and return MST edges for minimal spatial structure."""
        cursor = self.graph_builder.cursor

        cursor.execute('SELECT id, score, agent_reviewed FROM nodes')
        rows = cursor.fetchall()
        all_nodes = {row[0]: row[1] for row in rows}
        unreviewed_nodes = {row[0] for row in rows if row[2] == 0}

        if not all_nodes or not unreviewed_nodes:
            return None, None

        cursor.execute('SELECT source_id, target_id, weight FROM edges')
        edges = cursor.fetchall()

        G = nx.Graph()
        G.add_nodes_from(all_nodes.keys())
        G.add_weighted_edges_from(edges)

        eligible_node_ids = set(unreviewed_nodes)
        for node_id in unreviewed_nodes:
            eligible_node_ids.update(
                nx.single_source_shortest_path_length(G, node_id, cutoff=2).keys()
            )

        eligible_nodes = {
            node_id: all_nodes[node_id]
            for node_id in eligible_node_ids
            if node_id in all_nodes
        }
        eligible_graph = G.subgraph(eligible_nodes.keys()).copy()

        mst_edges = []
        if len(eligible_graph.edges()) > 0:
            mst = nx.minimum_spanning_tree(eligible_graph, weight='weight')
            mst_edges = [(u, v, d['weight']) for u, v, d in mst.edges(data=True)]

        return eligible_nodes, mst_edges

    def _build_prompt(self, all_nodes, edges):
        """Build LLM prompt with eligible nodes and MST edges."""
        node_list = [
            {
                "id": node_id,
                "score": round(score)
            }
            for node_id, score in sorted(all_nodes.items())
        ]
        edge_list = [
            {
                "from": src,
                "from_score": round(all_nodes[src]),
                "to": dst,
                "to_score": round(all_nodes[dst]),
                "dist": round(weight)
            }
            for src, dst, weight in edges
            if src in all_nodes and dst in all_nodes
        ]

        graph_json = json.dumps(
            {
                "nodes": node_list,
                "edges": edge_list
            },
            indent=2
        )

        print(f"\n=== Sent Graph ===")
        print(graph_json)
        print(f"=== End Graph ===\n")

        return GRAPH_AGENT_PROMPT.format(
            task_description=self.task_description,
            nodes_text=graph_json
        )

    def _mark_reviewed(self, node_ids):
        """Mark prompted nodes as reviewed after a successful model response."""
        if not node_ids:
            return

        cursor = self.graph_builder.cursor
        cursor.executemany(
            'UPDATE nodes SET agent_reviewed = 1 WHERE id = ?',
            [(node_id,) for node_id in node_ids]
        )
        self.graph_builder.conn.commit()

    def _call_local_model(self, prompt):
        """Call the configured Ollama model and return parsed JSON."""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "keep_alive": self.keep_alive,
                    "options": {
                        "temperature": 0,
                        "num_ctx": self.num_ctx,
                    },
                },
                timeout=self.timeout
            )

            if response.status_code != 200:
                return None, f"API error: {response.status_code}: {response.text}"

            result = response.json()
            output = result.get("response", "").strip()

            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                try:
                    return json.loads(output[json_start:json_end]), output
                except json.JSONDecodeError:
                    return None, output
            return None, output

        except Exception as e:
            return None, str(e)

    def _update_scores(self, updates):
        """Apply score deltas and update colors in SQLite."""
        if not updates:
            return []

        cursor = self.graph_builder.cursor
        changes = []

        for update in updates:
            if not isinstance(update, dict):
                continue

            node_id = update.get("node_id")
            delta = update.get("delta")

            if not node_id or delta is None:
                continue

            cursor.execute(
                'SELECT score, color_b, color_g, color_r FROM nodes WHERE id = ?',
                (node_id,)
            )
            row = cursor.fetchone()
            if not row:
                continue

            old_score, old_b, old_g, old_r = row
            try:
                delta = max(-20, min(20, int(delta)))
            except (TypeError, ValueError):
                continue

            new_score = max(0, min(100, old_score + delta))
            score_ratio = new_score / old_score if old_score > 0 else 1.0
            new_b = max(0, min(255, int(old_b * score_ratio)))
            new_g = max(0, min(255, int(old_g * score_ratio)))
            new_r = max(0, min(255, int(old_r * score_ratio)))

            cursor.execute(
                '''UPDATE nodes SET score = ?, color_b = ?, color_g = ?, color_r = ?
                   WHERE id = ?''',
                (new_score, new_b, new_g, new_r, node_id)
            )
            changes.append((node_id, old_score, new_score))

        self.graph_builder.conn.commit()
        return changes

    def update_priorities(self):
        """Main agent loop: get context, prompt LLM, update scores."""
        all_nodes, edges = self._get_context()
        if all_nodes is None:
            return

        prompt = self._build_prompt(all_nodes, edges)

        start_time = time.time()
        response, raw_output = self._call_local_model(prompt)
        elapsed = time.time() - start_time

        print(f"\n=== Graph Agent ===")
        print(f"Model: {self.model}")
        print(f"Inference time: {elapsed:.2f} seconds\n")

        if response is None:
            print(f"Raw:\n{raw_output}\n")
            print(f"=== End Graph Agent ===\n")
            return

        reasoning = response.get("reasoning", "")
        if reasoning:
            print(f"Reasoning: {reasoning}\n")

        updates = response.get("updates", [])
        changes = self._update_scores(updates)
        self._mark_reviewed(all_nodes.keys())

        if changes:
            for node_id, old, new in changes:
                print(f"  {node_id}: {old:.0f}→{new:.0f}")
        else:
            print(f"Updates: None")

        print(f"=== End Graph Agent ===\n")
