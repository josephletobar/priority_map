import json
import os

import requests

from priority_map.config.prompts import GRAPH_AGENT_PROMPT
from priority_map.modules.GraphDatabase import PriorityMapDatabase


class GraphAgent:
    """One-shot reviewer for an already-created PriorityMap graph database."""

    def __init__(self, original_task, update, model=None, debug=False):
        self.original_task = original_task or "Unavailable"
        self.update = update
        self.debug = debug
        self.model = model or os.getenv("GRAPH_AGENT_MODEL", "phi4-mini-reasoning")
        self.ollama_url = os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
        self.keep_alive = os.getenv("GRAPH_AGENT_KEEP_ALIVE", "0")
        self.num_ctx = int(os.getenv("GRAPH_AGENT_NUM_CTX", "4096"))
        self.timeout = int(os.getenv("GRAPH_AGENT_TIMEOUT", "120"))

    def _build_prompt(self, nodes, edges):
        graph_json = json.dumps({"nodes": nodes, "edges": edges}, indent=2)
        if self.debug:
            print(f"\n=== PriorityMap Agent Graph ===\n{graph_json}\n=== End Graph ===\n")
        return GRAPH_AGENT_PROMPT.format(
            original_task=self.original_task,
            update=self.update,
            nodes_text=graph_json,
        )

    def _call_model(self, prompt):
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "keep_alive": self.keep_alive,
                "options": {"temperature": 0, "num_ctx": self.num_ctx},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw_output = response.json().get("response", "").strip()
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError(f"Graph agent response did not contain JSON: {raw_output}")
        return json.loads(raw_output[start:end]), raw_output

    def review(self, database):
        nodes, edges = database.graph_context()
        if not nodes:
            return {"reasoning": "The graph has no nodes to review.", "changes": []}
        response, _ = self._call_model(self._build_prompt(nodes, edges))
        changes = database.apply_score_deltas(response.get("updates", []))
        return {"reasoning": str(response.get("reasoning", "")), "changes": changes}


def review_priority_map_db(db_path, update, original_task=None, debug=False):
    if not str(update).strip():
        raise ValueError("A non-empty update is required.")
    database = PriorityMapDatabase(db_path)
    try:
        stored_task = database.set_original_task_if_missing(original_task)
        result = GraphAgent(stored_task, str(update), debug=debug).review(database)
        return {
            "db_path": database.db_path,
            "original_task": stored_task,
            **result,
        }
    finally:
        database.close()
