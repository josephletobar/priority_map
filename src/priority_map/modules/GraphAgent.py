import json
import os

from dotenv import load_dotenv

from priority_map.config.prompts import GRAPH_AGENT_QA_PROMPT
from priority_map.modules.GraphDatabase import PriorityMapDatabase
from priority_map.modules.scene_vlm import (
    create_scene_vlm_provider,
    parse_scene_model,
)


class GraphAgent:
    """One-shot question-answering agent for a PriorityMap graph database."""

    def __init__(
        self,
        original_task,
        question,
        model=None,
        debug=False,
        scene_model=None,
    ):
        load_dotenv()
        self.original_task = original_task or "Unavailable"
        self.question = question
        self.debug = debug
        if scene_model is None:
            scene_model = f"ollama:{model or os.getenv('GRAPH_AGENT_MODEL', 'phi4-mini-reasoning')}"
        model_config = parse_scene_model(scene_model)
        self.provider = model_config.provider
        self.model = model_config.model
        self.provider_adapter = create_scene_vlm_provider(self.provider)

    def _build_prompt(self, nodes, edges, model_edges=None, visuals=None):
        graph_json = json.dumps(
            {
                "nodes": nodes,
                "edges": edges,
                "model_edges": model_edges,
                "visuals": visuals or [],
            },
            indent=2,
        )
        if self.debug:
            print(f"\n=== PriorityMap Agent Graph ===\n{graph_json}\n=== End Graph ===\n")
        return GRAPH_AGENT_QA_PROMPT.format(
            original_task=self.original_task,
            question=self.question,
            nodes_text=graph_json,
        )

    def _call_model(
        self,
        prompt,
        images_base64=None,
        image_mime_types=None,
        image_labels=None,
    ):
        raw_output = self.provider_adapter.analyze_many(
            self.model,
            prompt,
            images_base64 or [],
            image_mime_types=image_mime_types,
            json_mode=False,
            image_labels=image_labels,
        ).strip()
        if not raw_output:
            raise ValueError("Graph agent returned an empty answer")
        return raw_output

    def answer(self, database):
        nodes, edges = database.graph_context()
        if not nodes:
            return {"answer": "The graph contains no nodes to answer from."}
        model_edges = database.get_model_edges()
        visual_inputs = database.get_visual_inputs()
        visual_refs = [
            {
                "image_index": index,
                "table": visual["table"],
                "node_id": visual["node_id"],
                "kind": visual["kind"],
                "mime_type": visual["mime_type"],
            }
            for index, visual in enumerate(visual_inputs)
        ]
        visual_labels = [
            (
                f"Attached visual {visual['image_index']} belongs to "
                f"{visual['table']}/{visual['node_id']} and is a {visual['kind']}."
            )
            for visual in visual_refs
        ]
        prompt = self._build_prompt(nodes, edges, model_edges, visual_refs)
        if visual_inputs:
            answer = self._call_model(
                prompt,
                [visual["image_base64"] for visual in visual_inputs],
                [visual["mime_type"] for visual in visual_inputs],
                visual_labels,
            )
        else:
            answer = self._call_model(prompt)
        return {"answer": answer}


def ask_priority_map_db(
    db_path,
    question,
    original_task=None,
    debug=False,
    scene_model=None,
):
    if not str(question).strip():
        raise ValueError("A non-empty question is required.")
    database = PriorityMapDatabase(db_path)
    try:
        stored_task = database.set_original_task_if_missing(original_task)
        result = GraphAgent(
            stored_task,
            str(question),
            debug=debug,
            scene_model=scene_model,
        ).answer(database)
        return {
            "db_path": database.db_path,
            "original_task": stored_task,
            "question": str(question),
            **result,
        }
    finally:
        database.close()


def review_priority_map_db(
    db_path,
    update,
    original_task=None,
    debug=False,
    scene_model=None,
):
    """Backward-compatible alias for the question-answering graph agent."""
    return ask_priority_map_db(
        db_path,
        question=update,
        original_task=original_task,
        debug=debug,
        scene_model=scene_model,
    )
