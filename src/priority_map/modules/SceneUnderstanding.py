import cv2
import numpy as np
import base64
import json
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from dotenv import load_dotenv
from priority_map.config.prompts import GPT_VISION_PROMPT
from priority_map.modules.scene_vlm import (
    create_scene_vlm_provider,
    parse_scene_model,
)


@dataclass
class SceneUnderstandingResult:
    labels: dict
    edge_intents: list[dict]


class SceneUnderstanding:
    def __init__(
        self,
        debug=False,
        model=None,
        provider_adapter=None,
    ):
        load_dotenv()
        model_config = parse_scene_model(model)
        self.model = model_config.model
        self.provider = model_config.provider
        self.provider_adapter = (
            provider_adapter
            if provider_adapter is not None
            else create_scene_vlm_provider(self.provider)
        )
        self.debug = debug
        self.vocabulary = {}
        self.vocabulary_alpha = 0.90
        self.scene_history = deque(maxlen=1)
        self.current_scene_dict = {}

    def _debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def _update_vocabulary(self, scene_dict):
        # Update vocabulary with EMA'd scores and return scene_dict with smoothed scores
        updated_dict = {}
        for label, label_info in scene_dict.items():
            score = float(label_info["score"])

            if label not in self.vocabulary:
                self.vocabulary[label] = score
            else:
                new_score = self.vocabulary_alpha * score + (1 - self.vocabulary_alpha) * self.vocabulary[label]
                self.vocabulary[label] = score

            updated_dict[label] = {
                "reasoning": label_info["reasoning"],
                "score": self.vocabulary[label]
            }

        return updated_dict

    def _merge_scene_dicts(self, current_dict):
        # Merge current scene with last 3, keeping most recent scores when labels overlap
        merged = {}
        for scene_dict in self.scene_history:
            for label, score in scene_dict.items():
                if label not in merged:
                    merged[label] = score

        for label, score in current_dict.items():
            merged[label] = score

        return merged

    def _loads_json_object(self, text):
        text = text.strip()

        text = (
            text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        def parse_json(candidate):
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            return json.loads(candidate)

        try:
            obj = parse_json(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise

            try:
                obj = parse_json(text[start:end + 1])
            except json.JSONDecodeError:
                self._debug_print("INVALID VLM JSON:")
                self._debug_print(text)
                raise

        if isinstance(obj, list):
            if len(obj) == 1 and isinstance(obj[0], dict):
                obj = obj[0]
            else:
                raise ValueError(
                    f"Expected JSON object, got list of length {len(obj)}"
                )

        return obj

    def _normalize_scene_response(self, scene_response):
        normalized = {}
        edge_intents = []

        if not isinstance(scene_response, dict):
            raise ValueError(f"Expected scene response, got {type(scene_response).__name__}")
        scene_dict = scene_response.get("labels")
        if not isinstance(scene_dict, dict):
            raise ValueError("Scene response must include a labels object")

        for key, label_info in scene_dict.items():
            if not isinstance(label_info, dict):
                raise ValueError(f"Expected label info object for {key!r}")
            if "reasoning" not in label_info or "score" not in label_info:
                raise ValueError(f"Scene label {key!r} must include reasoning and score")

            label = str(key).strip()
            reasoning = str(label_info["reasoning"]).strip()
            score = float(label_info["score"])

            if not label or not reasoning:
                raise ValueError(f"Scene label {key!r} must have a non-empty label and reasoning")

            normalized[label] = {
                "reasoning": reasoning,
                "score": score,
            }

        for key, label_info in scene_dict.items():
            source_label = str(key).strip()
            if source_label not in normalized:
                continue

            raw_edges = label_info.get("edges", [])
            if raw_edges is None:
                raw_edges = []
            if not isinstance(raw_edges, list):
                raw_edges = []

            for raw_edge in raw_edges:
                if not isinstance(raw_edge, dict):
                    continue
                text = self._normalize_edge_text(raw_edge.get("text", ""))
                to_label = str(raw_edge.get("to_label", "")).strip()
                to_node_id = str(raw_edge.get("to_node_id", "")).strip()
                if not text or (not to_label and not to_node_id):
                    continue

                edge_intent = {"source_label": source_label, "text": text}
                if to_label:
                    edge_intent["to_label"] = to_label
                if to_node_id:
                    edge_intent["to_node_id"] = to_node_id
                edge_intents.append(edge_intent)

        return SceneUnderstandingResult(normalized, edge_intents)

    def _normalize_edge_text(self, text):
        words = re.findall(r"[a-z0-9]+", str(text).strip().lower())
        if not 1 <= len(words) <= 2:
            return ""
        return "_".join(words)
    
    def _vlm_inference(self, image, task, recent_graph_context=None):
        image = cv2.resize(
            image,
            (512, 512),
            interpolation=cv2.INTER_AREA
        )

        _, buffer = cv2.imencode(".jpg", image)
        image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        prompt = GPT_VISION_PROMPT.format(
            task=task,
            vocabulary=json.dumps(self.vocabulary, indent=2),
            recent_graph_context=json.dumps(
                recent_graph_context or {"nodes": [], "spatial_edges": [], "model_edges": []},
                indent=2,
            ),
        )

        start = time.perf_counter()
        text = self.provider_adapter.analyze(
            self.model,
            prompt,
            image_b64,
        )
        end = time.perf_counter()
        self._debug_print(
            f"\nVLM inference time: {end - start:.2f} seconds "
            f"({self.provider}: {self.model})"
        )

        self._debug_print(text)

        scene_result = self._normalize_scene_response(self._loads_json_object(text))

        return scene_result

    def get_labels(self, image: np.ndarray, task: str, recent_graph_context=None):

        # return debug()

        try:
            scene_result = self._vlm_inference(
                image,
                task,
                recent_graph_context=recent_graph_context,
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            print(f"Warning: skipping VLM scene update: {exc}", file=sys.stderr)
            return None

        scene_dict = self._update_vocabulary(scene_result.labels)
        merged_dict = self._merge_scene_dicts(scene_dict)
        self.scene_history.append(scene_dict)

        # print(self.vocabulary)

        return SceneUnderstandingResult(merged_dict, scene_result.edge_intents)
        

def debug():
    return SceneUnderstandingResult(labels={
        "trees": {
            "reasoning": "Chosen as a major scene category, but scored at zero because trees are not useful for the example car-search task compared with roads or vehicles.",
            "score": 0,
        },

        "field": {
            "reasoning": "Chosen because fields are broad searchable terrain, but scored low because they are weak context for cars relative to roads, buildings, and vehicles.",
            "score": 30,
        },

        "road": {
            "reasoning": "Chosen because roads are strong car-search context and likely access paths, so they receive a high relevance score.",
            "score": 90,
        },

        "building": {
            "reasoning": "Chosen because buildings indicate human activity and possible nearby parking or access, giving them moderate relevance for a car-search task.",
            "score": 55,
        },

        "vehicle": {
            "reasoning": "Chosen because vehicles directly match the example car-search objective, so they receive the highest relevance score.",
            "score": 100,
        },
    }, edge_intents=[])
