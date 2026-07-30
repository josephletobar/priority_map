import cv2
import numpy as np
import base64
import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from priority_map.config.prompts import GPT_VISION_PROMPT


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_GEMMA_MODEL = "google/gemma-4-31b-it"
DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"


@dataclass
class SceneUnderstandingResult:
    labels: dict
    edge_intents: list[dict]


class SceneUnderstanding:
    def __init__(self, debug=False, model=None, api_key=None, base_url=None):
        load_dotenv()
        self.model = self._resolve_model(model)
        self.provider = self._provider_for_model(self.model)
        self.client = self._create_client(api_key=api_key, base_url=base_url)
        self.reasoning_effort = os.getenv("SCENE_REASONING_EFFORT", "none")
        self.image_detail = os.getenv("SCENE_IMAGE_DETAIL", "low")
        self.debug = debug
        self.vocabulary = {}
        self.vocabulary_alpha = 0.90
        self.scene_history = deque(maxlen=1)
        self.current_scene_dict = {}

    def _resolve_model(self, model):
        requested_model = model or os.getenv("SCENE_UNDERSTANDING_MODEL")
        if requested_model:
            requested_model = requested_model.strip()
            if requested_model.lower() in {"gemma", "openrouter"}:
                return DEFAULT_GEMMA_MODEL
            if requested_model.lower() == "openai":
                return DEFAULT_OPENAI_MODEL
            return requested_model

        return DEFAULT_OPENAI_MODEL

    def _provider_for_model(self, model):
        normalized = model.lower()
        if normalized.startswith("google/") or "gemma" in normalized:
            return "gemma"
        return "openai"

    def _create_client(self, api_key=None, base_url=None):
        if self.provider == "openai":
            client_kwargs = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
            if base_url:
                client_kwargs["base_url"] = base_url
            return OpenAI(**client_kwargs)

        return OpenAI(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url or os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        )

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
            if "score" not in label_info:
                raise ValueError(f"Scene label {key!r} must include score")

            label = str(key).strip()
            score = float(label_info["score"])

            if not label:
                raise ValueError(f"Scene label {key!r} must have a non-empty label")

            normalized[label] = {
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
        request_options = dict(
            model=self.model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": self.image_detail,
                    },
                ],
            }],
            max_output_tokens=1200,
        )
        if self.provider == "openai":
            request_options["reasoning"] = {"effort": self.reasoning_effort}
            request_options["text"] = {"verbosity": "low"}

        response = self.client.responses.create(**request_options)
        end = time.perf_counter()
        self._debug_print(
            f"\nVLM inference time: {end - start:.2f} seconds "
            f"({self.provider}: {self.model})"
        )

        text = response.output_text
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
            "score": 0,
        },

        "field": {
            "score": 30,
        },

        "road": {
            "score": 90,
        },

        "building": {
            "score": 55,
        },

        "vehicle": {
            "score": 100,
        },
    }, edge_intents=[])
