import cv2
import numpy as np
import base64
import json
from config.prompts import VLM_PROMPT
from scripts.llama_request_helper import LlamaVlmClient
from ollama import chat
import time

class SceneUnderstanding:
    def __init__(self):
        self.model = None
        self.vocabulary = {}
        self.vocabulary_alpha = 0.25

        self.client = LlamaVlmClient(host="169.254.89.19", port=8600)
        # print(help.analyze(prompt="hi"))

    def _vocabulary_labels(self):
        return sorted(self.vocabulary.keys())

    def _update_vocabulary(self, scene_dict):
        for label, label_info in scene_dict.items():
            score = float(label_info["score"])

            if label not in self.vocabulary:
                self.vocabulary[label] = score
            else:
                previous_score = self.vocabulary[label]
                self.vocabulary[label] = (
                    self.vocabulary_alpha * score
                    + (1 - self.vocabulary_alpha) * previous_score
                )

    def _loads_json_object(self, text):
        text = text.strip()

        text = (
            text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise

            obj = json.loads(text[start:end + 1])

        if isinstance(obj, list):
            if len(obj) == 1 and isinstance(obj[0], dict):
                obj = obj[0]
            else:
                raise ValueError(
                    f"Expected JSON object, got list of length {len(obj)}"
                )

        return obj

    def _normalize_scene_dict(self, scene_dict):
        normalized = {}

        if not isinstance(scene_dict, dict):
            raise ValueError(f"Expected scene dictionary, got {type(scene_dict).__name__}")

        for key, label_info in scene_dict.items():
            if not isinstance(label_info, dict) or "score" not in label_info:
                continue

            score = float(label_info["score"])

            if "prompt" in label_info:
                label = str(key).strip()
                prompts = label_info["prompt"]
            elif "label" in label_info:
                label = str(label_info["label"]).strip()
                prompts = key
            else:
                continue

            if isinstance(prompts, str):
                prompts = [prompts]

            prompts = [
                str(prompt).strip()
                for prompt in prompts
                if str(prompt).strip()
            ]

            if not label or not prompts:
                continue

            normalized[label] = {
                "prompt": prompts,
                "score": score,
            }

        return normalized

    def get_labels(self, image: np.ndarray, task: str):

        # return debug()

        image = cv2.resize(
            image,
            (384, 384),
            interpolation=cv2.INTER_AREA
        )

        _, buffer = cv2.imencode(".jpg", image)

        image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        
        vlm_prompt = VLM_PROMPT.format(
            task=json.dumps(task, indent=2),
            vocabulary=json.dumps(self._vocabulary_labels(), indent=2),
        )
        
        start = time.perf_counter()
        response = chat(
            model="qwen2.5vl:3b",
            messages=[
                {
                    "role": "user",
                    "content": vlm_prompt,
                    "images": [image_b64],
                }
            ],
        )
        end = time.perf_counter()
        print(f"\nInference time: {end - start:.2f} seconds")

        text = response["message"]["content"]
        scene_dict = self._normalize_scene_dict(self._loads_json_object(text))

        self._update_vocabulary(scene_dict)

        print(text)

        return scene_dict
        

def debug():
    return {
        "trees": {
            "prompt": "dense forest, woodland, tree canopy, or heavily wooded area",
            "score": 0,
        },

        "field": {
            "prompt": "open field, grassland, meadow, pasture, lawn",
            "score": 30,
        },

        "road": {
            "prompt": "road, street, or highway",
            "score": 90,
        },

        "building": {
            "prompt": "building, house, facility",
            "score": 80,
        },

        "vehicle": {
            "prompt": "vehicle, car, truck, van, or motorized ground transportation",
            "score": 100,
        },
    }