import cv2
import numpy as np
import base64
import json
import re
from config.prompts import VLM_PROMPT, LLM_PROMPT
from scripts.llama_request_helper import LlamaVlmClient
from ollama import chat
import time

class SceneUnderstanding:
    def __init__(self):
        self.model = None
        self.vocabulary = {}
        self.vocabulary_alpha = 0.15

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
                print("INVALID VLM JSON:")
                print(text)
                raise

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

        return debug()

        image = cv2.resize(
            image,
            (224, 224),
            interpolation=cv2.INTER_AREA
        )

        _, buffer = cv2.imencode(".jpg", image)

        image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    
        # VLM CALL
        start = time.perf_counter()
        response = chat(
            model="qwen2.5vl:3b",
            messages=[
                {
                    "role": "user",
                    "content": VLM_PROMPT,
                    "images": [image_b64],
                }
            ],
        )
        end = time.perf_counter()
        print(f"\nVLM inference time: {end - start:.2f} seconds")
        text = response["message"]["content"]
        print(text)

        # LLM CALL
        llm_prompt = LLM_PROMPT.format(
            observation = text,
            task=json.dumps(task, indent=2),
            vocabulary=json.dumps(self._vocabulary_labels(), indent=2),
        )
        start = time.perf_counter()
        response = chat(
            model="qwen2.5vl:3b",
            messages=[
                {
                    "role": "user",
                    "content": llm_prompt,
                }
            ],
        )
        end = time.perf_counter()
        print(f"\nLLM inference time: {end - start:.2f} seconds")
        text = response["message"]["content"]
        print(text)

        scene_dict = self._normalize_scene_dict(self._loads_json_object(text))

        self._update_vocabulary(scene_dict)
        print(self.vocabulary)

        # time.sleep(500)


        return scene_dict
        

def debug():
    return {
        "trees": {
            "prompt": "trees",
            "score": 0,
        },

        "field": {
            "prompt": "field",
            "score": 30,
        },

        "road": {
            "prompt": "road",
            "score": 90,
        },

        "building": {
            "prompt": "structure, rooftops",
            "score": 80,
        },

        "vehicle": {
            "prompt": "vehicle, car",
            "score": 100,
        },
    }
