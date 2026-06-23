import cv2
import numpy as np
import base64
import json
import re
import time
from openai import OpenAI
from config.prompts import GPT_VISION_PROMPT

class SceneUnderstanding:
    def __init__(self):
        self.client = OpenAI()
        self.vocabulary = {}
        self.vocabulary_alpha = 0.15


    def _update_vocabulary(self, scene_dict):
        for label, label_info in scene_dict.items():
            score = float(label_info["score"])
            prompts = label_info["prompt"]

            if label not in self.vocabulary:
                self.vocabulary[label] = {
                    "prompt": prompts,
                    "score": score,
                }
            else:
                previous_score = self.vocabulary[label]["score"]
                new_score = (
                    self.vocabulary_alpha * score
                    + (1 - self.vocabulary_alpha) * previous_score
                )
                self.vocabulary[label]["score"] = new_score
                self.vocabulary[label]["prompt"] = prompts

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

        # return debug()

        image = cv2.resize(
            image,
            (224, 224),
            interpolation=cv2.INTER_AREA
        )

        _, buffer = cv2.imencode(".jpg", image)
        image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        prompt = GPT_VISION_PROMPT.format(
            task=task,
            vocabulary=json.dumps(self.vocabulary, indent=2),
        )

        start = time.perf_counter()
        response = self.client.responses.create(
            model="gpt-5-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "high"
                    },
                ],
            }],
        )
        end = time.perf_counter()
        print(f"\nGPT Vision inference time: {end - start:.2f} seconds")

        text = response.output_text
        print(text)

        scene_dict = self._normalize_scene_dict(self._loads_json_object(text))

        self._update_vocabulary(scene_dict)
        # print(self.vocabulary)111

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
            "score": 55,
        },

        "vehicle": {
            "prompt": "vehicle, car",
            "score": 100,
        },
    }
