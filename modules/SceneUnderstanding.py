from openai import OpenAI
import cv2
import numpy as np
import base64
import json
from config.prompts import REASONING_PROMPT, VLM_PROMPT
from modules.llama_request_helper import LlamaVlmClient

class SceneUnderstanding:
    def __init__(self):
        self.model = None
        self.client = OpenAI()
        self.vocabulary = {}
        self.vocabulary_alpha = 0.25

        self.client = LlamaVlmClient(host="169.254.89.19", port=8600)
        # print(help.analyze(prompt="hi"))

    def _vocabulary_labels(self):
        return sorted(self.vocabulary.keys())

    def _update_vocabulary(self, labels):
        for label_info in labels.values():
            label = label_info["label"]
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

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise

            return json.loads(text[start:end + 1])

    def _normalize_labels(self, labels):
        normalized = {}

        for prompt, label_info in labels.items():
            if not isinstance(label_info, dict):
                continue

            if "label" not in label_info or "score" not in label_info:
                continue

            normalized[prompt] = {
                "label": str(label_info["label"]),
                "score": float(label_info["score"]),
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
        
        # Stage 1: VLM Perception

        observations = self._loads_json_object(
            self.client.analyze(prompt=VLM_PROMPT, image_base64=image_b64)
        )["observations"]
        
        # response = self.client.responses.create(
        #     model="gpt-4.1-mini",
        #     input=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 {
        #                     "type": "input_text",
        #                     "text": VLM_PROMPT,
        #                 },
        #                 {
        #                     "type": "input_image",
        #                     "image_url": f"data:image/jpeg;base64,{image_b64}",
        #                 },
        #             ],
        #         }
        #     ],
        # )
        # observations = self._loads_json_object(response.output_text)["observations"]

        # Stage 2: Instruction / Reasoning Model

        reasoning_prompt = REASONING_PROMPT.format(
            task=task,
            observations=json.dumps(observations, indent=2),
            vocabulary=json.dumps(self._vocabulary_labels(), indent=2),
        )

        # response = self.client.responses.create(
        #     model="o4-mini",
        #     input=reasoning_prompt,
        # )

        response = self.client.analyze(prompt=reasoning_prompt)

        labels = self._normalize_labels(self._loads_json_object(response))

        self._update_vocabulary(labels)

        print("OBSERVATIONS:")
        print(observations)

        print("LABELS:")
        print(labels)

        return labels
        

def debug():
    return {
        "dense forest, woodland, tree canopy, or heavily wooded area": {
            "label": "trees",
            "score": 0,
        },

        "open field, grassland, meadow, pasture, lawn": {
            "label": "field",
            "score": 30,
        },

        "road, street, or highway": {
            "label": "road",
            "score": 90,
        },

        "building, house, facility": {
            "label": "building",
            "score": 80,
        },

        "vehicle, car, truck, van, or motorized ground transportation": {
            "label": "vehicle",
            "score": 100,
        },
    }
