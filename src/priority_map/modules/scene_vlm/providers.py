import os
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MAX_OUTPUT_TOKENS = 1200


class SceneVlmProvider(Protocol):
    def analyze(self, model: str, prompt: str, image_base64: str) -> str:
        """Analyze one image and return the provider's response text."""
        ...


@dataclass(frozen=True)
class SceneModelConfig:
    provider: str
    model: str


class _ResponsesProvider:
    def analyze(self, model: str, prompt: str, image_base64: str) -> str:
        request_options = dict(
            model=model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": getattr(self, "image_detail", "low"),
                    },
                ],
            }],
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        request_options.update(self._extra_request_options())
        response = self.client.responses.create(**request_options)
        return response.output_text

    def _extra_request_options(self):
        return {}


class OpenAIProvider(_ResponsesProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.image_detail = os.getenv("SCENE_IMAGE_DETAIL", "low")
        self.reasoning_effort = os.getenv("SCENE_REASONING_EFFORT", "none")

    def _extra_request_options(self):
        return {
            "reasoning": {
                "effort": getattr(self, "reasoning_effort", "none"),
            },
            "text": {"verbosity": "low"},
        }


class OpenRouterProvider(_ResponsesProvider):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=OPENROUTER_BASE_URL,
        )
        self.image_detail = os.getenv("SCENE_IMAGE_DETAIL", "low")


class OllamaProvider:
    def __init__(self):
        self.client = OpenAI(
            api_key="ollama",
            base_url=OLLAMA_BASE_URL,
        )

    def analyze(self, model: str, prompt: str, image_base64: str) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_base64}",
                    },
                ],
            }],
            response_format={"type": "json_object"},
            max_tokens=MAX_OUTPUT_TOKENS,
            stream=False,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Ollama returned an empty scene-understanding response")
        return content


PROVIDER_REGISTRY = {
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
    "ollama": OllamaProvider,
}


def parse_scene_model(value: str | None) -> SceneModelConfig:
    supported = ", ".join(PROVIDER_REGISTRY)
    if not isinstance(value, str) or not value:
        raise ValueError(
            "scene_model is required in 'provider:model' format; "
            f"supported providers: {supported}"
        )

    if ":" not in value:
        raise ValueError(
            "scene_model must use 'provider:model' format; "
            f"supported providers: {supported}"
        )

    raw_provider, model = value.split(":", 1)
    provider = raw_provider.lower()
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unsupported scene VLM provider {raw_provider!r}; "
            f"supported providers: {supported}"
        )
    if not model or model.isspace():
        raise ValueError("scene_model must include a non-empty model after the provider")

    return SceneModelConfig(provider=provider, model=model)


def create_scene_vlm_provider(provider: str) -> SceneVlmProvider:
    try:
        provider_class = PROVIDER_REGISTRY[provider]
    except KeyError as exc:
        supported = ", ".join(PROVIDER_REGISTRY)
        raise ValueError(
            f"Unsupported scene VLM provider {provider!r}; "
            f"supported providers: {supported}"
        ) from exc
    return provider_class()
