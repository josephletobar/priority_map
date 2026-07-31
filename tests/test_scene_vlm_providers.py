import contextlib
import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from priority_map.cli import parse_args
from priority_map.modules.SceneUnderstanding import SceneUnderstanding
from priority_map.modules.scene_vlm.providers import (
    OLLAMA_BASE_URL,
    OPENROUTER_BASE_URL,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    PROVIDER_REGISTRY,
    create_scene_vlm_provider,
    parse_scene_model,
)


SCENE_RESPONSE = """{
    "labels": {
        "vehicle": {
            "reasoning": "Directly supports the mission, with no limiting context.",
            "score": 100,
            "edges": []
        }
    }
}"""


class SceneModelConfigTests(unittest.TestCase):
    def test_accepts_only_registered_providers(self):
        self.assertEqual(
            set(PROVIDER_REGISTRY),
            {"openai", "openrouter", "ollama"},
        )
        for provider in PROVIDER_REGISTRY:
            with self.subTest(provider=provider):
                config = parse_scene_model(f"{provider}:provider-owned-model")
                self.assertEqual(config.provider, provider)
                self.assertEqual(config.model, "provider-owned-model")

    def test_provider_is_case_insensitive_and_model_is_unchanged(self):
        config = parse_scene_model("OlLaMa:org/model:version:quant")

        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.model, "org/model:version:quant")

    def test_rejects_missing_malformed_and_unsupported_values(self):
        invalid_values = (
            None,
            "",
            "gpt-5.4",
            "openai",
            "openai:",
            "openai:   ",
            ":model",
            "anthropic:model",
        )

        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_scene_model(value)

    def test_cli_requires_provider_model_and_preserves_the_value(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args([])

        args = parse_args(["--scene-model", "ollama:custom/model:q4"])
        self.assertEqual(args.scene_model, "ollama:custom/model:q4")


class ProviderRegistryTests(unittest.TestCase):
    @patch("priority_map.modules.scene_vlm.providers.OpenAI")
    def test_factory_routes_each_provider_to_its_adapter(self, _openai):
        expected_types = {
            "openai": OpenAIProvider,
            "openrouter": OpenRouterProvider,
            "ollama": OllamaProvider,
        }

        for provider, expected_type in expected_types.items():
            with self.subTest(provider=provider):
                adapter = create_scene_vlm_provider(provider)
                self.assertIsInstance(adapter, expected_type)

    def test_factory_rejects_unregistered_provider(self):
        with self.assertRaisesRegex(ValueError, "Unsupported scene VLM provider"):
            create_scene_vlm_provider("other")

    @patch("priority_map.modules.scene_vlm.providers.OpenAI")
    def test_adapters_construct_provider_specific_clients(self, openai):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "openai-key",
            "OPENROUTER_API_KEY": "router-key",
        }):
            OpenAIProvider()
            OpenRouterProvider()
            OllamaProvider()

        self.assertEqual(openai.call_args_list[0].kwargs, {"api_key": "openai-key"})
        self.assertEqual(
            openai.call_args_list[1].kwargs,
            {"api_key": "router-key", "base_url": OPENROUTER_BASE_URL},
        )
        self.assertEqual(
            openai.call_args_list[2].kwargs,
            {"api_key": "ollama", "base_url": OLLAMA_BASE_URL},
        )


class ProviderTransportTests(unittest.TestCase):
    def test_cloud_adapters_use_responses_with_unchanged_models(self):
        for provider_type in (OpenAIProvider, OpenRouterProvider):
            with self.subTest(provider=provider_type.__name__):
                adapter = provider_type.__new__(provider_type)
                adapter.client = MagicMock()
                adapter.client.responses.create.return_value = SimpleNamespace(
                    output_text=SCENE_RESPONSE
                )

                text = adapter.analyze("org/model:release", "prompt", "encoded-image")

                self.assertEqual(text, SCENE_RESPONSE)
                request = adapter.client.responses.create.call_args.kwargs
                self.assertEqual(request["model"], "org/model:release")
                image = request["input"][0]["content"][1]
                self.assertEqual(
                    image["image_url"],
                    "data:image/jpeg;base64,encoded-image",
                )

    def test_ollama_uses_multimodal_chat_completions(self):
        adapter = OllamaProvider.__new__(OllamaProvider)
        adapter.client = MagicMock()
        adapter.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=SCENE_RESPONSE))]
        )

        text = adapter.analyze("custom/model:q4", "prompt", "encoded-image")

        self.assertEqual(text, SCENE_RESPONSE)
        request = adapter.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "custom/model:q4")
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertFalse(request["stream"])
        content = request["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "prompt"})
        self.assertEqual(
            content[1]["image_url"],
            "data:image/jpeg;base64,encoded-image",
        )
        self.assertFalse(adapter.client.responses.create.called)

    def test_provider_errors_are_not_substituted_or_suppressed(self):
        adapter = OllamaProvider.__new__(OllamaProvider)
        adapter.client = MagicMock()
        adapter.client.chat.completions.create.side_effect = RuntimeError(
            "model is not installed"
        )

        with self.assertRaisesRegex(RuntimeError, "model is not installed"):
            adapter.analyze("missing-model", "prompt", "encoded-image")


class SceneUnderstandingRoutingTests(unittest.TestCase):
    def test_scene_understanding_uses_injected_adapter_and_normalizes_output(self):
        adapter = MagicMock()
        adapter.analyze.return_value = SCENE_RESPONSE
        scene = SceneUnderstanding(
            model="ollama:custom/model:q4",
            provider_adapter=adapter,
        )

        result = scene.get_labels(
            np.zeros((4, 4, 3), dtype=np.uint8),
            "Find vehicles",
        )

        self.assertEqual(scene.provider, "ollama")
        self.assertEqual(scene.model, "custom/model:q4")
        self.assertEqual(result.labels["vehicle"]["score"], 100)
        model, prompt, image_base64 = adapter.analyze.call_args.args
        self.assertEqual(model, "custom/model:q4")
        self.assertIn("Mission objective: Find vehicles", prompt)
        self.assertTrue(image_base64)

    def test_scene_understanding_surfaces_provider_errors(self):
        adapter = MagicMock()
        adapter.analyze.side_effect = RuntimeError("provider rejected model")
        scene = SceneUnderstanding(
            model="ollama:missing-model",
            provider_adapter=adapter,
        )

        with self.assertRaisesRegex(RuntimeError, "provider rejected model"):
            scene.get_labels(
                np.zeros((4, 4, 3), dtype=np.uint8),
                "Find vehicles",
            )


if __name__ == "__main__":
    unittest.main()
