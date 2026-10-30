"""Provider routing tests; no external API calls."""

import os
import importlib.util
import unittest
from unittest.mock import patch
from pathlib import Path

from pipeline.model_provider import sdk_base_url
from pipeline.gpt_api import GPTAPI


class ModelProviderTests(unittest.TestCase):
    def test_full_endpoint_becomes_sdk_base(self):
        self.assertEqual(
            sdk_base_url("https://api.modelarts-maas.com/v2/chat/completions"),
            "https://api.modelarts-maas.com/v2",
        )

    def test_maas_model_routes_m3_to_maas(self):
        with patch.dict(os.environ, {
            "MAAS_API_KEY": "test-key", "MAAS_MODEL": "deepseek-v4.1-flash",
            "MAAS_BASE_URL": "https://api.modelarts-maas.com/v2/chat/completions",
            "DEEPSEEK_API_KEY": "wrong-provider-key",
        }):
            client = GPTAPI(model="deepseek-v4.1-flash")
        self.assertEqual(client.api_key, "test-key")
        self.assertEqual(str(client.client.base_url), "https://api.modelarts-maas.com/v2/")

    def test_explicit_maas_provider_allows_other_model_names(self):
        with patch.dict(os.environ, {"MAAS_API_KEY": "test-key"}):
            client = GPTAPI(model="another-model", provider="maas")
        self.assertEqual(client.model, "another-model")
        self.assertEqual(client.api_key, "test-key")

    def test_maas_preset_routes_m1_to_maas(self):
        import config
        # Load the client without importing the extraction package's heavy
        # extractor dependencies; this test only covers provider routing.
        path = Path(__file__).resolve().parents[1] / "pipeline/extraction/inference.py"
        spec = importlib.util.spec_from_file_location("chemrex_inference_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        preset = dict(config.MODEL_PRESETS["huawei-maas"], api_key="test-key",
                      model="deepseek-v4.1-flash")
        with patch.dict(config.MODEL_PRESETS, {"huawei-maas": preset}):
            client = module.InferenceClient(preset="huawei-maas")
        self.assertEqual(client.model, "deepseek-v4.1-flash")
        self.assertEqual(str(client.client.base_url), "https://api.modelarts-maas.com/v2/")
        self.assertFalse(client.enable_thinking)


if __name__ == "__main__":
    unittest.main()
