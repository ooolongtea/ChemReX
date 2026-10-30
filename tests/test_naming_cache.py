import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.standardize.compound_standardizer import CompoundStandardizer


class FakeNamingClient:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = 0
        self.model = "fake-naming"
        self.provider = "fake"

    def chat(self, prompt, user_content=None):
        self.calls += 1
        batch = json.loads(prompt.rsplit("\n", 1)[-1])
        return "\n".join(f"{name}|{self.mapping.get(name, 'UNKNOWN')}" for name in batch)


def install_resolution(std, smiles, status=None):
    def fake_batch(names, std_info):
        result = {}
        for name in names:
            value = smiles.get(name)
            std_info.setdefault(name, {}).update(
                resolution_status=status or ("resolved" if value else "unresolved"),
                resolution_trace=[])
            result[name] = value
        return result
    std._name_to_smiles_batch = fake_batch


class NamingCacheTest(unittest.TestCase):
    def standardizer(self, path, client, use_cache=True):
        return CompoundStandardizer(
            use_opsin=False, naming_client=client, cache_path=path, use_cache=use_cache)

    def test_missing_response_retries_only_missing_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeNamingClient({})
            from unittest.mock import Mock
            client.chat = Mock(side_effect=['Ethanol|ethanol', 'Methanol|methanol'])
            std = self.standardizer(Path(tmp) / 'cache.db', client)
            install_resolution(std, {'Ethanol': 'CCO', 'Methanol': 'CO'})
            result = std.standardize(['Ethanol', 'Methanol'], language='en')
            self.assertEqual(result['Methanol']['smiles'], 'CO')
            second = json.loads(client.chat.call_args_list[1].args[0].rsplit('\n', 1)[-1])
            self.assertEqual(second, ['Methanol'])

    def test_mismatched_response_is_retryable_and_raw_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            from unittest.mock import Mock
            client = FakeNamingClient({})
            client.chat = Mock(return_value='(R)-compound|a different stereoisomer')
            path = Path(tmp) / 'cache.db'
            for _ in range(2):
                std = self.standardizer(path, client)
                std._resolve_smiles = Mock(side_effect=AssertionError('must not resolve unmatched names'))
                result = std.standardize(['(S)-compound'], language='en')['(S)-compound']
                self.assertEqual(result['resolution_status'], 'response_mismatch')
                self.assertIsNone(result['standard_name'])
                self.assertIsNone(result['smiles'])
                self.assertIn('(R)-compound', result['resolution_trace'][0]['responses'][0])
                self.assertNotIn('(S)-compound', std._resolved_cache)
            self.assertEqual(client.chat.call_count, 4)

    def test_old_synthetic_unknown_is_ignored_but_explicit_unknown_is_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.db'
            client = FakeNamingClient({'Ethanol': 'ethanol'})
            std = self.standardizer(path, client)
            std._naming_cache.put_many({
                'Ethanol': {'standard_name': 'UNKNOWN', 'resolution_status': 'unusable_standard_name'},
                'Unobtainium': {'standard_name': 'UNKNOWN', 'resolution_status': 'unusable_standard_name',
                               'naming_raw': 'Unobtainium|UNKNOWN'},
            }, 'en', std._cache_config_version)
            install_resolution(std, {'Ethanol': 'CCO'})
            result = std.standardize(['Ethanol', 'Unobtainium'], language='en')
            self.assertEqual(client.calls, 1)
            self.assertEqual(result['Ethanol']['smiles'], 'CCO')
            self.assertEqual(result['Unobtainium']['resolution_status'], 'unusable_standard_name')
            std.standardize(['Ethanol', 'Unobtainium'], language='en')
            self.assertEqual(client.calls, 1)

    def test_positive_and_negative_hits_survive_new_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "naming.sqlite3"
            client = FakeNamingClient({"Ethanol": "ethanol"})
            first_std = self.standardizer(path, client)
            install_resolution(first_std, {"Ethanol": "CCO"})
            first = first_std.standardize(["Ethanol", "Unobtainium"], language="en")

            self.assertEqual(client.calls, 1)
            self.assertEqual(first["Ethanol"]["smiles"], "CCO")
            self.assertIsNone(first["Unobtainium"]["smiles"])

            client2 = FakeNamingClient({})
            second_std = self.standardizer(path, client2)
            install_resolution(second_std, {"Ethanol": "CCO"})
            second = second_std.standardize(["Ethanol", "Unobtainium"], language="en")

            self.assertEqual(client2.calls, 0)
            self.assertEqual(second["Ethanol"]["smiles"], "CCO")
            self.assertIsNone(second["Unobtainium"]["smiles"])
            self.assertEqual(second["Unobtainium"]["resolution_status"], "unresolved")

    def test_config_version_change_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "naming.sqlite3"
            client = FakeNamingClient({"Ethanol": "ethanol"})
            std = self.standardizer(path, client)
            install_resolution(std, {"Ethanol": "CCO"})
            std.standardize(["Ethanol"], language="en")
            self.assertEqual(client.calls, 1)

            client2 = FakeNamingClient({"Ethanol": "ethanol"})
            std2 = self.standardizer(path, client2)
            std2._cache_config_version = "different-config"
            install_resolution(std2, {"Ethanol": "CCO"})
            std2.standardize(["Ethanol"], language="en")
            self.assertEqual(client2.calls, 1)

    def test_disabled_cache_preserves_positive_only_memory_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "naming.sqlite3"
            client = FakeNamingClient({})
            std = self.standardizer(path, client, use_cache=False)
            install_resolution(std, {})
            std.standardize(["Unobtainium"], language="en")
            std.standardize(["Unobtainium"], language="en")
            self.assertEqual(client.calls, 2)

    def test_transient_negative_is_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "naming.sqlite3"
            client = FakeNamingClient({})
            std = self.standardizer(path, client)
            install_resolution(std, {}, status="timeout")
            std.standardize(["FlakyName"], language="en")
            std.standardize(["FlakyName"], language="en")
            self.assertEqual(client.calls, 2)

    def test_prior_naming_avoids_standardization_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "naming.sqlite3"
            client = FakeNamingClient({})
            std = self.standardizer(path, client, use_cache=False)
            install_resolution(std, {"Ethanol": "CCO"})
            result = std.standardize(
                ["Ethanol"],
                prior_naming={"Ethanol": {
                    "standard_name": "ethanol",
                    "naming_raw": "Ethanol|ethanol",
                    "requires_structure_resolution": True,
                }},
                language="en")
            self.assertEqual(client.calls, 0)
            self.assertEqual(result["Ethanol"]["standard_name"], "ethanol")
            self.assertEqual(result["Ethanol"]["smiles"], "CCO")

    def test_coref_accepts_shared_standardizer(self):
        from pipeline.coref.coref_resolver import CorefResolver
        sentinel = object()
        with patch("pipeline.coref.coref_resolver.GPTAPI"):
            resolver = CorefResolver(standardizer=sentinel)
        self.assertIs(resolver.standardizer, sentinel)

    def test_pipeline_injects_shared_standardizer_into_coref(self):
        from pipeline.pipeline import ChemReXPipeline
        pipe = ChemReXPipeline(enable_image=False, enable_coref=True)
        sentinel = object()
        pipe._standardize = sentinel
        with patch("pipeline.coref.coref_resolver.CorefResolver") as resolver_cls:
            pipe._get_coref()
            resolver_cls.assert_called_once_with(standardizer=sentinel)


if __name__ == "__main__":
    unittest.main()
