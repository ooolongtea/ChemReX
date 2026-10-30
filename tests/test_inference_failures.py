"""M1 transport failures must remain retryable, not completed review records."""

import unittest
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import httpx
from openai import BadRequestError, AuthenticationError, APIConnectionError, RateLimitError

from pipeline.extraction.inference import InferenceClient


def response(content=None, reasoning=None, finish="stop", choices=True):
    return NS(choices=[NS(message=NS(content=content, reasoning_content=reasoning),
                          finish_reason=finish)] if choices else [],
              usage=NS(completion_tokens=512))


class InferenceFailuresTest(unittest.TestCase):
    def setUp(self):
        self.client = InferenceClient(api_key="test", base_url="http://localhost/v1")
        self.client._maas = True
        self.create = Mock()
        self.client.client = NS(chat=NS(completions=NS(create=self.create)))
        for target in ("pipeline.extraction.inference.wait_for_slot",
                       "pipeline.extraction.inference.time.sleep",
                       "pipeline.call_metrics.record_call"):
            patcher = patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_empty_response_recovers_and_keeps_thinking_disabled(self):
        self.create.side_effect = [response(" "), response('{"reaction_product": ["产物"]}')]
        self.assertIn("产物", self.client.chat("实验步骤"))
        self.assertEqual(self.create.call_count, 2)
        self.assertIn("extra_body", self.create.call_args.kwargs)

    def test_reasoning_only_response_is_not_final_extraction(self):
        self.create.return_value = response(reasoning="分析文本：需要提取产物", finish="length")
        with self.assertRaisesRegex(RuntimeError, "reasoning_only=True"):
            self.client.chat("实验步骤")
        self.assertEqual(self.create.call_count, 4)
        self.assertEqual(self.create.call_args.kwargs["max_tokens"], 4096)

    def test_no_choices_is_retried_then_raises(self):
        self.create.return_value = response(choices=False)
        with self.assertRaisesRegex(RuntimeError, "empty final response"):
            self.client.chat("实验步骤")
        self.assertEqual(self.create.call_count, 4)

    def test_connection_error_is_preserved_after_retries(self):
        error = APIConnectionError(request=httpx.Request("POST", "http://localhost/v1"))
        self.create.side_effect = error
        with self.assertRaises(APIConnectionError):
            self.client.chat("实验步骤")
        self.assertEqual(self.create.call_count, 4)

    def test_auth_error_is_not_retried_or_swallowed(self):
        error = AuthenticationError("bad credentials", response=httpx.Response(
            401, request=httpx.Request("POST", "http://localhost/v1")), body={})
        self.create.side_effect = error
        with self.assertRaises(AuthenticationError):
            self.client.chat("实验步骤")
        self.assertEqual(self.create.call_count, 1)

    def test_rate_limit_exhaustion_is_preserved(self):
        error = RateLimitError("too many requests", response=httpx.Response(
            429, request=httpx.Request("POST", "http://localhost/v1")), body={})
        self.create.side_effect = error
        with self.assertRaises(RateLimitError):
            self.client.chat("实验步骤")
        self.assertEqual(self.create.call_count, 4)
        self.assertTrue(all("extra_body" in call.kwargs for call in self.create.call_args_list))

    def test_disk_error_does_not_drop_thinking_or_return_empty(self):
        self.create.side_effect = OSError(28, "No space left on device")
        with self.assertRaises(OSError):
            self.client.chat("实验步骤")
        self.assertEqual(self.create.call_count, 1)

    def test_explicit_unsupported_thinking_option_can_fallback(self):
        error = BadRequestError("unsupported chat_template_kwargs", response=httpx.Response(
            400, request=httpx.Request("POST", "http://localhost/v1")), body={})
        self.create.side_effect = [error, response("[]")]
        self.assertEqual(self.client.chat("实验步骤"), "[]")
        self.assertNotIn("extra_body", self.create.call_args.kwargs)


class ParallelExtractionTest(unittest.TestCase):
    def extractor(self):
        from pipeline.extraction.extractor import ChemReXExtractor
        extractor = ChemReXExtractor.__new__(ChemReXExtractor)
        extractor.configuration = Mock(return_value={'model': 'fake'})
        extractor.extract_ner = Mock(return_value={'ner': {'SOLVENT': ['water']},
            'ner_records': [{'SOLVENT': ['water']}], 'ner_raw': 'ner raw', 'parse_status': 'ok'})
        extractor.extract_action = Mock(return_value={'action': ['stir'], 'action_raw': 'action raw',
            'parse_status': 'ok', 'action_parse_details': {'status': 'ok'}})
        return extractor

    def test_ner_and_action_overlap_and_preserve_output(self):
        extractor = self.extractor()
        barrier = threading.Barrier(2)
        def overlap(value):
            barrier.wait(timeout=3)
            return value
        extractor.extract_ner.side_effect = lambda *_: overlap(extractor.extract_ner.return_value)
        extractor.extract_action.side_effect = lambda *_: overlap(extractor.extract_action.return_value)
        result = extractor.extract('text', 'en')
        self.assertEqual(result['ner_records'], [{'SOLVENT': ['water']}])
        self.assertEqual(result['raw'], {'ner': 'ner raw', 'action': 'action raw'})
        self.assertEqual(result['parse_status'], {'ner': 'ok', 'action': 'ok'})

    def test_partial_extraction_does_not_call_the_other_module(self):
        extractor = self.extractor()
        extractor.extract('text', 'en', task='ner')
        extractor.extract_action.assert_not_called()
        extractor.extract_ner.assert_called_once_with('text', 'en', None)

    def test_parallel_transport_failure_propagates(self):
        extractor = self.extractor()
        extractor.extract_ner.side_effect = RuntimeError('transport failed')
        with self.assertRaisesRegex(RuntimeError, 'transport failed'):
            extractor.extract('text', 'en')


if __name__ == "__main__":
    unittest.main()
