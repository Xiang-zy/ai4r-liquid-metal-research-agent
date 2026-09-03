import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agents import LLMClient
from sciverse_client import SciverseClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class SciverseCompatibilityTests(unittest.TestCase):
    def test_successful_responses_are_cached(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            client = SciverseClient("test-key", max_retries=0, cache_dir=cache_dir)
            client.min_interval_seconds = 0
            response = FakeResponse({"hits": [{"doc_id": "d1"}]})
            with patch("urllib.request.urlopen", return_value=response) as mocked:
                self.assertEqual(client.agentic_search("same"), [{"doc_id": "d1"}])
                self.assertEqual(client.agentic_search("same"), [{"doc_id": "d1"}])
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(client.cache_hits, 1)

    def test_current_content_schema_uses_next_offset(self):
        client = SciverseClient("test-key", max_retries=0)
        responses = [
            {"text": "first", "next_offset": 5, "more": True},
            {"text": "second", "next_offset": 11, "more": False},
        ]
        with patch.object(client, "get_content", side_effect=responses) as mocked:
            self.assertEqual(client.get_content_multi("doc", num_chunks=4), ["first", "second"])
        self.assertEqual(mocked.call_args_list[1].kwargs["offset"], 5)

    def test_legacy_content_schema_is_still_accepted(self):
        client = SciverseClient("test-key", max_retries=0)
        with patch.object(client, "get_content", return_value={"content": "legacy"}):
            self.assertEqual(client.get_content_multi("doc"), ["legacy"])

    def test_agentic_search_omits_obsolete_stream_field(self):
        client = SciverseClient("test-key", max_retries=0)
        with patch.object(client, "_post", return_value={"hits": []}) as mocked:
            client.agentic_search("query", top_k=2)
        payload = mocked.call_args.args[1]
        self.assertEqual(payload, {"query": "query", "top_k": 2})


class MiniMaxCompatibilityTests(unittest.TestCase):
    def test_wire_payload_uses_max_completion_tokens(self):
        response = FakeResponse({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"total_tokens": 2},
        })
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}, clear=True):
            client = LLMClient()
            with patch("urllib.request.urlopen", return_value=response) as mocked:
                self.assertEqual(client.chat("hello", max_tokens=7), "ok")
        request = mocked.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["max_completion_tokens"], 7)
        self.assertNotIn("max_tokens", payload)


if __name__ == "__main__":
    unittest.main()
