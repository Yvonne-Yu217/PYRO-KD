import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import urllib.error

from src.generation import backends


class APISpendRegressionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.ledger = Path(temp.name) / "ledger.json"
        self.enterContext(patch.dict(os.environ, {"OPENAI_API_KEY": "test", "GEMINI_API_KEY": "test"}))
        self.enterContext(patch.object(backends.time, "sleep"))
        # An empty prompt with this price model reserves exactly $0.05 per attempt.
        self.request = backends.GenRequest("moment-1", "", max_new_tokens=8)
        self.prices = {"input_per_million": 1000.0, "output_per_million": 0.0}

    def backend(self, transport, cap=0.06, backend_type=backends.OpenAIBackend, **config):
        return backend_type("test", {"ledger_path": str(self.ledger), "spend_cap_usd": cap,
                                     "prices": self.prices, "max_retries": 1, **config}, transport)

    def state(self):
        return json.loads(self.ledger.read_text(encoding="utf-8"))

    @staticmethod
    def openai_reply(input_tokens=30):
        return {"choices": [{"message": {"content": "A dog runs."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": input_tokens, "completion_tokens": 3}}

    def test_lost_reply_cannot_reuse_the_first_reservation_for_a_retry(self):
        transport = Mock(side_effect=[http.client.IncompleteRead(b"x"), self.openai_reply(50)])

        result, = self.backend(transport).generate([self.request])

        self.assertTrue((result.error or "").startswith("spend_cap:"))
        transport.assert_called_once()
        self.assertAlmostEqual(result.cost_usd, 0.05)
        self.assertAlmostEqual(self.state()["spent_usd"], 0.05)
        self.assertEqual(self.state()["reserved_usd"], 0.0)
        self.assertEqual(self.state()["calls"], 1)

    def test_successful_retry_includes_the_first_attempts_charge(self):
        snapshots = []

        def transport(*_args):
            snapshots.append(self.state())
            if len(snapshots) == 1:
                raise http.client.IncompleteRead(b"x")
            return self.openai_reply()

        result, = self.backend(transport, cap=0.1).generate([self.request])

        self.assertEqual(result.text, "A dog runs.")
        self.assertIsNone(result.error)
        self.assertEqual(result.new_tokens, 3)
        self.assertAlmostEqual(result.cost_usd, 0.08)
        self.assertEqual(len(snapshots), 2)
        self.assertAlmostEqual(snapshots[0]["reserved_usd"], 0.05)
        self.assertAlmostEqual(snapshots[1]["spent_usd"], 0.05)
        self.assertAlmostEqual(snapshots[1]["reserved_usd"], 0.05)
        self.assertAlmostEqual(self.state()["spent_usd"], 0.08)
        self.assertEqual(self.state()["reserved_usd"], 0.0)
        self.assertEqual(self.state()["calls"], 2)

    def test_definitive_rate_limit_releases_reservation_before_retry(self):
        rejection = urllib.error.HTTPError("https://example.test", 429, "Too Many Requests", {}, None)
        transport = Mock(side_effect=[rejection, self.openai_reply()])

        result, = self.backend(transport).generate([self.request])

        self.assertIsNone(result.error)
        self.assertAlmostEqual(result.cost_usd, 0.03)
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(self.state()["reserved_usd"], 0.0)
        self.assertEqual(self.state()["calls"], 2)

    def test_missing_or_invalid_usage_keeps_reserved_cost(self):
        for backend_type, reply, expected_cost in (
            (backends.OpenAIBackend, {**self.openai_reply(), "usage": {"prompt_tokens": -30, "completion_tokens": 3}}, 0.05),
            (backends.OpenAIBackend, {**self.openai_reply(), "usage": {"prompt_tokens": 60}}, 0.06),
            (backends.GeminiBackend, {"candidates": [{"content": {"parts": [{"text": "A dog runs."}]}}]}, 0.05),
        ):
            with self.subTest(backend=backend_type.name):
                self.ledger.unlink(missing_ok=True)
                transport = Mock(return_value=reply)
                result, = self.backend(transport, backend_type=backend_type).generate([self.request])

                self.assertIsNone(result.error)
                self.assertEqual(result.text, "A dog runs.")
                transport.assert_called_once()
                self.assertAlmostEqual(result.cost_usd, expected_cost)
                self.assertAlmostEqual(self.state()["spent_usd"], expected_cost)

    def test_shared_ledger_counts_in_flight_reservation_from_another_backend(self):
        started = threading.Event()
        release = threading.Event()

        def first_transport(*_args):
            started.set()
            if not release.wait(timeout=5):
                raise AssertionError("test never released first attempt")
            return self.openai_reply()

        first_backend = self.backend(first_transport, cap=0.09)
        second_transport = Mock(return_value=self.openai_reply())
        second_backend = self.backend(second_transport, cap=0.09)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(first_backend.generate, [self.request])
            try:
                self.assertTrue(started.wait(timeout=5))
                blocked, = second_backend.generate([self.request])
                self.assertTrue((blocked.error or "").startswith("spend_cap:"))
                second_transport.assert_not_called()
                self.assertAlmostEqual(self.state()["reserved_usd"], 0.05)
            finally:
                release.set()
            self.assertIsNone(first.result(timeout=5)[0].error)

        second, = second_backend.generate([self.request])
        self.assertIsNone(second.error)
        second_transport.assert_called_once()
        self.assertAlmostEqual(self.state()["spent_usd"], 0.06)
        self.assertEqual(self.state()["reserved_usd"], 0.0)

    def test_invalid_prices_and_zero_cap_never_dispatch(self):
        transport = Mock(return_value=self.openai_reply())
        for update in ({"input_per_million": float("nan")}, {"input_per_million": 0.0},
                       {"output_per_million": float("inf")}, {"output_per_million": -1.0},
                       {"image_tokens": 0}, {"image_tokens": 1.5}, {"image_tokens": True}):
            with self.subTest(prices=update), self.assertRaisesRegex(ValueError, "invalid cost model"):
                self.backend(transport, cap=0, prices={**self.prices, **update})

        valid_prices = {"input_per_million": "1000.0", "output_per_million": "0.0", "image_tokens": "1500"}
        result, = self.backend(transport, cap=0, prices=valid_prices).generate([self.request])
        self.assertTrue((result.error or "").startswith("spend_cap:"))
        transport.assert_not_called()

    def test_legacy_nonfinite_ledger_cannot_dispatch(self):
        self.ledger.write_text('{"spent_usd": NaN, "reserved_usd": 0, "calls": 1, "max_ratio": 1}',
                               encoding="utf-8")
        transport = Mock()

        with self.assertRaisesRegex(ValueError, "invalid API spend ledger"):
            self.backend(transport).generate([self.request])

        transport.assert_not_called()
