import json
import unittest

from retrieval.mmoa_lite.prompts import (
    parse_g_output,
    parse_q_output,
    parse_s_output,
)


class ParserTests(unittest.TestCase):
    def test_query_output_requires_all_three_queries(self):
        payload = {
            "primary": {"query_text": "positive, satisfied"},
            "confusion": {"query_text": "neutral, factual"},
            "counter": {"query_text": "negative, frustrated"},
        }
        result = parse_q_output(json.dumps(payload))
        self.assertTrue(result.valid)

    def test_selector_rejects_ids_outside_candidate_group(self):
        candidates = {
            "primary": [{"id": "p1"}],
            "confusion": [{"id": "c1"}],
            "counter": [{"id": "n1"}],
        }
        payload = {
            "primary": {"id": "p1"},
            "confusion": {"id": "unknown"},
            "counter": {"id": "n1"},
        }
        result = parse_s_output(json.dumps(payload), candidates)
        self.assertFalse(result.valid)

    def test_generator_normalizes_candidate_label(self):
        payload = {
            "prediction": "happy emotion",
            "confidence": 0.9,
            "reasoning": "Positive verbal and audiovisual cues.",
        }
        result = parse_g_output(json.dumps(payload), ["happy", "sad"])
        self.assertTrue(result.valid)
        self.assertEqual(result.prediction, "happy")


if __name__ == "__main__":
    unittest.main()
