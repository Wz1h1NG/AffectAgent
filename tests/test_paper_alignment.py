import os
import tempfile
import unittest

from affectagent.checkpointing import resolve_checkpoint_path
from affectagent.prompts import (
    build_query_planner_messages,
    parse_evidence_filter_output,
    parse_query_planner_output,
)
from affectagent.reward import AffectiveRewardComputer
from affectagent.schemas import (
    EmotionGeneratorOutput,
    RolloutResult,
    RolloutSample,
)


class PaperAlignmentTests(unittest.TestCase):
    def test_query_planner_observes_all_modalities(self):
        messages = build_query_planner_messages("hello", ["happy"], "en", "face")
        prompt = messages[1]["content"]
        self.assertIn("hello", prompt)
        self.assertIn("<FaceHere>", prompt)
        self.assertIn("<AudioHere>", prompt)

    def test_canonical_query_and_filter_names(self):
        query = parse_query_planner_output(
            '{"support":{"query_text":"a"},"confusion":{"query_text":"b"},'
            '"counter":{"query_text":"c"}}'
        )
        self.assertTrue(query.valid)
        self.assertEqual(query.primary, query.support)
        candidates = {
            "support": [{"id": "s"}],
            "confusion": [{"id": "f"}],
            "counter": [{"id": "c"}],
        }
        selected = parse_evidence_filter_output(
            '{"support":{"id":"s"},"confusion":{"id":"f"},"counter":{"id":"c"}}',
            candidates,
        )
        self.assertTrue(selected.valid)
        self.assertEqual(selected.primary_id, "s")

    def test_equations_1_to_4(self):
        result = RolloutResult(
            sample=RolloutSample("x", "", "happy", ["happy", "sad"]),
            generator_output=EmotionGeneratorOutput(prediction="happy"),
            label_baseline_output=EmotionGeneratorOutput(prediction="sad"),
            rank_baseline_output=EmotionGeneratorOutput(prediction="happy"),
        )
        reward = AffectiveRewardComputer(lambda_planner=0.5, lambda_filter=0.25)
        values = reward.compute_pipeline_rewards(result)
        self.assertEqual(values.score_full, 1.0)
        self.assertEqual(values.score_label, 0.0)
        self.assertEqual(values.score_rank, 1.0)
        self.assertEqual(values.r_shared, 1.0)
        self.assertEqual(values.r_planner, 1.5)
        self.assertEqual(values.r_filter, 1.0)
        self.assertEqual(values.r_generator, 1.0)

    def test_reward_requires_counterfactual_predictions(self):
        result = RolloutResult(
            sample=RolloutSample("x", "", "happy", ["happy"]),
            generator_output=EmotionGeneratorOutput(prediction="happy"),
        )
        with self.assertRaises(ValueError):
            AffectiveRewardComputer().compute_pipeline_rewards(result)

    def test_checkpoint_prefers_paper_name_and_accepts_legacy_name(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = os.path.join(directory, "support_fusion.pth")
            with open(legacy, "wb") as stream:
                stream.write(b"legacy")
            self.assertEqual(resolve_checkpoint_path(directory, "raaf"), legacy)
            canonical = os.path.join(directory, "raaf.pth")
            with open(canonical, "wb") as stream:
                stream.write(b"canonical")
            self.assertEqual(resolve_checkpoint_path(directory, "raaf"), canonical)


if __name__ == "__main__":
    unittest.main()
