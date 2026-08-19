import unittest

try:
    import torch
except (ImportError, OSError):
    torch = None


@unittest.skipIf(torch is None, "PyTorch runtime is unavailable")
class MappoAndFusionTests(unittest.TestCase):
    def test_terminal_reward_gae(self):
        from affectagent.mappo import compute_gae

        rewards = torch.tensor([0.0, 0.0, 1.0])
        values = torch.zeros(3)
        advantages, returns = compute_gae(rewards, values, gamma=0.9, gae_lambda=1.0)
        expected = torch.tensor([0.81, 0.9, 1.0])
        self.assertTrue(torch.allclose(advantages, expected))
        self.assertTrue(torch.allclose(returns, expected))

    def test_raaf_and_mbmoe_public_names(self):
        from affectagent.fusion_modules import MBMoE, RAAF, ModalityMoE, SupportFusion

        self.assertIs(RAAF, SupportFusion)
        self.assertIs(MBMoE, ModalityMoE)
        raaf = RAAF(dim=8, n_heads=2)
        mb_moe = MBMoE(dim=8, n_experts=3, top_k=2)
        current_video = torch.randn(2, 4, 8)
        current_audio = torch.randn(2, 3, 8)
        evidence_video = torch.randn(2, 5, 8)
        evidence_audio = torch.randn(2, 6, 8)
        fused_video, fused_audio, _, _ = raaf(
            current_video,
            current_audio,
            evidence_video,
            evidence_audio,
        )
        balanced_video, balanced_audio = mb_moe(fused_video, fused_audio)
        self.assertEqual(tuple(balanced_video.shape), (2, 4, 8))
        self.assertEqual(tuple(balanced_audio.shape), (2, 3, 8))

    def test_full_rollout_produces_both_reward_counterfactuals(self):
        from affectagent.orchestrator import AffectAgentPipeline
        from affectagent.reward import AffectiveRewardComputer
        from affectagent.schemas import (
            EmotionGeneratorOutput,
            EvidenceFilterOutput,
            QueryPlannerOutput,
            RolloutSample,
        )

        candidates = {
            "support": [{"id": "s", "text": "support", "label_hint": "happy"}],
            "confusion": [{"id": "f", "text": "confusion", "label_hint": "sad"}],
            "counter": [{"id": "c", "text": "counter", "label_hint": "sad"}],
        }

        class DummyRetriever:
            multimodal_ready = False

            def retrieve_channel_A(self, queries, top_k, exclude_sample_id):
                return candidates

        class FakePipeline(AffectAgentPipeline):
            def __init__(self):
                self.retriever = DummyRetriever()
                self.compute_counterfactual_rewards = True
                self.enable_raaf = True
                self.enable_mb_moe = True
                self.retrieval_top_k = 3
                self.generator_calls = 0

            def extract_multimodal_features(self, sample_data, face_or_frame):
                return {"face": object(), "audio": object()}, {}, None, None

            def run_query_planner(self, sample, img_list, face_or_frame):
                output = QueryPlannerOutput(
                    support={"query_text": "happy", "target_label": "happy"},
                    confusion={"query_text": "sad"},
                    counter={"query_text": "sad"},
                )
                return output, torch.tensor([1]), torch.tensor([2])

            def run_retrieval_channel_a(self, *args, **kwargs):
                return candidates

            def run_evidence_filter(self, sample, items, img_list, face_or_frame):
                output = EvidenceFilterOutput(
                    support_id="s",
                    confusion_id="f",
                    counter_id="c",
                )
                return output, torch.tensor([1]), torch.tensor([2])

            def run_emotion_generator(self, *args, **kwargs):
                predictions = ["happy", "sad", "happy"]
                prediction = predictions[self.generator_calls]
                self.generator_calls += 1
                return EmotionGeneratorOutput(prediction=prediction), torch.tensor([1]), torch.tensor([2])

        pipeline = FakePipeline()
        sample = RolloutSample("x", "hello", "happy", ["happy", "sad"], lang="en")
        result = pipeline.full_rollout(sample, {"synthetic": True}, "face")
        self.assertEqual(pipeline.generator_calls, 3)
        self.assertEqual(result.generator_output.prediction, "happy")
        self.assertEqual(result.label_baseline_output.prediction, "sad")
        self.assertEqual(result.rank_baseline_output.prediction, "happy")
        reward = AffectiveRewardComputer().compute_pipeline_rewards(result)
        self.assertEqual((reward.score_full, reward.score_label, reward.score_rank), (1.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
