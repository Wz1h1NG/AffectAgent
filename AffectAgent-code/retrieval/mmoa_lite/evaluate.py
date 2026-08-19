"""
evaluate.py — MMOA-Lite V3 推理评估脚本
加载 PPO 训练后的 AffectGPT + SupportFusion + MoE 权重，
在测试集上执行完整 rollout (Q → R → S → Fusion → G)，计算 WAR / UAR / F1。

用法:
  python retrieval/mmoa_lite/evaluate.py \
    --cfg-path train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml \
    --ckpt-dir retrieval/mmoa_lite/artifacts/rl_checkpoints/best_model \
    --dataset mer2023 \
    --semantic-index-dir retrieval/mmoa_lite/artifacts/semantic_index \
    --multimodal-index-dir retrieval/faiss/artifacts/mercaptionplus \
    --output-dir retrieval/mmoa_lite/artifacts/eval_results
"""

import os
import sys
import json
import argparse
import re
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import decord
decord.bridge.set_bridge('torch')

from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *
from my_affectgpt.common.config import Config
from my_affectgpt.common.registry import registry
from my_affectgpt.conversation.conversation_video import Chat
from my_affectgpt.datasets.builders.image_text_pair_builder import *

from retrieval.mmoa_lite.schemas import RolloutSample, RolloutResult
from retrieval.mmoa_lite.orchestrator import MmoaOrchestrator
from retrieval.mmoa_lite.reward import EmotionRewardComputer
from retrieval.mmoa_lite.retriever_service import DoubleChannelRetriever
from retrieval.mmoa_lite.fusion_modules import SupportFusion, ModalityMoE
from retrieval.mmoa_lite.train_ppo import (
    load_affectgpt_model, resolve_face_or_frame,
    get_dataset_cls, get_candidate_labels, get_lang, set_seed,
    RLTrainDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MMOA-Lite V3 Evaluation")
    parser.add_argument("--cfg-path", type=str, required=True)
    parser.add_argument("--options", nargs="+", default=None)
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="PPO checkpoint 目录 (含 affectgpt_trainable.pth / support_fusion.pth / modality_moe.pth)")
    parser.add_argument("--dataset", type=str, default="mer2023")
    parser.add_argument("--semantic-index-dir", type=str,
                        default="retrieval/mmoa_lite/artifacts/semantic_index")
    parser.add_argument("--multimodal-index-dir", type=str,
                        default="retrieval/faiss/artifacts/mercaptionplus")
    parser.add_argument("--output-dir", type=str,
                        default="retrieval/mmoa_lite/artifacts/eval_results")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-candidate-labels", type=int, default=64)
    parser.add_argument("--fusion-dim", type=int, default=0)
    parser.add_argument("--fusion-heads", type=int, default=8)
    parser.add_argument("--moe-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--max-new-tokens-q", type=int, default=256)
    parser.add_argument("--max-new-tokens-s", type=int, default=128)
    parser.add_argument("--max-new-tokens-g", type=int, default=256)
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--channel-b-top-k", type=int, default=1)
    parser.add_argument(
        "--diagnostic-method",
        choices=["baseline", "direct", "raaf"],
        default="raaf",
        help="Metadata tag only; configure the evaluated model variant separately.",
    )
    parser.add_argument(
        "--diagnostic-partner-condition",
        choices=["", "helpful", "neutral", "conflicting"],
        default="",
        help="Post-hoc partner-quality tag for diagnostic aggregation.",
    )
    parser.add_argument(
        "--diagnostic-input-state",
        choices=["", "balanced", "video_dominant", "audio_dominant", "conflict"],
        default="",
        help="Input-state tag for diagnostic aggregation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def _normalize(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[\[\]\{\}\"'`]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """计算 WAR, UAR, per-class recall, macro-F1。"""
    assert len(predictions) == len(ground_truths)
    n = len(predictions)
    if n == 0:
        return {"war": 0.0, "uar": 0.0, "macro_f1": 0.0, "n": 0}

    preds = [_normalize(p) for p in predictions]
    gts = [_normalize(g) for g in ground_truths]

    correct = sum(1 for p, g in zip(preds, gts) if p == g)
    war = correct / n

    classes = sorted(set(gts))
    class_tp = {c: 0 for c in classes}
    class_fn = {c: 0 for c in classes}
    class_fp = {c: 0 for c in classes}

    for p, g in zip(preds, gts):
        if p == g:
            class_tp[g] += 1
        else:
            class_fn[g] += 1
            if p in class_fp:
                class_fp[p] += 1

    recalls = {}
    precisions = {}
    for c in classes:
        tp = class_tp[c]
        recalls[c] = tp / (tp + class_fn[c]) if (tp + class_fn[c]) > 0 else 0.0
        precisions[c] = tp / (tp + class_fp.get(c, 0)) if (tp + class_fp.get(c, 0)) > 0 else 0.0

    uar = sum(recalls.values()) / len(recalls) if recalls else 0.0

    f1s = []
    for c in classes:
        p, r = precisions[c], recalls[c]
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "war": round(war, 4),
        "uar": round(uar, 4),
        "macro_f1": round(macro_f1, 4),
        "n": n,
        "per_class_recall": {c: round(v, 4) for c, v in recalls.items()},
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    lang = get_lang(args.dataset)

    # 1. 加载模型
    print("=" * 60)
    print("[1/5] Loading AffectGPT model...")
    cfg = Config(argparse.Namespace(cfg_path=args.cfg_path, options=args.options))
    model, chat = load_affectgpt_model(cfg, device)

    trainable_path = os.path.join(args.ckpt_dir, "affectgpt_trainable.pth")
    if os.path.isfile(trainable_path):
        ckpt = torch.load(trainable_path, map_location=device)
        model_state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print(f"  Loaded trainable weights: {len(model_state)} params, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
    else:
        print(f"  [WARN] No trainable weights at {trainable_path}, using base model.")
    model.eval()

    # 2. 加载 Fusion 模块
    print("=" * 60)
    print("[2/5] Loading SupportFusion + ModalityMoE...")
    llama_hidden_size = int(model.llama_model.config.hidden_size)
    fusion_dim = llama_hidden_size if args.fusion_dim <= 0 else args.fusion_dim

    support_fusion = SupportFusion(dim=fusion_dim, n_heads=args.fusion_heads).to(device)
    modality_moe = ModalityMoE(dim=fusion_dim, n_experts=args.moe_experts, top_k=args.moe_top_k).to(device)

    sf_path = os.path.join(args.ckpt_dir, "support_fusion.pth")
    moe_path = os.path.join(args.ckpt_dir, "modality_moe.pth")
    if os.path.isfile(sf_path):
        support_fusion.load_state_dict(torch.load(sf_path, map_location=device))
        print("  Loaded SupportFusion weights.")
    if os.path.isfile(moe_path):
        modality_moe.load_state_dict(torch.load(moe_path, map_location=device))
        print("  Loaded ModalityMoE weights.")
    support_fusion.eval()
    modality_moe.eval()

    # 3. 加载检索器
    print("=" * 60)
    print("[3/5] Loading Retriever...")
    retriever = DoubleChannelRetriever(
        semantic_index_dir=args.semantic_index_dir,
        multimodal_index_dir=args.multimodal_index_dir,
    )
    token_store_path = os.path.join(args.multimodal_index_dir, "token_store", "tokens.h5")
    id_to_row_path = os.path.join(args.multimodal_index_dir, "token_store", "id_to_row.json")

    # 4. 加载测试数据
    print("=" * 60)
    print("[4/5] Loading test data...")
    face_or_frame = resolve_face_or_frame(cfg.datasets_cfg)
    dataset_cls = get_dataset_cls(args.dataset)
    candidate_labels = get_candidate_labels(dataset_cls, max_labels=args.max_candidate_labels)

    test_dataset = RLTrainDataset(
        dataset_cls=dataset_cls,
        candidate_labels=candidate_labels,
        face_or_frame=face_or_frame,
        cfg=cfg,
        lang=lang,
        max_samples=args.max_samples,
        split="test",
    )

    orchestrator = MmoaOrchestrator(
        chat=chat,
        retriever=retriever,
        support_fusion=support_fusion,
        modality_moe=modality_moe,
        token_store_path=token_store_path,
        id_to_row_path=id_to_row_path,
        max_new_tokens_q=args.max_new_tokens_q,
        max_new_tokens_s=args.max_new_tokens_s,
        max_new_tokens_g=args.max_new_tokens_g,
        retrieval_top_k=args.retrieval_top_k,
        channel_b_top_k=args.channel_b_top_k,
    )

    reward_computer = EmotionRewardComputer()

    # 5. 推理
    print("=" * 60)
    print(f"[5/5] Running inference on {len(test_dataset)} samples...")

    predictions = []
    ground_truths = []
    detailed_results = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            rollout_sample, sample_data = test_dataset[idx]
            result = orchestrator.full_rollout(rollout_sample, sample_data, face_or_frame)
            bd = reward_computer.compute_pipeline_rewards(result)
            result.rewards = bd

            pred = result.generator_output.prediction if result.generator_output else ""
            gt = rollout_sample.ground_truth
            diag = result.fusion_diagnostics

            predictions.append(pred)
            ground_truths.append(gt)

            is_hit = int(_normalize(pred) == _normalize(gt))
            detailed_results.append({
                "dataset": args.dataset,
                "seed": args.seed,
                "method": args.diagnostic_method,
                "partner_condition": args.diagnostic_partner_condition,
                "input_state": args.diagnostic_input_state,
                "sample_id": rollout_sample.sample_id,
                "prediction": pred,
                "ground_truth": gt,
                "correct": is_hit,
                "hit": is_hit,
                "confidence": result.generator_output.confidence if result.generator_output else 0.0,
                "q_valid": result.query_output.valid if result.query_output else False,
                "s_valid": result.selector_output.valid if result.selector_output else False,
                "g_valid": result.generator_output.valid if result.generator_output else False,
                "r_total": bd.r_total,
                "r_task": bd.r_task,
                "fusion_partner_id": diag.fusion_partner_id if diag else "",
                "fusion_partner_label": diag.fusion_partner_label if diag else "",
                "fusion_partner_discrete_label": (
                    diag.fusion_partner_discrete_label if diag else ""
                ),
                "video_gate_mean": diag.video_gate_mean if diag else 0.0,
                "audio_gate_mean": diag.audio_gate_mean if diag else 0.0,
                "moe_experts_activated": diag.moe_experts_activated if diag else [],
                "moe_expert_weights": diag.moe_expert_weights if diag else [],
            })

    metrics = compute_metrics(predictions, ground_truths)

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"  WAR:      {metrics['war']:.4f}")
    print(f"  UAR:      {metrics['uar']:.4f}")
    print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"  N:        {metrics['n']}")
    if "per_class_recall" in metrics:
        print("  Per-class Recall:")
        for cls, recall in metrics["per_class_recall"].items():
            print(f"    {cls}: {recall:.4f}")
    print("=" * 60)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    details_path = os.path.join(args.output_dir, "detailed_results.jsonl")
    with open(details_path, "w", encoding="utf-8") as f:
        for item in detailed_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Metrics saved to {metrics_path}")
    print(f"Detailed results saved to {details_path}")


if __name__ == "__main__":
    main()
