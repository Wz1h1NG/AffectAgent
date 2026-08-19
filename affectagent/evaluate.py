"""Official AffectAgent evaluation and paper ablation entry point."""

import os
import sys
import json
import argparse
import re
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

from affectagent.checkpointing import resolve_checkpoint_path
from affectagent.orchestrator import AffectAgentPipeline
from affectagent.reward import AffectiveRewardComputer
from affectagent.retriever_service import DualChannelRetriever
from affectagent.fusion_modules import MBMoE, RAAF
from affectagent.train_ppo import (
    load_affectgpt_model, resolve_face_or_frame,
    get_dataset_cls, get_candidate_labels, get_lang, set_seed,
    RLTrainDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Official AffectAgent evaluation")
    parser.add_argument("--cfg-path", type=str, required=True)
    parser.add_argument("--options", nargs="+", default=None)
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Checkpoint directory containing actor, raaf.pth and mb_moe.pth")
    parser.add_argument("--dataset", type=str, default="mer2023")
    parser.add_argument("--semantic-index-dir", type=str,
                        default="affectagent/artifacts/semantic_index")
    parser.add_argument("--multimodal-index-dir", type=str,
                        default="retrieval/faiss/artifacts/mercaptionplus")
    parser.add_argument("--output-dir", type=str,
                        default="affectagent/artifacts/eval_results")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-candidate-labels", type=int, default=64)
    parser.add_argument("--fusion-dim", type=int, default=0)
    parser.add_argument("--fusion-heads", type=int, default=8)
    parser.add_argument("--moe-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--max-new-tokens-q", type=int, default=256)
    parser.add_argument("--max-new-tokens-filter", "--max-new-tokens-s",
                        dest="max_new_tokens_f", type=int, default=128)
    parser.add_argument("--max-new-tokens-g", type=int, default=256)
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--channel-b-top-k", type=int, default=1)
    parser.add_argument(
        "--variant",
        choices=["full", "no_planner", "no_filter", "no_raaf", "no_mb_moe"],
        default="full",
        help="Implemented paper ablation; missing-modality variants are intentionally excluded.",
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

    trainable_path = resolve_checkpoint_path(args.ckpt_dir, "actor")
    if trainable_path:
        ckpt = torch.load(trainable_path, map_location=device)
        model_state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print(f"  Loaded trainable weights: {len(model_state)} params, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
    else:
        print(f"  [WARN] No actor checkpoint under {args.ckpt_dir}; using base model.")
    model.eval()

    # 2. Load RAAF and MB-MoE.
    print("=" * 60)
    print("[2/5] Loading RAAF + MB-MoE...")
    llama_hidden_size = int(model.llama_model.config.hidden_size)
    fusion_dim = llama_hidden_size if args.fusion_dim <= 0 else args.fusion_dim

    raaf = RAAF(dim=fusion_dim, n_heads=args.fusion_heads).to(device)
    mb_moe = MBMoE(dim=fusion_dim, n_experts=args.moe_experts, top_k=args.moe_top_k).to(device)

    raaf_path = resolve_checkpoint_path(args.ckpt_dir, "raaf")
    mb_moe_path = resolve_checkpoint_path(args.ckpt_dir, "mb_moe")
    if raaf_path:
        raaf.load_state_dict(torch.load(raaf_path, map_location=device))
        print(f"  Loaded RAAF weights from {os.path.basename(raaf_path)}.")
    if mb_moe_path:
        mb_moe.load_state_dict(torch.load(mb_moe_path, map_location=device))
        print(f"  Loaded MB-MoE weights from {os.path.basename(mb_moe_path)}.")
    raaf.eval()
    mb_moe.eval()

    # 3. 加载检索器
    print("=" * 60)
    print("[3/5] Loading Retriever...")
    retriever = DualChannelRetriever(
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

    need_counterfactual = args.variant in {"no_planner", "no_filter"}
    pipeline = AffectAgentPipeline(
        chat=chat,
        retriever=retriever,
        raaf=raaf,
        mb_moe=mb_moe,
        token_store_path=token_store_path,
        id_to_row_path=id_to_row_path,
        max_new_tokens_q=args.max_new_tokens_q,
        max_new_tokens_s=args.max_new_tokens_f,
        max_new_tokens_g=args.max_new_tokens_g,
        retrieval_top_k=args.retrieval_top_k,
        channel_b_top_k=args.channel_b_top_k,
        compute_counterfactual_rewards=need_counterfactual,
        enable_raaf=args.variant != "no_raaf",
        enable_mb_moe=args.variant != "no_mb_moe",
    )

    # 5. 推理
    print("=" * 60)
    print(f"[5/5] Running inference on {len(test_dataset)} samples...")

    predictions = []
    ground_truths = []
    detailed_results = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            rollout_sample, sample_data = test_dataset[idx]
            result = pipeline.full_rollout(rollout_sample, sample_data, face_or_frame)
            selected_output = result.generator_output
            if args.variant == "no_planner":
                selected_output = result.label_baseline_output
            elif args.variant == "no_filter":
                selected_output = result.rank_baseline_output
            pred = selected_output.prediction if selected_output else ""
            gt = rollout_sample.ground_truth
            diag = result.fusion_diagnostics

            predictions.append(pred)
            ground_truths.append(gt)

            is_hit = int(_normalize(pred) == _normalize(gt))
            detailed_results.append({
                "dataset": args.dataset,
                "seed": args.seed,
                "variant": args.variant,
                "partner_condition": args.diagnostic_partner_condition,
                "input_state": args.diagnostic_input_state,
                "sample_id": rollout_sample.sample_id,
                "prediction": pred,
                "ground_truth": gt,
                "correct": is_hit,
                "hit": is_hit,
                "confidence": selected_output.confidence if selected_output else 0.0,
                "planner_valid": result.query_output.valid if result.query_output else False,
                "filter_valid": result.filter_output.valid if result.filter_output else False,
                "generator_valid": selected_output.valid if selected_output else False,
                "sample_f1": AffectiveRewardComputer.compute_f1_score(pred, gt),
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
