"""Official AffectAgent MAPPO training with counterfactual rewards and token GAE."""

import os
import sys
import json
import copy
import argparse
import random
import time
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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

from affectagent.schemas import RewardBreakdown, RolloutResult, RolloutSample
from affectagent.orchestrator import AffectAgentPipeline
from affectagent.reward import AffectiveRewardComputer
from affectagent.retriever_service import DualChannelRetriever
from affectagent.fusion_modules import MBMoE, RAAF
from affectagent.mappo import PolicyGradientUpdater, SequenceValueHead


# ═══════════════════════════════════════════════════════════════════════════════
# AffectGPT 原生模型加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_affectgpt_model(cfg, device):
    """
    使用 AffectGPT 原生 from_config 加载模型（原始预训练权重，非SFT微调权重）。
    返回 (model, chat) 元组。
    """
    model_cfg = cfg.model_cfg

    # 通过 registry 加载 AffectGPT 模型
    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg)
    model = model.to(device)

    # 创建 Chat 对象
    chat = Chat(model, model_cfg, device=device)

    return model, chat


def resolve_face_or_frame(datasets_cfg):
    """从 datasets_cfg 中解析 face_or_frame 配置。"""
    face_or_frame_candidates = []
    if 'mercaptionplus' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['mercaptionplus'].face_or_frame)
    if 'ovmerd' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['ovmerd'].face_or_frame)
    if face_or_frame_candidates:
        return face_or_frame_candidates[0]
    return 'multiface_audio_face_text'


# ═══════════════════════════════════════════════════════════════════════════════
# 多模态训练数据集
# ═══════════════════════════════════════════════════════════════════════════════

def get_dataset_cls(dataset_name):
    """获取 AffectGPT 原生数据集类实例。"""
    from my_affectgpt.datasets.datasets.mercaptionplus_dataset import MERCaptionPlus_Dataset
    from my_affectgpt.datasets.datasets.mer2023 import MER2023_Dataset
    from my_affectgpt.datasets.datasets.mer2024 import MER2024_Dataset
    from my_affectgpt.datasets.datasets.meld import MELD_Dataset
    from my_affectgpt.datasets.datasets.iemocap import IEMOCAPFour_Dataset

    mapping = {
        'mercaptionplus': MERCaptionPlus_Dataset,
        'mer2023': MER2023_Dataset,
        'mer2024': MER2024_Dataset,
        'meld': MELD_Dataset,
        'iemocap': IEMOCAPFour_Dataset,
        'iemocapfour': IEMOCAPFour_Dataset,
    }
    cls = mapping.get(dataset_name.lower())
    if cls is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(mapping.keys())}")
    return cls()


class RLTrainDataset(Dataset):
    """
    使用 AffectGPT 原生数据集类加载训练样本，提供多模态数据读取能力。
    每个样本包含 RolloutSample 元信息和 sample_data（多模态原始数据）。
    """

    def __init__(
        self,
        dataset_cls,
        candidate_labels: List[str],
        face_or_frame: str,
        cfg,
        lang: str = "zh",
        max_samples: int = -1,
        split: str = "train",
    ):
        self.dataset_cls = dataset_cls
        self.face_or_frame = face_or_frame
        self.candidate_labels = candidate_labels
        self.lang = lang

        # 配置数据集的多模态读取
        dataset_cls.needed_data = dataset_cls.get_needed_data(face_or_frame)

        # 配置处理器
        inference_cfg = cfg.inference_cfg if hasattr(cfg, 'inference_cfg') else None
        dataset_cls.vis_processor = BaseProcessor()
        dataset_cls.img_processor = BaseProcessor()
        if inference_cfg is not None:
            vis_processor_cfg = inference_cfg.get("vis_processor")
            img_processor_cfg = inference_cfg.get("img_processor")
            if vis_processor_cfg is not None:
                dataset_cls.vis_processor = registry.get_processor_class(
                    vis_processor_cfg.train.name
                ).from_config(vis_processor_cfg.train)
            if img_processor_cfg is not None:
                dataset_cls.img_processor = registry.get_processor_class(
                    img_processor_cfg.train.name
                ).from_config(img_processor_cfg.train)
        dataset_cls.n_frms = cfg.model_cfg.vis_processor.train.n_frms

        name2subtitle = dataset_cls.name2subtitle

        self.samples = []
        if split == "test" and hasattr(dataset_cls, "read_test_names"):
            test_names = dataset_cls.read_test_names()
            name2gt = dataset_cls.get_test_name2gt()
            idx2emo = {}
            if hasattr(dataset_cls, "get_emo2idx_idx2emo"):
                _, idx2emo = dataset_cls.get_emo2idx_idx2emo()
            for name in test_names:
                gt = name2gt.get(name, "")
                if isinstance(gt, (int, np.integer)) and int(gt) in idx2emo:
                    gt = idx2emo[int(gt)]
                if not name or gt == "":
                    continue
                subtitle = name2subtitle.get(name, "")
                sub_str = str(subtitle) if subtitle and not (isinstance(subtitle, float) and subtitle != subtitle) else ""
                self.samples.append({
                    "name": name,
                    "subtitle": sub_str,
                    "ground_truth": str(gt),
                    "description": "",
                })
        else:
            # MERCaptionPlus uses ovlabel; categorical datasets use onehot.
            for ann in dataset_cls.annotation:
                name = ann.get("name", "")
                subtitle = ann.get("subtitle", name2subtitle.get(name, ""))
                gt = ann.get("ovlabel", ann.get("onehot", ""))
                description = ann.get("description", "")
                if not name or gt == "":
                    continue
                sub_str = str(subtitle) if subtitle and not (isinstance(subtitle, float) and subtitle != subtitle) else ""
                self.samples.append({
                    "name": name,
                    "subtitle": sub_str,
                    "ground_truth": str(gt),
                    "description": str(description) if description else "",
                })

        if 0 < max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

        print(
            f"[RLTrainDataset] Loaded {len(self.samples)} {split} samples, "
            f"face_or_frame={face_or_frame}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        name = info["name"]

        # 构建 RolloutSample
        rollout_sample = RolloutSample(
            sample_id=name,
            subtitle=info["subtitle"],
            ground_truth=info["ground_truth"],
            candidate_labels=self.candidate_labels,
            dataset=self.dataset_cls.__class__.__name__,
            lang=self.lang,
        )

        # 读取多模态原始数据（AffectGPT 原生方式）
        sample = {"name": name}
        video_path = self.dataset_cls._get_video_path(sample) if hasattr(self.dataset_cls, '_get_video_path') else None
        audio_path = self.dataset_cls._get_audio_path(sample) if hasattr(self.dataset_cls, '_get_audio_path') else None
        face_npy = self.dataset_cls._get_face_path(sample) if hasattr(self.dataset_cls, '_get_face_path') else None

        try:
            sample_data = self.dataset_cls.read_frame_face_audio_text(
                video_path=video_path,
                face_npy=face_npy,
                audio_path=audio_path,
                image_path=None,
            )
        except Exception as e:
            print(f"[WARN] Failed to read multimodal data for {name}: {e}")
            sample_data = None

        return rollout_sample, sample_data

    @staticmethod
    def collate_fn(batch):
        """自定义 collate：分离 rollout_samples 和 sample_data_list。"""
        rollout_samples = []
        sample_data_list = []
        for rollout_sample, sample_data in batch:
            if sample_data is not None:
                rollout_samples.append(rollout_sample)
                sample_data_list.append(sample_data)
        return rollout_samples, sample_data_list


# ═══════════════════════════════════════════════════════════════════════════════
# 参数解析
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Official AffectAgent MAPPO training")

    # AffectGPT 配置（加载原始权重）
    parser.add_argument("--cfg-path", type=str, required=True,
                        help="AffectGPT YAML 配置文件路径")
    parser.add_argument("--options", nargs="+", default=None,
                        help="覆盖配置项, e.g. model.ckpt_3=xxx")

    # 数据
    parser.add_argument("--dataset", type=str, default="mercaptionplus",
                        help="数据集名称: mercaptionplus")
    parser.add_argument("--semantic-index-dir", type=str,
                        default="affectagent/artifacts/semantic_index",
                        help="Channel A E5 语义索引目录")
    parser.add_argument("--multimodal-index-dir", type=str,
                        default="retrieval/faiss/artifacts/mercaptionplus",
                        help="Channel B 多模态 FAISS 索引目录")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="最大训练样本数 (-1 为全部)")
    parser.add_argument("--max-candidate-labels", type=int, default=64,
                        help="候选标签池上限，按 openset 标签频次截断；<=0 表示不截断")

    # 训练超参
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.1,
                        help="KL 散度正则化系数 (β)")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="每次 rollout 后进行多少轮 PPO 更新")
    parser.add_argument("--ppo-clip-range", type=float, default=0.2,
                        help="PPO ratio clipping 范围")
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="value loss 系数")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="GAE discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="Generalized Advantage Estimation coefficient")
    parser.add_argument("--lambda-planner", type=float, default=1.0,
                        help="Equation (3) local incremental reward coefficient")
    parser.add_argument("--lambda-filter", type=float, default=1.0,
                        help="Equation (4) local incremental reward coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)

    # Fusion 模块超参
    parser.add_argument("--fusion-dim", type=int, default=0,
                        help="RAAF / MB-MoE feature dimension; <=0 uses LLM hidden size")
    parser.add_argument("--fusion-heads", type=int, default=8,
                        help="RAAF cross-attention heads")
    parser.add_argument("--moe-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=int, default=2)

    # 生成
    parser.add_argument("--max-new-tokens-q", type=int, default=256)
    parser.add_argument("--max-new-tokens-filter", "--max-new-tokens-s",
                        dest="max_new_tokens_f", type=int, default=128)
    parser.add_argument("--max-new-tokens-g", type=int, default=256)
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--channel-b-top-k", type=int, default=1)

    # 输出
    parser.add_argument("--output-dir", type=str,
                        default="affectagent/artifacts/mappo_checkpoints")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_open_labels(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[,，;/；、|]+", str(text))
    labels = [part.strip() for part in parts if part and part.strip()]
    return labels


def get_candidate_labels(dataset_cls, max_labels: int = 64) -> List[str]:
    """从 MERCaptionPlus 的 annotation 中收集 openset 标签原子，并按频次排序。"""
    explicit_labels = getattr(dataset_cls, "candidate_labels", None)
    if isinstance(explicit_labels, str) and explicit_labels.strip():
        ranked_labels = split_open_labels(explicit_labels)
    elif isinstance(explicit_labels, (list, tuple)):
        ranked_labels = [str(label).strip() for label in explicit_labels if str(label).strip()]
    else:
        label_counter = Counter()
        for ann in dataset_cls.annotation:
            raw_labels = ann.get("ovlabel", ann.get("discrete", ann.get("label", "")))
            for label in split_open_labels(raw_labels):
                label_counter[label] += 1
        ranked_labels = [
            label for label, _ in sorted(label_counter.items(), key=lambda item: (-item[1], item[0]))
        ]
    if not ranked_labels:
        ranked_labels = ["neutral"]
    if max_labels > 0:
        ranked_labels = ranked_labels[:max_labels]
    return ranked_labels


def get_lang(dataset_name: str) -> str:
    return "zh" if "mer" in dataset_name.lower() else "en"


def get_trainable_state_dict(model) -> Dict[str, torch.Tensor]:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    state_dict = model.state_dict()
    return {
        name: tensor.detach().cpu()
        for name, tensor in state_dict.items()
        if name in trainable_names
    }


def save_affectagent_checkpoint(
    directory: str,
    model,
    raaf,
    mb_moe,
    value_head,
) -> Dict[str, torch.Tensor]:
    """Save canonical paper-named files plus a machine-readable manifest."""

    os.makedirs(directory, exist_ok=True)
    trainable_state = get_trainable_state_dict(model)
    torch.save({"model": trainable_state}, os.path.join(directory, "affectgpt_trainable.pth"))
    torch.save(raaf.state_dict(), os.path.join(directory, "raaf.pth"))
    torch.save(mb_moe.state_dict(), os.path.join(directory, "mb_moe.pth"))
    torch.save(value_head.state_dict(), os.path.join(directory, "critic.pth"))
    manifest = {
        "format_version": 2,
        "implementation": "AffectAgent",
        "files": {
            "actor": "affectgpt_trainable.pth",
            "raaf": "raaf.pth",
            "mb_moe": "mb_moe.pth",
            "critic": "critic.pth",
        },
        "legacy_fallbacks": {
            "raaf": "support_fusion.pth",
            "mb_moe": "modality_moe.pth",
            "critic": "value_head.pth",
        },
    }
    with open(os.path.join(directory, "checkpoint_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    return trainable_state


# ═══════════════════════════════════════════════════════════════════════════════
# 主训练流程
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    lang = get_lang(args.dataset)

    # ──────────────── 1. 加载 AffectGPT 原始模型 ────────────────
    print("=" * 60)
    print("[1/6] Loading AffectGPT actor from the role-specific SFT checkpoint...")

    # 构建 Config 对象
    cfg = Config(argparse.Namespace(cfg_path=args.cfg_path, options=args.options))
    model_cfg = cfg.model_cfg
    # 注意：如果训练流程是 SFT → PPO，需要保留 ckpt_3（SFT 微调权重）。
    # 仅清空 ckpt / ckpt_2（原始预训练的额外 checkpoint），保留 ckpt_3 以加载 SFT 后的 LoRA。
    # 如果要从头开始 PPO（不经过 SFT），可通过命令行 --options model.ckpt_3="" 手动清空。
    for ckpt_key in ["ckpt", "ckpt_2"]:
        if hasattr(model_cfg, ckpt_key):
            setattr(model_cfg, ckpt_key, "")

    # 加载原始 AffectGPT 模型（通过 registry，含 LoRA）
    model, chat = load_affectgpt_model(cfg, device)
    model.eval()

    llama_hidden_size = int(model.llama_model.config.hidden_size)
    fusion_dim = llama_hidden_size if args.fusion_dim <= 0 else args.fusion_dim
    if fusion_dim != llama_hidden_size:
        raise ValueError(
            f"fusion_dim ({fusion_dim}) must match llama hidden_size ({llama_hidden_size})."
        )

    # Paper section 3.5 uses a frozen copy of the complete SFT policy, including adapters.
    reference_model = copy.deepcopy(model.llama_model).to(device).eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    print("  Frozen SFT reference policy created.")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  AffectGPT loaded. Trainable params (LoRA): {trainable_params:,}")

    # ──────────────── 2. Initialize RAAF + MB-MoE ────────────────
    print("=" * 60)
    print("[2/6] Initializing RAAF + MB-MoE...")

    raaf = RAAF(dim=fusion_dim, n_heads=args.fusion_heads).to(device)
    mb_moe = MBMoE(dim=fusion_dim, n_experts=args.moe_experts, top_k=args.moe_top_k).to(device)

    fusion_params = sum(p.numel() for p in raaf.parameters()) + \
                    sum(p.numel() for p in mb_moe.parameters())
    print(f"  Fusion module params: {fusion_params:,}")

    # ──────────────── 3. 加载检索器 ────────────────
    print("=" * 60)
    print("[3/6] Loading Retriever (frozen, dual-channel)...")

    retriever = DualChannelRetriever(
        semantic_index_dir=args.semantic_index_dir,
        multimodal_index_dir=args.multimodal_index_dir,
    )

    # Channel B token store 路径
    token_store_path = os.path.join(args.multimodal_index_dir, "token_store", "tokens.h5")
    id_to_row_path = os.path.join(args.multimodal_index_dir, "token_store", "id_to_row.json")

    # ──────────────── 4. 加载训练数据 ────────────────
    print("=" * 60)
    print("[4/6] Loading training data (multimodal)...")

    face_or_frame = resolve_face_or_frame(cfg.datasets_cfg)
    dataset_cls = get_dataset_cls(args.dataset)
    lang = getattr(dataset_cls, "subtitle_lang", lang)
    candidate_labels = get_candidate_labels(dataset_cls, max_labels=args.max_candidate_labels)
    print(f"  Candidate labels ({len(candidate_labels)}): {candidate_labels[:10]}...")

    train_dataset = RLTrainDataset(
        dataset_cls=dataset_cls,
        candidate_labels=candidate_labels,
        face_or_frame=face_or_frame,
        cfg=cfg,
        lang=lang,
        max_samples=args.max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=RLTrainDataset.collate_fn,
        drop_last=True,
        num_workers=0,
    )

    # ──────────────── 5. 初始化组件 ────────────────
    print("=" * 60)
    print("[5/6] Initializing Orchestrator, Reward, Optimizer...")

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
    )

    reward_computer = AffectiveRewardComputer(
        lambda_planner=args.lambda_planner,
        lambda_filter=args.lambda_filter,
    )
    value_head = SequenceValueHead(llama_hidden_size).to(device)

    # 优化器：同时优化 AffectGPT LoRA 参数 + Fusion 模块参数 + Value Head
    trainable_param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.learning_rate},
        {"params": list(raaf.parameters()) + list(mb_moe.parameters()),
         "lr": args.learning_rate * 10},  # Fusion 模块用更高学习率
        {"params": list(value_head.parameters()), "lr": args.learning_rate},
    ]

    optimizer = torch.optim.AdamW(trainable_param_groups, weight_decay=0.01)

    total_steps = args.num_epochs * len(train_loader) * max(args.ppo_epochs, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.learning_rate, args.learning_rate * 10, args.learning_rate],
        total_steps=max(total_steps, 1),
        pct_start=args.warmup_ratio,
        anneal_strategy="cos",
    )

    updater = PolicyGradientUpdater(
        model=model,
        reference_model=reference_model,
        value_head=value_head,
        optimizer=optimizer,
        kl_coef=args.kl_coef,
        ppo_epochs=args.ppo_epochs,
        clip_range=args.ppo_clip_range,
        value_coef=args.value_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        device=device,
    )

    # ──────────────── 6. 训练循环 ────────────────
    print("=" * 60)
    print("[6/6] Starting AffectAgent MAPPO training...")
    print(f"  Epochs:       {args.num_epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Total steps:  {total_steps}")
    print(f"  KL coef (β):  {args.kl_coef}")
    print(f"  PPO epochs:   {args.ppo_epochs}")
    print(f"  PPO clip:     {args.ppo_clip_range}")
    print(f"  GAE gamma:    {args.gamma}")
    print(f"  GAE lambda:   {args.gae_lambda}")
    print(f"  Fusion dim:   {fusion_dim}")
    print(f"  face_or_frame:{face_or_frame}")
    print("=" * 60)

    log_path = os.path.join(args.output_dir, "training_log.jsonl")
    reward_log_path = os.path.join(args.output_dir, "reward_log.jsonl")

    global_step = 0
    best_reward = -float("inf")

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_reward = 0.0
        epoch_accuracy = 0.0
        epoch_count = 0
        reward_computer.reset_uar_stats()  # 每个 epoch 重置 UAR 统计

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not sys.stdout.isatty(),
        )

        for batch_samples, batch_sample_data in pbar:
            if len(batch_samples) == 0:
                continue

            global_step += 1
            step_start = time.time()

            # ──── Rollout ────
            model.eval()
            raaf.eval()
            mb_moe.eval()

            rollout_results = pipeline.batch_rollout(
                batch_samples, batch_sample_data, face_or_frame
            )

            # Equations (1)-(4): full, label-baseline, and rank-baseline scores.
            reward_breakdowns = reward_computer.batch_compute_rewards(rollout_results)

            # ──── 诊断日志（前 5 step 打印 Planner/Filter/Generator 输出）────
            if global_step <= 5:
                for i, (rr, bd) in enumerate(zip(rollout_results, reward_breakdowns)):
                    q_valid = rr.query_output.valid if rr.query_output else False
                    f_valid = rr.filter_output.valid if rr.filter_output else False
                    n_cand = sum(len(v) for v in (rr.candidates or {}).values())
                    n_evi = sum(1 for v in (rr.selected_evidence or {}).values() if v)
                    g_out = rr.generator_output
                    g_pred = g_out.prediction if g_out else "None"
                    g_valid = g_out.valid if g_out else False
                    gt = rr.sample.ground_truth if rr.sample else "?"
                    print(f"  [DIAG step={global_step} sample={i}] "
                          f"Planner={q_valid} Filter={f_valid} cand={n_cand} evi={n_evi} | "
                          f"gt={gt!r} | pred={g_pred!r} G={g_valid} | "
                          f"full={bd.score_full:.2f} label={bd.score_label:.2f} "
                          f"rank={bd.score_rank:.2f} shared={bd.r_shared:.3f}")
                    if global_step <= 2:
                        g_raw = (g_out.raw_text[:200] if g_out else "None") if g_out else "None"
                        print(f"    G_raw: {g_raw}")

            # ──── 收集 PPO 序列 ────
            trajectories = pipeline.collect_ppo_trajectories(rollout_results)

            # ──── 策略梯度更新 ────
            # update_step 内部会先 eval 算 old_log_probs，再 train 做 PPO
            # fusion 模块也需在 train 模式（orchestrator replay G 时会重跑 fusion）
            raaf.train()
            mb_moe.train()
            stats = updater.update_step(trajectories, pipeline)

            for _ in range(stats.get("update_rounds", 0)):
                scheduler.step()

            # ──── 统计 ────
            step_time = time.time() - step_start
            batch_accuracy = sum(bd.score_full for bd in reward_breakdowns) / max(len(reward_breakdowns), 1)
            batch_r_total = sum(bd.r_shared for bd in reward_breakdowns) / max(len(reward_breakdowns), 1)
            epoch_loss += stats["loss"]
            epoch_reward += stats["mean_reward"]
            epoch_accuracy += batch_accuracy
            epoch_count += 1

            # ──── 日志 ────
            if global_step % args.logging_steps == 0:
                log_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": round(stats["loss"], 4),
                    "policy_loss": round(stats["policy_loss"], 4),
                    "value_loss": round(stats["value_loss"], 4),
                    "mean_reward": round(stats["mean_reward"], 4),
                    "mean_kl": round(stats["mean_kl"], 6),
                    "batch_accuracy": round(batch_accuracy, 4),
                    "batch_r_total": round(batch_r_total, 4),
                    "avg_loss": round(epoch_loss / epoch_count, 4),
                    "avg_accuracy": round(epoch_accuracy / epoch_count, 4),
                    "lr": round(scheduler.get_last_lr()[0], 8),
                    "step_time": round(step_time, 2),
                }

                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                pbar.set_postfix(
                    loss=f"{stats['loss']:.4f}",
                    R=f"{batch_r_total:.3f}",
                    acc=f"{batch_accuracy:.3f}",
                    kl=f"{stats['mean_kl']:.4f}",
                )

                for bd in reward_breakdowns:
                    with open(reward_log_path, "a") as f:
                        f.write(json.dumps(asdict(bd), ensure_ascii=False) + "\n")

            # ──── 保存 checkpoint ────
            if global_step % args.save_steps == 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                trainable_state = save_affectagent_checkpoint(
                    ckpt_dir,
                    model,
                    raaf,
                    mb_moe,
                    value_head,
                )

                print(f"\n  [Checkpoint] Saved to {ckpt_dir}")

                current_avg_reward = epoch_reward / epoch_count
                if current_avg_reward > best_reward:
                    best_reward = current_avg_reward
                    best_dir = os.path.join(args.output_dir, "best_model")
                    save_affectagent_checkpoint(best_dir, model, raaf, mb_moe, value_head)
                    print(f"  [Best] New best reward: {best_reward:.4f}")

        print(f"\n  Epoch {epoch + 1} finished.")
        print(f"    Avg Loss:     {epoch_loss / max(epoch_count, 1):.4f}")
        print(f"    Avg Reward:   {epoch_reward / max(epoch_count, 1):.4f}")
        print(f"    Avg Accuracy: {epoch_accuracy / max(epoch_count, 1):.4f}")

    # ──── 训练结束 ────
    final_dir = os.path.join(args.output_dir, "final_model")
    save_affectagent_checkpoint(final_dir, model, raaf, mb_moe, value_head)

    print(f"\n{'=' * 60}")
    print(f"Training finished! Final model saved to {final_dir}")
    print(f"Best reward: {best_reward:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
