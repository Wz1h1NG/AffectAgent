"""
rebuild_text_index.py
=====================
用 SFT 微调后的 LLM hidden state 替换 TF-IDF 文本向量,
重建 text.npy / text.index / text_ids.npy。

运行方式:
    python retrieval/faiss/rebuild_text_index.py \
        --cfg-path train_configs/agent_sft.yaml \
        --sft-ckpt output/agent_sft_ckpt/agent_sft_best.pt \
        --faiss-root retrieval/faiss/artifacts/mercaptionplus \
        --gpu 0 \
        --batch-size 32

它会:
  1. 加载 AffectGPT + SFT LoRA checkpoint
  2. 读取 metadata.jsonl 中每条样本的 subtitle/reason/ovlabel
  3. 将文本送入 LLM，提取 last hidden state → mean pooling → L2 normalize
  4. 覆盖写入 text.npy / text.index（备份旧文件为 .tfidf.bak）
"""

import argparse
import json
import os
import shutil
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import faiss
except ImportError as exc:
    raise ImportError("faiss is required. Install faiss-cpu or faiss-gpu.") from exc

# ── 项目根目录 ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Registry side-effects
from my_affectgpt.tasks import *       # noqa: F401,F403
from my_affectgpt.models import *      # noqa: F401,F403
from my_affectgpt.runners import *     # noqa: F401,F403
from my_affectgpt.processors import *  # noqa: F401,F403
from my_affectgpt.datasets.builders import *  # noqa: F401,F403

from my_affectgpt.common.config import Config
from my_affectgpt.common.registry import registry


def parse_args():
    p = argparse.ArgumentParser(description="Rebuild text.index with SFT'd LLM hidden states")
    p.add_argument("--cfg-path", required=True, help="AffectGPT yaml config")
    p.add_argument("--options", nargs="+", default=None, help="Override config, e.g. model.ckpt_3=xxx")
    p.add_argument("--sft-ckpt", required=True, help="Path to SFT LoRA checkpoint (.pt)")
    p.add_argument("--faiss-root", default="retrieval/faiss/artifacts/mercaptionplus",
                   help="FAISS artifacts root (contains vectors/, indexes/, meta/)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512,
                   help="Max token length for text encoding")
    p.add_argument("--pool-method", choices=["mean", "last"], default="mean",
                   help="Pooling method: mean pooling or last token")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip backing up old TF-IDF files")
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
# Model loading
# ════════════════════════════════════════════════════════════════

def load_model_with_sft(cfg, sft_ckpt_path: str, device: torch.device):
    """加载 AffectGPT 模型并叠加 SFT LoRA 权重."""
    model_cfg = cfg.model_cfg

    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg)

    # 加载 SFT checkpoint (只恢复 LoRA 可训练参数)
    ckpt = torch.load(sft_ckpt_path, map_location="cpu", weights_only=True)
    model_state = ckpt.get("model", {})
    model_dict = dict(model.named_parameters())
    restored = 0
    for name, param in model_state.items():
        if name in model_dict:
            model_dict[name].data.copy_(param)
            restored += 1
    print(f"  Loaded SFT checkpoint: {sft_ckpt_path}")
    print(f"  Restored {restored}/{len(model_state)} LoRA params")

    model = model.to(device)
    model.eval()
    return model


# ════════════════════════════════════════════════════════════════
# Text encoding
# ════════════════════════════════════════════════════════════════

def build_text_raw(subtitle: str, reason: str, ovlabel: str) -> str:
    """与 build_mercaptionplus_faiss.py 保持一致的文本拼接方式."""
    subtitle = subtitle or ""
    reason = reason or ""
    ovlabel = ovlabel or ""
    return f"{subtitle}\n{reason}\n{ovlabel}".strip()


def load_metadata(faiss_root: str) -> List[dict]:
    """从 metadata.jsonl 读取所有样本."""
    meta_path = os.path.join(faiss_root, "meta", "metadata.jsonl")
    records = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@torch.no_grad()
def encode_texts_batch(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 512,
    pool_method: str = "mean",
) -> np.ndarray:
    """
    将一批文本编码为 LLM hidden state 向量。
    
    Returns:
        np.ndarray of shape [batch, hidden_dim], L2 normalized.
    """
    # Tokenize
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)

    # Forward pass through LLM to get hidden states
    # model.llama_model is the PeftModel (LoRA-wrapped)
    with torch.cuda.amp.autocast():
        outputs = model.llama_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    # 取最后一层 hidden state
    last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

    if pool_method == "mean":
        # Mean pooling (只在非 padding token 上)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        sum_hidden = (last_hidden * mask_expanded).sum(dim=1)  # [batch, hidden_dim]
        count = mask_expanded.sum(dim=1).clamp(min=1.0)       # [batch, 1]
        pooled = sum_hidden / count
    elif pool_method == "last":
        # 取每个序列最后一个非 padding token 的 hidden state
        seq_lens = attention_mask.sum(dim=1).long() - 1  # [batch]
        pooled = last_hidden[torch.arange(last_hidden.size(0), device=device), seq_lens]
    else:
        raise ValueError(f"Unknown pool method: {pool_method}")

    # L2 normalize
    pooled = pooled.float()
    norms = pooled.norm(dim=1, keepdim=True).clamp(min=1e-8)
    pooled = pooled / norms

    return pooled.cpu().numpy()


def encode_all_texts(
    model,
    tokenizer,
    texts: List[str],
    global_ids: List[int],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
    pool_method: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    编码所有文本，返回向量矩阵和对应的 global_id 数组。
    """
    all_vectors = []
    all_ids = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding texts with LLM"):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_ids = global_ids[start:end]

        vectors = encode_texts_batch(
            model, tokenizer, batch_texts, device,
            max_length=max_length, pool_method=pool_method,
        )
        all_vectors.append(vectors)
        all_ids.extend(batch_ids)

    text_vectors = np.vstack(all_vectors).astype(np.float32)
    text_ids = np.array(all_ids, dtype=np.int64)
    return text_vectors, text_ids


# ════════════════════════════════════════════════════════════════
# FAISS index
# ════════════════════════════════════════════════════════════════

def build_and_save_faiss(vectors: np.ndarray, out_path: str):
    """构建 FAISS IndexFlatIP 并写入文件."""
    if vectors.shape[0] == 0:
        print(f"  WARNING: No vectors to index, skipping {out_path}")
        return
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    faiss.write_index(index, out_path)
    print(f"  FAISS index saved: {out_path} ({vectors.shape[0]} vectors, dim={d})")


def backup_file(path: str, suffix: str = ".tfidf.bak"):
    """备份旧文件."""
    if os.path.exists(path):
        bak_path = path + suffix
        shutil.copy2(path, bak_path)
        print(f"  Backed up: {path} → {bak_path}")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ── 验证路径 ──
    meta_path = os.path.join(args.faiss_root, "meta", "metadata.jsonl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.jsonl not found: {meta_path}")
    if not os.path.exists(args.sft_ckpt):
        raise FileNotFoundError(f"SFT checkpoint not found: {args.sft_ckpt}")

    vector_dir = os.path.join(args.faiss_root, "vector")
    index_dir = os.path.join(args.faiss_root, "index")
    os.makedirs(vector_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    # ── Step 1: 加载模型 ──
    print("=" * 60)
    print("[1/4] Loading AffectGPT + SFT checkpoint...")
    cfg = Config(args)
    model = load_model_with_sft(cfg, args.sft_ckpt, device)
    tokenizer = model.llama_tokenizer

    # ── Step 2: 读取 metadata ──
    print("\n" + "=" * 60)
    print("[2/4] Loading metadata...")
    records = load_metadata(args.faiss_root)
    print(f"  Total samples: {len(records)}")

    # 构建文本内容和 ID
    texts = []
    global_ids = []
    for r in records:
        text_raw = build_text_raw(
            r.get("subtitle", ""),
            r.get("reason", ""),
            r.get("ovlabel", ""),
        )
        if text_raw:  # 只编码有文本的样本
            texts.append(text_raw)
            global_ids.append(r["global_id"])

    print(f"  Samples with text: {len(texts)}")

    # ── Step 3: 编码文本 ──
    print("\n" + "=" * 60)
    print(f"[3/4] Encoding texts with LLM (pool={args.pool_method}, max_len={args.max_length})...")
    t0 = time.time()

    text_vectors, text_ids = encode_all_texts(
        model, tokenizer, texts, global_ids, device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pool_method=args.pool_method,
    )

    elapsed = time.time() - t0
    print(f"  Encoded {text_vectors.shape[0]} texts in {elapsed:.1f}s")
    print(f"  Vector dim: {text_vectors.shape[1]} (was TF-IDF 4096)")
    print(f"  Vector norm check (should be ~1.0): {np.linalg.norm(text_vectors[:5], axis=1)}")

    # ── Step 4: 保存 ──
    print("\n" + "=" * 60)
    print("[4/4] Saving new text vectors and FAISS index...")

    text_npy_path = os.path.join(vector_dir, "text.npy")
    text_ids_path = os.path.join(vector_dir, "text_ids.npy")
    text_index_path = os.path.join(index_dir, "text.index")

    # 备份旧的 TF-IDF 文件
    if not args.no_backup:
        for p in [text_npy_path, text_ids_path, text_index_path]:
            backup_file(p)

    # 写入新文件
    np.save(text_npy_path, text_vectors)
    np.save(text_ids_path, text_ids)
    print(f"  Saved: {text_npy_path} shape={text_vectors.shape}")
    print(f"  Saved: {text_ids_path} shape={text_ids.shape}")

    build_and_save_faiss(text_vectors, text_index_path)

    # 更新 build_summary.json
    summary_path = os.path.join(args.faiss_root, "build_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["dims"]["text_dim"] = int(text_vectors.shape[1])
        summary["text_index_method"] = "llm_hidden_state"
        summary["text_pool_method"] = args.pool_method
        summary["sft_ckpt"] = args.sft_ckpt
        summary["text_rebuild_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  Updated: {summary_path}")

    print("\n" + "=" * 60)
    print("Done! Text index rebuilt with LLM hidden states.")
    print(f"  Old: TF-IDF dim=4096")
    print(f"  New: LLM hidden state dim={text_vectors.shape[1]}")
    print(f"  Vectors: {text_vectors.shape[0]} samples")


if __name__ == "__main__":
    main()
