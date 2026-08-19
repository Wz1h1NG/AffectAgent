import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 将项目根目录加入 sys.path，以便能够导入 my_affectgpt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import faiss
except ImportError:
    raise ImportError("faiss is required. Please install faiss-cpu or faiss-gpu.")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence_transformers is required. Please install it.")

from my_affectgpt.datasets.datasets.mercaptionplus_dataset import MERCaptionPlus_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Build Channel A (E5 Semantic) FAISS index for MER-Caption+")
    parser.add_argument(
        "--model-name",
        type=str,
        default="intfloat/multilingual-e5-base",
        help="HuggingFace model name for E5",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="affectagent/artifacts/semantic_index",
        help="Directory to save the E5 index and metadata",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for sentence-transformers encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for encoding (cuda or cpu)",
    )
    return parser.parse_args()


def format_e5_query(text: str) -> str:
    """E5 models require 'query: ' for queries and 'passage: ' for documents."""
    return f"passage: {text}"


def build_document_text(sample: dict) -> str:
    """
    构建语义检索的文档文本。
    去掉结构化前缀（Subtitle:/Description:/Emotion Label:），
    用自然空格拼接，使 E5 编码的语义信号更纯净。

    最终格式示例:
      "我真的受够了每次都是这样 The speaker expresses frustration and anger angry, frustrated"

    三段内容各有检索价值：
      - subtitle（中文对话）: 与 Q 的中文描述式查询直接语义匹配
      - description（英文情感推理）: 与 Q 查询中的情感特征描述跨语言匹配
      - ovlabel（英文情感标签）: 与 Q 查询末尾附加的标签词精确匹配
    """
    parts = []
    for field in ("subtitle", "description", "ovlabel"):
        value = sample.get(field, "")
        if isinstance(value, str):
            value = value.strip()
        else:
            value = ""
        if value:
            parts.append(value)
    return " ".join(parts)


def main():
    import torch
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading MER-Caption+ dataset...")
    dataset = MERCaptionPlus_Dataset()
    
    # 离线环境：尝试将模型名解析为本地路径
    model_name = args.model_name
    if not os.path.isdir(model_name):
        short_name = model_name.split("/")[-1] if "/" in model_name else model_name
        model_root = os.environ.get("AFFECTAGENT_MODEL_ROOT", "models")
        for candidate in [short_name, os.path.join(model_root, short_name), os.path.join("models", short_name)]:
            if os.path.isdir(candidate):
                model_name = os.path.abspath(candidate)
                print(f"  -> Resolved E5 to local path: {model_name}")
                break

    print(f"Loading E5 model: {model_name}")
    model = SentenceTransformer(model_name, device=args.device)

    # 保存构建配置，供 retriever_service.py 运行时读取
    build_config = {"model_name": args.model_name}
    with open(os.path.join(args.output_dir, "build_config.json"), "w") as f:
        json.dump(build_config, f, indent=2)
    
    global_ids = []
    texts = []
    metadata = []
    
    print("Preparing documents...")
    for idx, sample in enumerate(tqdm(dataset.annotation)):
        doc_text = build_document_text(sample)
        # E5 requires "passage: " prefix for indexing corpus
        formatted_text = format_e5_query(doc_text)
        
        texts.append(formatted_text)
        global_ids.append(idx)
        metadata.append({
            "global_id": idx,
            "name": sample["name"],
            "subtitle": sample.get("subtitle", ""),
            "description": sample.get("description", ""),
            "ovlabel": sample.get("ovlabel", ""),
            "raw_text": doc_text
        })
        
    print(f"Encoding {len(texts)} documents...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # E5 uses cosine similarity, so we normalize and use inner product
    )
    
    print("Building FAISS IndexFlatIP...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))
    
    index_path = os.path.join(args.output_dir, "e5_semantic.index")
    faiss.write_index(index, index_path)
    print(f"Index saved to {index_path}")
    
    meta_path = os.path.join(args.output_dir, "metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    
    print(f"Metadata saved to {meta_path}")
    
    id_map_path = os.path.join(args.output_dir, "id_mapping.npy")
    np.save(id_map_path, np.array(global_ids, dtype=np.int64))
    
    print("Done!")


if __name__ == "__main__":
    main()
