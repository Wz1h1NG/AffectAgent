import os
import sys
import json
import argparse
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
from my_affectgpt.datasets.datasets.mer2023 import MER2023_Dataset
from my_affectgpt.datasets.datasets.mer2024 import MER2024_Dataset
from my_affectgpt.datasets.datasets.meld import MELD_Dataset
from my_affectgpt.datasets.datasets.iemocap import IEMOCAPFour_Dataset
from affectagent.retriever_service import DualChannelRetriever

S_SYSTEM_PROMPT_ZH = """你是AffectAgent的多模态证据过滤器（Evidence Filter）。
你的任务是从检索召回的多个候选证据中，为当前样本挑选出最合适的三条证据（Primary, Confusion, Counter 各一条）。

你将收到以下信息：
1. 当前样本的真实标签和多模态特征描述 (仅供你做决策参考)。
2. 当前样本的字幕。
3. 检索到的候选证据列表 (分为 Primary组, Confusion组, Counter组)。每条证据都有全局ID (id) 和文本描述。

请综合当前样本的特征，从【每组】中分别选出【一条】最符合当前样本多模态表现、最有帮助的证据。
严格输出显式指定每组所选证据ID的纯JSON对象，不要输出任何解释或Markdown格式。

输出格式要求：
{
  "support": {"id": "id1"},
  "confusion": {"id": "id2"},
  "counter": {"id": "id3"}
}"""

S_SYSTEM_PROMPT_EN = """You are AffectAgent's multimodal Evidence Filter.
Your task is to select the three most suitable evidence items (one each for Primary, Confusion, and Counter) for the current sample from the retrieved candidates.

You will receive the following information:
1. The ground truth label and multimodal feature description of the current sample (for your reference only).
2. The subtitle of the current sample.
3. The retrieved candidate evidence list (divided into Primary, Confusion, and Counter groups). Each evidence has a global ID (id) and a text description.

Please comprehensively consider the features of the current sample, and select exactly ONE most helpful and matching evidence item from EACH group.
Strictly output a pure JSON object explicitly specifying the selected evidence IDs for each group, without any explanations or Markdown formatting.

Output format requirement:
{
  "support": {"id": "id1"},
  "confusion": {"id": "id2"},
  "counter": {"id": "id3"}
}"""

def parse_args():
    parser = argparse.ArgumentParser(description="Construct SFT data for the Evidence Filter")
    parser.add_argument("--dataset", type=str, default="mer2023", help="Target dataset (mer2023, meld, iemocap)")
    parser.add_argument("--q-data-path", type=str, default="affectagent/artifacts/sft_data/query_planner_sft.jsonl")
    parser.add_argument("--index-dir", type=str, default="affectagent/artifacts/semantic_index")
    parser.add_argument("--output-dir", type=str, default="affectagent/artifacts/sft_data")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--model", type=str, default="gemini-3.1")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--face-or-frame", type=str, default="face", help="Video processing strategy: face or frame")
    return parser.parse_args()

def resolve_annotation_label(annotation, dataset_labels):
    label_value = annotation.get("onehot")
    if isinstance(label_value, str) and label_value:
        return label_value

    if label_value is not None:
        label_array = np.asarray(label_value)
        if label_array.ndim == 0:
            scalar_value = label_array.item()
            if isinstance(scalar_value, str) and scalar_value:
                return scalar_value
        elif label_array.dtype.kind not in {"U", "S", "O"}:
            label_idx = int(np.argmax(label_array))
            if 0 <= label_idx < len(dataset_labels):
                return dataset_labels[label_idx]

    for key in ["discrete", "ovlabel", "label"]:
        fallback_value = annotation.get(key)
        if isinstance(fallback_value, str) and fallback_value:
            return fallback_value

    return "neutral"


def split_candidate_labels(text):
    if not text:
        return []
    parts = re.split(r"[,，;/；、|]+", str(text))
    return [part.strip() for part in parts if part and part.strip()]


def resolve_candidate_labels(dataset):
    explicit_labels = getattr(dataset, "candidate_labels", None)
    if isinstance(explicit_labels, str) and explicit_labels.strip():
        labels = split_candidate_labels(explicit_labels)
        if labels:
            return labels
    elif isinstance(explicit_labels, (list, tuple)):
        labels = [str(label).strip() for label in explicit_labels if str(label).strip()]
        if labels:
            return labels

    label_counter = Counter()
    for ann in getattr(dataset, "annotation", []):
        raw_labels = ann.get("ovlabel", ann.get("discrete", ann.get("label", ann.get("onehot", ""))))
        if isinstance(raw_labels, str):
            for label in split_candidate_labels(raw_labels):
                label_counter[label] += 1

    if label_counter:
        return [label for label, _ in sorted(label_counter.items(), key=lambda item: (-item[1], item[0]))]
    return ["neutral"]

def resolve_sample_subtitle(sample_name, annotation_subtitle, subtitle_lookup, lang):
    preferred_subtitle = subtitle_lookup.get(sample_name, "") if lang == "zh" else annotation_subtitle
    fallback_subtitle = annotation_subtitle if lang == "zh" else subtitle_lookup.get(sample_name, "")

    for candidate in [preferred_subtitle, fallback_subtitle]:
        if pd.isna(candidate):
            continue
        candidate = str(candidate).strip()
        if candidate and candidate.lower() != "nan":
            return candidate

    return ""

def call_gpt_for_s(client, model, sample_id, subtitle, ground_truth, multimodal_desc, candidates, lang="zh"):
    """Call LLM to select 3 evidence IDs"""
    
    # 构造候选文本
    candidates_text = ""
    for group_name in ["support", "confusion", "counter"]:
        group_title = f"【{group_name.capitalize()}组】" if lang == "zh" else f"[{group_name.capitalize()} Group]"
        candidates_text += f"\n{group_title}\n"
        group_items = candidates.get(group_name, [])
        for i, item in enumerate(group_items):
            candidates_text += f"{i+1}. id={item['id']}, {'标签提示' if lang=='zh' else 'Label Hint'}={item['label_hint']}, {'文本' if lang=='zh' else 'Text'}=\"{item['text']}\"\n"
            
    if lang == "zh":
        sys_prompt = S_SYSTEM_PROMPT_ZH
        user_content = (
            f"当前样本字幕: \"{subtitle}\"\n"
            f"当前样本真实标签: {ground_truth}\n"
            f"当前样本多模态特征: {multimodal_desc}\n\n"
            f"候选证据列表:{candidates_text}"
        )
    else:
        sys_prompt = S_SYSTEM_PROMPT_EN
        user_content = (
            f"Current Sample Subtitle: \"{subtitle}\"\n"
            f"Current Sample Ground Truth: {ground_truth}\n"
            f"Current Sample Multimodal Features: {multimodal_desc}\n\n"
            f"Candidate Evidence List:{candidates_text}"
        )

    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.3,
        }
        if "gpt" in model.lower() and "qwen" not in model.lower():
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)

        content = response if isinstance(response, str) else response.choices[0].message.content
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
            
        return json.loads(content.strip())
    except Exception as e:
        print(f"Error calling API for S: {e}")
        return None

def format_sft_instance(sample_id, subtitle, candidates, selected_output, mm_ref=None, lang="zh", face_or_frame="face"):
    """
    Format an SFT training instance for the Evidence Filter.
    Input matches inference phase (without ground_truth and multimodal_desc).
    Uses AffectGPT native prompt format (System:/###Human:/###Assistant:)
    and includes multimodal placeholders consistent with prompts.py build_s_messages.
    """
    candidates_text = ""
    for group_name in ["support", "confusion", "counter"]:
        group_title = f"【{group_name.capitalize()}组】" if lang == "zh" else f"[{group_name.capitalize()} Group]"
        candidates_text += f"\n{group_title}\n"
        for i, item in enumerate(candidates.get(group_name, [])):
            candidates_text += f"{i+1}. id={item['id']}, \"{item.get('text', '')}\"\n"

    video_patch_token = "<FaceHere>" if "face" in face_or_frame else "<FrameHere>"
    s_output_schema = '{"support": {"id": "..."}, "confusion": {"id": "..."}, "counter": {"id": "..."}}'

    if lang == "zh":
        sft_input = (
            "System: 你是一个情感分析证据筛选器。你将收到当前样本的字幕、其多模态特征（视频和音频），"
            "以及三组候选证据（Primary/Confusion/Counter）。"
            "请从每组中分别选出一条最符合当前样本多模态表现的证据，"
            "严格输出仅包含保留证据ID的JSON对象（按 support/confusion/counter 分组）。\n"
            f"###Human: 当前样本字幕: \"{subtitle}\"\n"
            f"<Video>{video_patch_token}</Video>\n"
            f"<Audio><AudioHere></Audio>\n\n"
            f"候选证据列表:{candidates_text}\n\n"
            f"请从每组中各选一条，输出JSON：\n{s_output_schema}\n"
            "###Assistant:"
        )
    else:
        sft_input = (
            "System: You are AffectAgent's multimodal Evidence Filter. You will receive the current sample's "
            "subtitle, its multimodal features (video and audio), and three groups of candidate evidence "
            "(Primary/Confusion/Counter). Please select exactly one evidence item from each group "
            "that best matches the multimodal performance of the current sample. "
            "Strictly output a JSON object grouped by support/confusion/counter with only the retained evidence IDs. "
            "Roles are bound by candidate pool type.\n"
            f"###Human: Current Sample Subtitle: \"{subtitle}\"\n"
            f"<Video>{video_patch_token}</Video>\n"
            f"<Audio><AudioHere></Audio>\n\n"
            f"Candidate Evidence List:{candidates_text}\n\n"
            f"Select one from each group. Output JSON:\n{s_output_schema}\n"
            "###Assistant:"
        )
    
    return {
        "role": "evidence_selector",
        "sample_id": sample_id,
        "input": sft_input,
        "mm_ref": mm_ref,
        "output": json.dumps(selected_output, ensure_ascii=False, indent=2)
    }


def validate_selected_output(selected_output, candidates):
    if not isinstance(selected_output, dict):
        return None
    normalized = {}
    for group_name in ["support", "confusion", "counter"]:
        role_info = selected_output.get(group_name)
        if not isinstance(role_info, dict):
            return None
        selected_id = role_info.get("id")
        valid_ids = {item["id"] for item in candidates.get(group_name, [])}
        if not selected_id or selected_id not in valid_ids:
            return None
        normalized[group_name] = {"id": selected_id}
    return normalized

def override_with_test_split(dataset):
    if hasattr(dataset, "read_test_names") and hasattr(dataset, "get_test_name2gt"):
        try:
            test_names = dataset.read_test_names()
            test_name2gt = dataset.get_test_name2gt()
            test_annotation = []
            for name in test_names:
                item = {"name": name}
                if hasattr(dataset, "name2subtitle"):
                    item["subtitle"] = dataset.name2subtitle.get(name, "")
                gt = test_name2gt.get(name, "")
                item["onehot"] = gt
                item["discrete"] = gt
                item["ovlabel"] = gt
                test_annotation.append(item)
            dataset.annotation = test_annotation
            print(f"[INFO] Dynamically overridden dataset.annotation with TEST split ({len(test_annotation)} samples).")
        except Exception as e:
            print(f"[WARN] Failed to override with test split: {e}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.api_key:
        args.api_key = os.getenv("OPENAI_API_KEY", "")
    if not args.base_url:
        args.base_url = os.getenv("OPENAI_BASE_URL", "")
    if not args.api_key:
        print("Error: OpenAI API key is required. Set --api-key or OPENAI_API_KEY environment variable.")
        return
    lang = "zh" if "mer" in args.dataset.lower() else "en"
    default_desc = "说话者表现出对应情感的典型视听特征。" if lang == "zh" else "The speaker shows typical audiovisual features of the emotion."
    
    # 1. 初始化 Retriever
    print("Initializing Retriever...")
    retriever = DualChannelRetriever(semantic_index_dir=args.index_dir)
    
    # 2. 读取原始数据集 (获取 mme_4o_emotion 和 ground_truth)
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "mer2023":
        dataset = MER2023_Dataset()
        override_with_test_split(dataset)
        df_desc = pd.read_csv("dataset/mer2023-dataset-process/transcription_enhanced.csv")
        df_orig = pd.read_csv("dataset/mer2023-dataset-process/transcription-engchi-polish.csv")
        desc_dict = dict(zip(df_desc["name"], df_desc["mme_4o_emotion"]))
        orig_dict = dict(zip(df_orig["name"], df_orig["chinese"]))
    elif args.dataset == "mer2024":
        dataset = MER2024_Dataset()
        override_with_test_split(dataset)
        df_desc = pd.read_csv("dataset/mer2024-dataset-process/transcription_enhanced.csv")
        df_orig = pd.read_csv("dataset/mer2024-dataset-process/transcription_merge.csv")
        desc_dict = dict(zip(df_desc["name"], df_desc["mme_4o_emotion"]))
        orig_dict = dict(zip(df_orig["name"], df_orig["chinese"]))
    elif args.dataset == "meld":
        dataset = MELD_Dataset()
        override_with_test_split(dataset)
        df_desc = pd.read_csv("dataset/meld-process/transcription_enhanced.csv")
        df_orig = pd.read_csv("dataset/meld-process/transcription-engchi-polish.csv")
        desc_dict = dict(zip(df_desc["name"], df_desc["mme_4o_emotion"]))
        orig_dict = dict(zip(df_orig["name"], df_orig["english"]))
    elif args.dataset == "iemocap":
        dataset = IEMOCAPFour_Dataset()
        override_with_test_split(dataset)
        df_desc = pd.read_csv("dataset/iemocap-process/transcription_enhanced.csv")
        df_orig = pd.read_csv("dataset/iemocap-process/transcription-engchi-polish.csv")
        desc_dict = dict(zip(df_desc["name"], df_desc["mme_4o_emotion"]))
        orig_dict = dict(zip(df_orig["name"], df_orig["english"]))
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

    dataset_labels = resolve_candidate_labels(dataset)
    
    sample_meta = {}
    for ann in dataset.annotation:
        name = ann["name"]
        label = resolve_annotation_label(ann, dataset_labels)
        subtitle = resolve_sample_subtitle(name, ann.get("subtitle", ""), orig_dict, lang)
             
        sample_meta[name] = {
            "subtitle": subtitle,
            "label": label,
            "mme_4o_emotion": desc_dict.get(name, default_desc)
        }
        
    # 3. Read Query Planner SFT records.
    q_data = []
    if not os.path.exists(args.q_data_path):
        print(f"Error: Query Planner data not found at {args.q_data_path}")
        return
        
    with open(args.q_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                q_data.append(json.loads(line))
                
    if args.max_samples > 0:
        q_data = q_data[:args.max_samples]
        
    print(f"Processing {len(q_data)} samples for the Evidence Filter...")
    
    client = OpenAI(api_key=args.api_key, base_url=args.base_url) if args.base_url else OpenAI(api_key=args.api_key)
    
    sft_data = []
    
    # 4. 检索并使用 LLM 生成选择
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for item in q_data:
            sample_id = item["sample_id"]
            if sample_id not in sample_meta:
                continue
                
            # 解析 Q 的输出
            try:
                q_output = json.loads(item["output"])
            except:
                continue
                
            # 检索 Channel A (这里在主线程或子线程都可以，SentenceTransformer 在多线程下可能需要注意，但 FAISS 读取是线程安全的)
            candidates = retriever.retrieve_channel_A(q_output, top_k=3, exclude_sample_id=sample_id)
            
            meta = sample_meta[sample_id]
            future = executor.submit(
                call_gpt_for_s, client, args.model, sample_id, meta["subtitle"], 
                meta["label"], meta["mme_4o_emotion"], candidates, lang
            )
            futures[future] = (sample_id, meta["subtitle"], candidates, item.get("mm_ref"))
            
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating S-Selections"):
            sample_id, subtitle, candidates, mm_ref = futures[future]
            result = future.result()
            validated_output = validate_selected_output(result, candidates)
            
            if validated_output:
                sft_instance = format_sft_instance(
                    sample_id=sample_id,
                    subtitle=subtitle,
                    candidates=candidates,
                    selected_output=validated_output,
                    mm_ref=mm_ref,
                    lang=lang,
                    face_or_frame=args.face_or_frame,
                )
                sft_data.append(sft_instance)
                
    output_path = os.path.join(args.output_dir, "evidence_filter_sft.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Evidence Filter SFT data saved to {output_path} ({len(sft_data)} samples)")

if __name__ == "__main__":
    main()
