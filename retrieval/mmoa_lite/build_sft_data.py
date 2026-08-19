import os
import sys
import json
import argparse
import re
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# 将项目根目录加入 sys.path，以便能够导入 my_affectgpt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai package not installed. SFT data generation requires it.")

from my_affectgpt.datasets.datasets.mer2023 import MER2023_Dataset
from my_affectgpt.datasets.datasets.mer2024 import MER2024_Dataset
from my_affectgpt.datasets.datasets.meld import MELD_Dataset
from my_affectgpt.datasets.datasets.iemocap import IEMOCAPFour_Dataset


Q_SYSTEM_PROMPT_ZH = """你是一个多模态情感分析检索查询规划器（Query Planner）。
你的任务是将复杂的情感判断任务拆解为3条有针对性的自然语言检索查询（Query），用于在外部情感证据库中检索相关样本。

你将收到以下信息：
1. 真实的情感标签 (Ground Truth)
2. 说话者的字幕 (Subtitle)
3. 视频和音频的多模态特征描述 (Multimodal Features)
4. 候选的情感标签列表

请根据上述信息，严格输出包含3条子查询的纯JSON。不要输出任何解释，不要包含 Markdown 格式的 ```json ```，只输出合法的JSON对象。

输出格式要求：
{
  "primary": {
    "query_text": "描述支持当前情感的典型多模态表现和语境。例如：说话者表面微笑但语气冰冷，表现出压抑的愤怒",
    "target_label": "ground_truth_label"
  },
  "confusion": {
    "query_text": "描述如何区分当前情感和最容易混淆的另一种情感。例如：区分真正的平静和压抑愤怒的假装平静，关键看面部肌肉是否紧绷",
    "contrast_label": "confusing_label"
  },
  "counter": {
    "query_text": "描述当前情感的反面表现，用于寻找反证。例如：说话者真心认同，语气舒缓平和没有攻击性",
    "counter_direction": "opposite_label"
  }
}"""

Q_SYSTEM_PROMPT_EN = """You are a multimodal emotion analysis query planner (Query Planner).
Your task is to decompose a complex emotion judgment task into 3 targeted natural language retrieval queries, used to retrieve relevant samples from an external emotion evidence database.

You will receive the following information:
1. Ground Truth Emotion Label
2. Speaker's Subtitle
3. Multimodal feature description of video and audio
4. Candidate emotion labels list

Based on the above information, strictly output a pure JSON object containing 3 sub-queries. Do not output any explanations, do not include Markdown formatting like ```json ```, only output a valid JSON object.

Output format requirement:
{
  "primary": {
    "query_text": "Describe the typical multimodal performance and context supporting the current emotion. Example: The speaker smiles superficially but has a cold tone, showing suppressed anger.",
    "target_label": "ground_truth_label"
  },
  "confusion": {
    "query_text": "Describe how to distinguish the current emotion from the most easily confused one. Example: Distinguish true calmness from fake calmness hiding anger, key is whether facial muscles are tense.",
    "contrast_label": "confusing_label"
  },
  "counter": {
    "query_text": "Describe the opposite performance of the current emotion, used to find counter-evidence. Example: The speaker truly agrees, with a soothing and peaceful tone and no aggressiveness.",
    "counter_direction": "opposite_label"
  }
}"""

def parse_args():
    parser = argparse.ArgumentParser(description="Construct SFT data for Agent-Q")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mer2023",
        help="Target dataset to build SFT data (e.g., mer2023, meld)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="retrieval/mmoa_lite/artifacts/sft_data",
        help="Directory to save SFT data",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API Key (OpenAI or Bailian). If not provided, will check OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Base URL for the API. If not provided, will check OPENAI_BASE_URL environment variable.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.1",
        help="Model to use for generating queries",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (for debugging)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of concurrent API requests",
    )
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


def call_gpt4o_for_q(client, model, sample_data, candidate_labels, lang="zh"):
    """
    Call GPT-4o to generate the 3 queries for Agent-Q.
    """
    subtitle = sample_data.get("subtitle", "")
    ground_truth = sample_data.get("label", "neutral")
    multimodal_desc = sample_data.get("mme_4o_emotion", "说话者表现出对应情感的典型视听特征。" if lang == "zh" else "The speaker shows typical audiovisual features of the emotion.")

    if lang == "zh":
        sys_prompt = Q_SYSTEM_PROMPT_ZH
        user_content = (
            f"字幕: {subtitle}\n"
            f"真实情感标签: {ground_truth}\n"
            f"多模态特征描述: {multimodal_desc}\n"
            f"候选标签: {', '.join(candidate_labels)}\n"
        )
    else:
        sys_prompt = Q_SYSTEM_PROMPT_EN
        user_content = (
            f"Subtitle: {subtitle}\n"
            f"Ground Truth Label: {ground_truth}\n"
            f"Multimodal Features: {multimodal_desc}\n"
            f"Candidate Labels: {', '.join(candidate_labels)}\n"
        )

    max_retries = 5
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content}
                ],
                "temperature": 0.7,
            }
            # Only use JSON mode for OpenAI models, as deepseek/gemini/qwen proxies might fail or return strings
            if "gpt" in model.lower() and "qwen" not in model.lower():
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)

            if isinstance(response, str):
                print(f"[WARN] Proxy returned string instead of object: {response}")
                content = response
            else:
                content = response.choices[0].message.content

            # 处理部分模型（如deepseek等）可能会包裹markdown的情况
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            result = json.loads(content.strip())

            # Rate limit protection: sleep a bit after success to avoid triggering RPM limits
            time.sleep(2.0)
            return result

        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "不可用" in error_str or "429" in error_str:
                # Ratelimit or banned token, sleep longer and retry
                sleep_time = 10 * (attempt + 1)
                print(f"\n[RATE LIMIT / 401] Triggered. Sleeping {sleep_time}s before retry {attempt+1}/{max_retries}...")
                time.sleep(sleep_time)
            else:
                print(f"Error calling API: {e}")
                time.sleep(2)

    print(f"[FAIL] Exceeded max retries for sample {sample_data.get('name', 'unknown')}")
    return None


def format_sft_instance(sample_id, subtitle, candidate_labels, gpt4_output, mm_ref=None, lang="zh"):
    """
    Format the generated query into the SFT training instance format for AffectGPT.
    Note: The input replaces `mme_4o_emotion` with placeholders for raw tokens.
    """
    q_output_schema = '{\n  "primary":   {"query_text": "...", "target_label": "..."},\n  "confusion": {"query_text": "...", "contrast_label": "..."},\n  "counter":   {"query_text": "...", "counter_direction": "..."}\n}'
    if lang == "zh":
        sft_input = (
            "System: 你是一个情感分析查询规划器。根据字幕和候选标签，生成三条检索查询。"
            "每条查询是一段自然语言描述，用于在情感样本库中检索证据。严格输出JSON格式。\n"
            f"###Human: 字幕: \"{subtitle}\"\n"
            f"候选标签: {', '.join(candidate_labels)}\n\n"
            f"请输出如下格式的JSON:\n{q_output_schema}\n"
            "###Assistant:"
        )
    else:
        sft_input = (
            "System: You are an emotion analysis query planner. Based on the subtitle and candidate labels, "
            "generate three retrieval queries. Each query is a natural language description used to "
            "retrieve evidence in an emotion sample database. Output strictly in JSON format.\n"
            f"###Human: Subtitle: \"{subtitle}\"\n"
            f"Candidate Labels: {', '.join(candidate_labels)}\n\n"
            f"Please output JSON in the following format:\n{q_output_schema}\n"
            "###Assistant:"
        )

    return {
        "role": "query_planner",
        "sample_id": sample_id,
        "input": sft_input,
        "mm_ref": mm_ref,
        "output": json.dumps(gpt4_output, ensure_ascii=False, indent=2)
    }


def build_current_sample_mm_ref(dataset, sample):
    video_path = dataset._get_video_path(sample) if hasattr(dataset, "_get_video_path") else None
    audio_path = dataset._get_audio_path(sample) if hasattr(dataset, "_get_audio_path") else None
    face_path = dataset._get_face_path(sample) if hasattr(dataset, "_get_face_path") else None
    return {
        "source": "raw_sample",
        "sample_id": sample["name"],
        "video_path": os.path.abspath(video_path) if video_path and os.path.exists(video_path) else None,
        "audio_path": os.path.abspath(audio_path) if audio_path and os.path.exists(audio_path) else None,
        "face_path": os.path.abspath(face_path) if face_path and os.path.exists(face_path) else None,
    }


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

    try:
        if args.base_url:
            client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        else:
            client = OpenAI(api_key=args.api_key)
    except NameError:
        print("Please install openai: pip install openai")
        return

    print(f"Loading {args.dataset} dataset...")
    lang = "zh" if "mer" in args.dataset.lower() else "en"
    default_desc = "说话者表现出对应情感的典型视听特征。" if lang == "zh" else "The speaker shows typical audiovisual features of the emotion."

    # 根据用户计划，应该使用训练集 (如 mer2023, meld) 来构造 SFT 数据
    if args.dataset == "mer2023":
        dataset = MER2023_Dataset()
        override_with_test_split(dataset)
        df_desc = pd.read_csv("dataset/mer2023-dataset-process/transcription_enhanced.csv")
        df_orig = pd.read_csv("dataset/mer2023-dataset-process/transcription-engchi-polish.csv")
        desc_dict = dict(zip(df_desc["name"], df_desc["mme_4o_emotion"]))
        orig_dict = dict(zip(df_orig["name"], df_orig["chinese"]))
        dataset_label_space = resolve_candidate_labels(dataset)

        samples = []
        for ann in dataset.annotation:
            name = ann["name"]
            label = resolve_annotation_label(ann, dataset_label_space)
            subtitle = resolve_sample_subtitle(name, ann.get("subtitle", ""), orig_dict, lang)

            samples.append({
                "name": name,
                "subtitle": subtitle,
                "label": label,
                "mme_4o_emotion": desc_dict.get(name, default_desc)
            })
    elif args.dataset == "mer2024":
        dataset = MER2024_Dataset()
        override_with_test_split(dataset)
        df_desc = pd.read_csv("dataset/mer2024-dataset-process/transcription_enhanced.csv")
        df_orig = pd.read_csv("dataset/mer2024-dataset-process/transcription_merge.csv")
        desc_dict = dict(zip(df_desc["name"], df_desc["mme_4o_emotion"]))
        orig_dict = dict(zip(df_orig["name"], df_orig["chinese"]))  # mer2024 no english col
        dataset_label_space = resolve_candidate_labels(dataset)

        samples = []
        for ann in dataset.annotation:
            name = ann["name"]
            label = resolve_annotation_label(ann, dataset_label_space)
            subtitle = resolve_sample_subtitle(name, ann.get("subtitle", ""), orig_dict, lang)

            samples.append({
                "name": name,
                "subtitle": subtitle,
                "label": label,
                "mme_4o_emotion": desc_dict.get(name, default_desc)
            })
    elif args.dataset == "meld":
        dataset = MELD_Dataset()
        override_with_test_split(dataset)
        df_desc = pd.read_csv("dataset/meld-process/transcription_enhanced.csv")
        df_orig = pd.read_csv("dataset/meld-process/transcription-engchi-polish.csv")
        desc_dict = dict(zip(df_desc["name"], df_desc["mme_4o_emotion"]))
        orig_dict = dict(zip(df_orig["name"], df_orig["english"]))
        dataset_label_space = resolve_candidate_labels(dataset)

        samples = []
        for ann in dataset.annotation:
            name = ann["name"]
            label = resolve_annotation_label(ann, dataset_label_space)
            subtitle = resolve_sample_subtitle(name, ann.get("subtitle", ""), orig_dict, lang)

            samples.append({
                "name": name,
                "subtitle": subtitle,
                "label": label,
                "mme_4o_emotion": desc_dict.get(name, default_desc)
            })
    elif args.dataset == "iemocap":
        dataset = IEMOCAPFour_Dataset()
        override_with_test_split(dataset)
        df_desc = pd.read_csv("dataset/iemocap-process/transcription_enhanced.csv")
        df_orig = pd.read_csv("dataset/iemocap-process/transcription-engchi-polish.csv")
        desc_dict = dict(zip(df_desc["name"], df_desc["mme_4o_emotion"]))
        orig_dict = dict(zip(df_orig["name"], df_orig["english"]))
        dataset_label_space = resolve_candidate_labels(dataset)

        samples = []
        for ann in dataset.annotation:
            name = ann["name"]
            label = resolve_annotation_label(ann, dataset_label_space)
            subtitle = resolve_sample_subtitle(name, ann.get("subtitle", ""), orig_dict, lang)

            samples.append({
                "name": name,
                "subtitle": subtitle,
                "label": label,
                "mme_4o_emotion": desc_dict.get(name, default_desc)
            })
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented for SFT yet.")
    
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    print(f"Processing {len(samples)} samples to generate Agent-Q SFT data...")

    dataset_labels = resolve_candidate_labels(dataset)
    
    sft_data = []
    
    # Use ThreadPoolExecutor for concurrent API requests
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(call_gpt4o_for_q, client, args.model, sample, dataset_labels, lang): (idx, sample)
            for idx, sample in enumerate(samples)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Queries"):
            idx, sample = futures[future]
            result = future.result()
            
            if result:
                mm_ref = build_current_sample_mm_ref(dataset, sample)
                sft_instance = format_sft_instance(
                    sample_id=sample["name"],
                    subtitle=sample.get("subtitle", ""),
                    candidate_labels=dataset_labels,
                    gpt4_output=result,
                    mm_ref=mm_ref,
                    lang=lang
                )
                sft_data.append(sft_instance)

    output_path = os.path.join(args.output_dir, "agent_q_sft.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Agent-Q SFT data saved to {output_path} ({len(sft_data)} samples)")


if __name__ == "__main__":
    main()
