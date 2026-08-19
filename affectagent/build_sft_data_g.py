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

G_SYSTEM_PROMPT_ZH = """你是一个多模态情感分析推理生成器（Emotion Generator）。
你的任务是根据当前样本的特征和提供的参考证据，生成严谨的推理过程，并判断最终情感。

你将收到：
1. [当前样本] 的字幕、真实标签和多模态特征。
2. [支持证据 - support]：验证特定情感主假设的证据。
3. [混淆证据 - confusion]：用于区分易混淆情感的证据。
4. [反面证据 - counter]：用于检验主假设是否成立的反证。
5. 融合增强信息（用于告诉你当前样本吸收了哪个相似样本的特征，仅供参考）。

请输出JSON格式，包含以下字段：
- "prediction": 根据特征推断出的情感标签（必须与真实标签一致，用于SFT）
- "confidence": 对判断的置信度 (0.0~1.0)
- "reasoning": 2-4句推理过程。必须引用[支持/对比/反面证据]中的内容，并结合当前样本的多模态特征进行对比分析。

输出格式要求：
{
  "prediction": "angry",
  "confidence": 0.85,
  "reasoning": "待分析样本的音频特征经融合后与support证据的被动攻击模式一致。对比confusion证据的平和语气，待分析样本明显更紧张急促。counter证据描述的真诚认同与待分析样本不符。综合判断为angry。"
}"""

G_SYSTEM_PROMPT_EN = """You are a multimodal emotion analysis Emotion Generator.
Your task is to generate rigorous reasoning based on the current sample's features and the provided reference evidence, and predict the final emotion.

You will receive:
1. The subtitle, ground truth label, and multimodal features of the [Current Sample].
2. [Support Evidence]: Evidence to verify the main hypothesis of a specific emotion.
3. [Confusion Evidence]: Evidence used to distinguish easily confused emotions.
4. [Counter Evidence]: Counter-evidence used to test whether the main hypothesis holds.
5. Fusion enhancement information (tells you which similar sample's features the current sample has absorbed, for reference only).

Please output in JSON format, containing the following fields:
- "prediction": The emotion label inferred from features (must be consistent with the ground truth label, used for SFT)
- "confidence": Confidence in the judgment (0.0~1.0)
- "reasoning": A 2-4 sentence reasoning process. You must cite content from the [Support/Confusion/Counter] evidence and analyze the current sample's multimodal features.

Output format requirement:
{
  "prediction": "angry",
  "confidence": 0.85,
  "reasoning": "The audio features of the analyzed sample, after fusion, are consistent with the passive-aggressive mode of the support evidence. Compared with the confusion evidence, the analyzed sample is more tense and rushed. The counter evidence does not match the analyzed sample. Overall judgment is angry."
}"""

def parse_args():
    parser = argparse.ArgumentParser(description="Construct SFT data for the Emotion Generator")
    parser.add_argument("--dataset", type=str, default="mer2023", help="Target dataset (mer2023, meld, iemocap)")
    parser.add_argument("--s-data-path", type=str, default="affectagent/artifacts/sft_data/evidence_filter_sft.jsonl")
    parser.add_argument("--index-dir", type=str, default="affectagent/artifacts/semantic_index")
    parser.add_argument("--multimodal-index-dir", type=str, default="retrieval/faiss/artifacts/mercaptionplus")
    parser.add_argument("--fusion-partner-map", type=str, default="", help="JSON file produced in Phase 1c that maps sample_id to fusion_partner_id")
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

def call_gpt_for_g(client, model, sample_id, subtitle, ground_truth, multimodal_desc, ev_texts, fusion_partner_id, lang="zh"):
    """Call the teacher model to generate Emotion Generator supervision."""
    
    evidence_prompt = ""
    for role, ev in ev_texts.items():
        if ev:
            role_map = {"support": "支持", "confusion": "混淆", "counter": "反面"}
            role_name = role_map.get(role, role) if lang == "zh" else role.capitalize()
            evidence_prompt += f"\n[{role_name}{'证据' if lang=='zh' else ' Evidence'} - {'意图' if lang=='zh' else 'Intent'}: {ev.get('subquery_type', '')}]\n\"{ev['text']}\"\n"
            
    if lang == "zh":
        sys_prompt = G_SYSTEM_PROMPT_ZH
        user_content = (
            f"{evidence_prompt}\n"
            f"[待分析样本]\n"
            f"字幕: \"{subtitle}\"\n"
            f"真实标签: {ground_truth}\n"
            f"多模态特征: {multimodal_desc}\n"
            f"（视听特征已与感知最相似样本 {fusion_partner_id} 在特征层融合增强）\n\n"
            f"请综合判断情感，并输出带有 reasoning 的 JSON。"
        )
    else:
        sys_prompt = G_SYSTEM_PROMPT_EN
        user_content = (
            f"{evidence_prompt}\n"
            f"[Current Sample]\n"
            f"Subtitle: \"{subtitle}\"\n"
            f"Ground Truth: {ground_truth}\n"
            f"Multimodal Features: {multimodal_desc}\n"
            f"(Audiovisual features have been fused and enhanced at the feature level with the most perceptually similar sample {fusion_partner_id})\n\n"
            f"Please comprehensively judge the emotion and output JSON with reasoning."
        )

    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.5,
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
        print(f"Error calling API for G: {e}")
        return None

EVIDENCE_INTENT_ZH = {
    "support": "验证主假设",
    "confusion": "区分易混淆情感",
    "counter": "寻找反证检验主假设是否成立",
}
EVIDENCE_INTENT_EN = {
    "support": "verify main hypothesis",
    "confusion": "distinguish easily confused emotions",
    "counter": "find counter-evidence to test the main hypothesis",
}

def _format_evidence_text_sft(ev_texts, lang="zh"):
    """Format evidence text consistent with prompts.py _format_evidence_text."""
    text = ""
    role_map_zh = {"support": "支持", "confusion": "混淆", "counter": "反面"}
    intent_map = EVIDENCE_INTENT_ZH if lang == "zh" else EVIDENCE_INTENT_EN
    for role, ev in ev_texts.items():
        if ev is None:
            continue
        label_hint = ev.get("label_hint", "")
        ev_text = ev.get("text", "")
        intent = intent_map.get(role, "")
        if lang == "zh":
            role_name = role_map_zh.get(role, role)
            header = f"[{role_name}证据 — {role}, {label_hint}]" if label_hint else f"[{role_name}证据 — {role}]"
            text += f"\n{header}\n检索意图: {intent}\n\"{ev_text}\"\n"
        else:
            header = f"[{role.capitalize()} Evidence — {role}, {label_hint}]" if label_hint else f"[{role.capitalize()} Evidence — {role}]"
            text += f"\n{header}\nRetrieval Intent: {intent}\n\"{ev_text}\"\n"
    return text


def format_sft_instance(sample_id, subtitle, ev_texts, fusion_partner_id, gpt_output, candidate_labels, current_mm_ref=None, partner_mm_ref=None, lang="zh", face_or_frame="face"):
    """
    Format an SFT training instance for the Emotion Generator.
    Uses AffectGPT native prompt format (System:/###Human:/###Assistant:)
    with multimodal placeholders and evidence formatting matching prompts.py build_g_messages.
    """
    evidence_prompt = _format_evidence_text_sft(ev_texts, lang)
    labels_str = ", ".join(candidate_labels)
    video_patch_token = "<FaceHere>" if "face" in face_or_frame else "<FrameHere>"

    if lang == "zh":
        sft_input = (
            "System: 你是一个多模态情感分析器。你将看到一个待分析样本（含融合增强后的视听特征）"
            "和几个参考证据。每个证据标注了检索意图。请综合判断情感并输出JSON。\n"
            f"###Human: {evidence_prompt}\n"
            f"[待分析样本]\n"
            f"\"{subtitle}\"\n"
            f"视觉: <Video>{video_patch_token}</Video>\n"
            f"听觉: <Audio><AudioHere></Audio>\n"
            f"（视听特征已与感知最相似样本在特征层融合增强）\n"
            f"候选标签池（参考）: {labels_str}\n\n"
            f"请综合判断情感。你可以优先参考候选标签池，也可以输出与其语义最接近的开放词汇情感标签：\n"
            f"- 支持证据描述了类似的情感模式，用于参照\n"
            f"- 对比证据用于区分易混淆情感\n"
            f"- 反面证据用于检验主假设是否成立\n\n"
            f"请输出JSON，包含 prediction, confidence, reasoning 三个字段。\n"
            "###Assistant:"
        )
    else:
        sft_input = (
            "System: You are a multimodal emotion analyzer. You will see a sample to be analyzed "
            "(containing fused enhanced audiovisual features) and several reference evidence items. "
            "Each evidence has a retrieval intent annotated. Please comprehensively judge the emotion "
            "and output JSON.\n"
            f"###Human: {evidence_prompt}\n"
            f"[Current Sample]\n"
            f"\"{subtitle}\"\n"
            f"Visual: <Video>{video_patch_token}</Video>\n"
            f"Audio: <Audio><AudioHere></Audio>\n"
            f"(Audiovisual features have been fused and enhanced with the perceptually most similar sample)\n"
            f"Candidate Label Pool (reference only): {labels_str}\n\n"
            f"Please comprehensively judge the emotion. You may prioritize the candidate label pool, "
            f"or output the closest open-vocabulary emotion label if needed:\n"
            f"- Support evidence describes a similar emotional pattern, used for reference.\n"
            f"- Contrast evidence is used to distinguish easily confused emotions.\n"
            f"- Counter evidence is used to test whether the main hypothesis holds.\n\n"
            f"Output JSON with fields: prediction, confidence, reasoning.\n"
            "###Assistant:"
        )
    
    return {
        "role": "emotion_generator",
        "sample_id": sample_id,
        "input": sft_input,
        "current_mm_ref": current_mm_ref,
        "fusion_partner_id": fusion_partner_id,
        "partner_mm_ref": partner_mm_ref,
        "selected_evidence": ev_texts,
        "output": json.dumps(gpt_output, ensure_ascii=False, indent=2)
    }


def load_fusion_partner_map(path):
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"fusion_partner_map not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("fusion_partner_map must be a JSON object: {sample_id: fusion_partner_id}")
    return data


def build_token_store_ref(multimodal_index_dir, sample_id, global_id):
    token_store_path = os.path.abspath(os.path.join(multimodal_index_dir, "token_store", "tokens.h5"))
    id_to_row_path = os.path.abspath(os.path.join(multimodal_index_dir, "token_store", "id_to_row.json"))
    if not os.path.isfile(token_store_path) or not os.path.isfile(id_to_row_path):
        raise FileNotFoundError(
            f"Missing token store artifacts under {multimodal_index_dir}. Expected token_store/tokens.h5 and token_store/id_to_row.json"
        )
    if global_id is None:
        raise ValueError(f"global_id is required to build mm_ref for sample {sample_id}")
    return {
        "source": "multimodal_token_store",
        "sample_id": sample_id,
        "global_id": int(global_id),
        "token_store_path": token_store_path,
        "id_to_row_path": id_to_row_path,
    }


def resolve_fusion_partner_record(partner_info, multimodal_name_to_global_id, multimodal_index_dir):
    if isinstance(partner_info, str):
        fusion_partner_id = partner_info
        partner_global_id = multimodal_name_to_global_id.get(fusion_partner_id)
        partner_mm_ref = build_token_store_ref(multimodal_index_dir, fusion_partner_id, partner_global_id)
        return fusion_partner_id, partner_mm_ref
    if isinstance(partner_info, dict):
        fusion_partner_id = partner_info.get("fusion_partner_id") or partner_info.get("partner_id") or partner_info.get("sample_id")
        if not fusion_partner_id:
            raise ValueError("fusion_partner_map entry must contain fusion_partner_id")
        partner_global_id = partner_info.get("partner_global_id")
        if partner_global_id is None:
            partner_global_id = multimodal_name_to_global_id.get(fusion_partner_id)
        partner_mm_ref = partner_info.get("partner_mm_ref")
        if partner_mm_ref is None:
            partner_mm_ref = build_token_store_ref(multimodal_index_dir, fusion_partner_id, partner_global_id)
        return fusion_partner_id, partner_mm_ref
    raise ValueError("fusion_partner_map values must be strings or JSON objects")


def validate_g_output(gpt_output):
    if not isinstance(gpt_output, dict):
        return None
    prediction = gpt_output.get("prediction")
    confidence = gpt_output.get("confidence")
    reasoning = gpt_output.get("reasoning")
    if prediction is None or confidence is None or reasoning is None:
        return None
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return None
    if confidence < 0.0 or confidence > 1.0:
        return None
    return {
        "prediction": prediction,
        "confidence": confidence,
        "reasoning": reasoning,
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
    lang = "zh" if "mer" in args.dataset.lower() else "en"
    default_desc = "说话者表现出对应情感的典型视听特征。" if lang == "zh" else "The speaker shows typical audiovisual features of the emotion."
    
    # 1. 读数据集获取元信息
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
        
    # 2. 读取 Retriever 库 (为了根据 S 选出的 ID 查出具体 Text)
    print("Loading Retriever Index...")
    retriever = DualChannelRetriever(
        semantic_index_dir=args.index_dir,
        multimodal_index_dir=args.multimodal_index_dir,
    )
    id2meta = {m["name"]: m for m in retriever.metadata}
    multimodal_name_to_global_id = {
        item.get("name"): gid for gid, item in retriever.multimodal_meta.items() if item.get("name")
    }
    fusion_partner_map = load_fusion_partner_map(args.fusion_partner_map)
    
    # 3. Read Evidence Filter SFT records.
    s_data = []
    if not os.path.exists(args.s_data_path):
        print(f"Error: Evidence Filter data not found at {args.s_data_path}")
        return
        
    with open(args.s_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                s_data.append(json.loads(line))
                
    if args.max_samples > 0:
        s_data = s_data[:args.max_samples]
        
    print(f"Processing {len(s_data)} samples for the Emotion Generator...")
    if not fusion_partner_map:
        raise RuntimeError(
            "Emotion Generator SFT construction requires a precomputed fusion_partner_map. "
            "Please provide --fusion-partner-map instead of using a mock Channel B partner."
        )
    missing_partner_ids = [item["sample_id"] for item in s_data if item.get("sample_id") not in fusion_partner_map]
    if missing_partner_ids:
        preview = ", ".join(missing_partner_ids[:5])
        raise KeyError(
            f"fusion_partner_map is missing {len(missing_partner_ids)} sample ids, e.g. {preview}"
        )
    
    client = OpenAI(api_key=args.api_key, base_url=args.base_url) if args.base_url else OpenAI(api_key=args.api_key)
    
    g_sft_data = []
    
    # 4. Construct Emotion Generator training records.
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for item in s_data:
            sample_id = item["sample_id"]
            if sample_id not in sample_meta:
                continue
            current_mm_ref = item.get("mm_ref")
            if not isinstance(current_mm_ref, dict):
                continue
                
            try:
                s_output = json.loads(item["output"])
            except:
                continue
                
            # 根据 S 选出的 ID，提取具体的证据文本并分配角色
            ev_texts = {"support": None, "confusion": None, "counter": None}
            role_map = {"support": "support", "confusion": "confusion", "counter": "counter"}
            
            for source_role, target_role in role_map.items():
                role_info = s_output.get(source_role, {})
                sid = role_info.get("id") if isinstance(role_info, dict) else None
                if sid and sid in id2meta:
                    meta = id2meta[sid]
                    description = meta.get("description", meta.get("reason", ""))
                    global_id = meta.get("global_id")
                    ev_texts[target_role] = {
                        "id": sid,
                        "global_id": global_id,
                        "text": f"{meta.get('subtitle', '')}。{description}".strip("。"),
                        "subquery_type": source_role,
                        "mm_ref": build_token_store_ref(args.multimodal_index_dir, sid, global_id)
                    }
            if any(ev_texts[role] is None for role in ["support", "confusion", "counter"]):
                continue
                    
            fusion_partner_id, partner_mm_ref = resolve_fusion_partner_record(
                fusion_partner_map[sample_id], multimodal_name_to_global_id, args.multimodal_index_dir
            )
            
            meta = sample_meta[sample_id]
            future = executor.submit(
                call_gpt_for_g, client, args.model, sample_id, meta["subtitle"], 
                meta["label"], meta["mme_4o_emotion"], ev_texts, fusion_partner_id, lang
            )
            futures[future] = (sample_id, meta["subtitle"], ev_texts, fusion_partner_id, current_mm_ref, partner_mm_ref)
            
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating G-Reasoning"):
            sample_id, subtitle, ev_texts, fusion_partner_id, current_mm_ref, partner_mm_ref = futures[future]
            result = future.result()
            validated_output = validate_g_output(result)
            
            if validated_output:
                sft_instance = format_sft_instance(
                    sample_id=sample_id,
                    subtitle=subtitle,
                    ev_texts=ev_texts,
                    fusion_partner_id=fusion_partner_id,
                    gpt_output=validated_output,
                    candidate_labels=dataset_labels,
                    current_mm_ref=current_mm_ref,
                    partner_mm_ref=partner_mm_ref,
                    lang=lang,
                    face_or_frame=args.face_or_frame,
                )
                g_sft_data.append(sft_instance)
                
    output_path = os.path.join(args.output_dir, "emotion_generator_sft.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in g_sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Emotion Generator SFT data saved to {output_path} ({len(g_sft_data)} samples)")

if __name__ == "__main__":
    main()
