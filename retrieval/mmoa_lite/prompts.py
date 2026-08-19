"""
prompts.py — Agent-Q / Agent-S / Agent-G 的 prompt 构造与输出解析
用于 RL rollout 阶段，将样本信息组装成 chat messages，并将模型输出解析为结构化对象。
注意：RL 阶段的 prompt 与 SFT 阶段的 input 保持一致（不泄露 ground_truth 和 mme_4o_emotion）。
"""

import json
from typing import List, Dict, Optional
try:
    from .schemas import QueryOutput, SelectorOutput, GeneratorOutput
except ImportError:
    from retrieval.mmoa_lite.schemas import QueryOutput, SelectorOutput, GeneratorOutput


# ═══════════════════════════════════════════════════════════════════════════════
# Agent-Q Prompts (RL 阶段：不含 ground_truth，模型需自行推断)
# ═══════════════════════════════════════════════════════════════════════════════

Q_SYSTEM_ZH = (
    "你是一个情感分析查询规划器。根据字幕和候选标签，生成三条检索查询，"
    "用于在情感样本库中找到不同类型的参考样本。\n"
    "每条查询应描述目标样本的说话方式、语气和情感特征（而非抽象的检索意图），"
    "并在末尾附上目标情感标签词（英文）。\n"
    "查询风格示例：\"说话者对反复出现的问题表现出不满和烦躁，语气急促带有攻击性 angry, frustrated\"\n\n"
    "三条查询的检索目标：\n"
    "- primary: 描述一个与当前样本情感一致的参考样本的特征\n"
    "- confusion: 描述一个容易与当前情感混淆的不同情感样本的特征\n"
    "- counter: 描述一个情感与当前样本相反的样本的特征\n\n"
    "严格输出JSON格式。"
)

Q_SYSTEM_EN = (
    "You are an emotion analysis query planner. Based on the subtitle and candidate labels, "
    "generate three retrieval queries to find different types of reference samples "
    "in an emotion sample database.\n"
    "Each query should describe the target sample's speaking style, tone, and emotional "
    "characteristics (NOT abstract retrieval intents), with target emotion label words "
    "appended at the end.\n"
    "Query style example: \"The speaker shows dissatisfaction about recurring problems "
    "with an impatient and aggressive tone. angry, frustrated\"\n\n"
    "Three query retrieval targets:\n"
    "- primary: Describe a reference sample with the same emotion as the current sample\n"
    "- confusion: Describe a sample with a different but easily confused emotion\n"
    "- counter: Describe a sample with the opposite emotion\n\n"
    "Output strictly in JSON format."
)

Q_OUTPUT_SCHEMA = """{
  "primary":   {"query_text": "...", "target_label": "..."},
  "confusion": {"query_text": "...", "contrast_label": "..."},
  "counter":   {"query_text": "...", "counter_direction": "..."}
}"""


def build_q_messages(subtitle: str, candidate_labels: List[str], lang: str = "zh") -> List[Dict]:
    labels_str = ", ".join(candidate_labels)
    if lang == "zh":
        return [
            {"role": "system", "content": Q_SYSTEM_ZH},
            {"role": "user", "content": (
                f"字幕: \"{subtitle}\"\n"
                f"候选标签: {labels_str}\n\n"
                f"请输出如下格式的JSON:\n{Q_OUTPUT_SCHEMA}"
            )},
        ]
    else:
        return [
            {"role": "system", "content": Q_SYSTEM_EN},
            {"role": "user", "content": (
                f"Subtitle: \"{subtitle}\"\n"
                f"Candidate Labels: {labels_str}\n\n"
                f"Please output JSON in the following format:\n{Q_OUTPUT_SCHEMA}"
            )},
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# Agent-S Prompts
# ═══════════════════════════════════════════════════════════════════════════════

S_SYSTEM_ZH = (
    "你是一个情感分析证据筛选器。你将收到当前样本的字幕、其多模态特征（视频和音频），"
    "以及三组候选证据（Primary/Confusion/Counter）。"
    "请从每组中分别选出一条最符合当前样本多模态表现的证据，"
    "严格输出仅包含选中证据ID的JSON对象（按 primary/confusion/counter 分组）。角色由候选池类型强绑定。"
)

S_SYSTEM_EN = (
    "You are an emotion analysis Evidence Selector. You will receive the current sample's "
    "subtitle, its multimodal features (video and audio), and three groups of candidate evidence "
    "(Primary/Confusion/Counter). Please select exactly one evidence item from each group "
    "that best matches the multimodal performance of the current sample. "
    "Strictly output a JSON object grouped by primary/confusion/counter with only the selected evidence IDs. "
    "Roles are bound by candidate pool type."
)


def _format_candidates_text(candidates: Dict[str, list], lang: str = "zh") -> str:
    text = ""
    for group_name in ["primary", "confusion", "counter"]:
        group_title = f"【{group_name.capitalize()}组】" if lang == "zh" else f"[{group_name.capitalize()} Group]"
        text += f"\n{group_title}\n"
        for i, item in enumerate(candidates.get(group_name, [])):
            text += f"{i+1}. id={item['id']}, \"{item.get('text', '')}\"\n"
    return text


def _resolve_video_patch_token(face_or_frame: str = "face") -> str:
    return "<FaceHere>" if "face" in face_or_frame else "<FrameHere>"


def build_s_messages(
    subtitle: str,
    candidates: Dict[str, list],
    lang: str = "zh",
    face_or_frame: str = "face",
) -> List[Dict]:
    candidates_text = _format_candidates_text(candidates, lang)
    video_patch_token = _resolve_video_patch_token(face_or_frame)
    if lang == "zh":
        return [
            {"role": "system", "content": S_SYSTEM_ZH},
            {"role": "user", "content": (
                f"当前样本字幕: \"{subtitle}\"\n"
                f"<Video>{video_patch_token}</Video>\n"
                f"<Audio><AudioHere></Audio>\n\n"
                f"候选证据列表:{candidates_text}\n\n"
                f"请从每组中各选一条，输出JSON：\n"
                f'{{"primary": {{"id": "..."}}, "confusion": {{"id": "..."}}, "counter": {{"id": "..."}}}}'  
            )},
        ]
    else:
        return [
            {"role": "system", "content": S_SYSTEM_EN},
            {"role": "user", "content": (
                f"Current Sample Subtitle: \"{subtitle}\"\n"
                f"<Video>{video_patch_token}</Video>\n"
                f"<Audio><AudioHere></Audio>\n\n"
                f"Candidate Evidence List:{candidates_text}\n\n"
                f"Select one from each group. Output JSON:\n"
                f'{{"primary": {{"id": "..."}}, "confusion": {{"id": "..."}}, "counter": {{"id": "..."}}}}'  
            )},
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# Agent-G Prompts
# ═══════════════════════════════════════════════════════════════════════════════

G_SYSTEM_ZH = (
    "你是一个多模态情感分析器。你将看到一个待分析样本（含融合增强后的视听特征）"
    "和几个参考证据。每个证据标注了检索意图。请综合判断情感并输出JSON。"
)

G_SYSTEM_EN = (
    "You are a multimodal emotion analyzer. You will see a sample to be analyzed "
    "(containing fused enhanced audiovisual features) and several reference evidence items. "
    "Each evidence has a retrieval intent annotated. Please comprehensively judge the emotion "
    "and output JSON."
)


# V3: 每条证据需包含 角色标签 + 检索意图
EVIDENCE_INTENT_ZH = {
    "support": "验证主假设",
    "contrast": "区分易混淆情感",
    "counter": "寻找反证检验主假设是否成立",
}
EVIDENCE_INTENT_EN = {
    "support": "verify main hypothesis",
    "contrast": "distinguish easily confused emotions",
    "counter": "find counter-evidence to test the main hypothesis",
}


def _format_evidence_text(evidence_items: Dict[str, object], lang: str = "zh") -> str:
    text = ""
    role_map_zh = {"support": "支持", "contrast": "对比", "counter": "反面"}
    intent_map = EVIDENCE_INTENT_ZH if lang == "zh" else EVIDENCE_INTENT_EN
    for role, ev in evidence_items.items():
        if ev is None:
            continue
        label_hint = ev.get("label_hint", "") if isinstance(ev, dict) else getattr(ev, "label_hint", "")
        ev_text = ev.get("text", "") if isinstance(ev, dict) else getattr(ev, "text", "")
        intent = intent_map.get(role, "")
        if lang == "zh":
            role_name = role_map_zh.get(role, role)
            header = f"[{role_name}证据 — {role}, {label_hint}]" if label_hint else f"[{role_name}证据 — {role}]"
            text += f"\n{header}\n检索意图: {intent}\n\"{ev_text}\"\n"
        else:
            header = f"[{role.capitalize()} Evidence — {role}, {label_hint}]" if label_hint else f"[{role.capitalize()} Evidence — {role}]"
            text += f"\n{header}\nRetrieval Intent: {intent}\n\"{ev_text}\"\n"
    return text


def build_g_messages(
    subtitle: str,
    evidence_items: Dict[str, object],
    candidate_labels: List[str],
    lang: str = "zh",
    face_or_frame: str = "face",
) -> List[Dict]:
    evidence_text = _format_evidence_text(evidence_items, lang)
    labels_str = ", ".join(candidate_labels)
    video_patch_token = _resolve_video_patch_token(face_or_frame)

    if lang == "zh":
        return [
            {"role": "system", "content": G_SYSTEM_ZH},
            {"role": "user", "content": (
                f"{evidence_text}\n"
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
                f"请输出JSON，包含 prediction, confidence, reasoning 三个字段。"
            )},
        ]
    else:
        return [
            {"role": "system", "content": G_SYSTEM_EN},
            {"role": "user", "content": (
                f"{evidence_text}\n"
                f"[Current Sample]\n"
                f"\"{subtitle}\"\n"
                f"Visual: <Video>{video_patch_token}</Video>\n"
                f"Audio: <Audio><AudioHere></Audio>\n"
                f"(Audiovisual features have been fused and enhanced with the perceptually most similar sample)\n"
                f"Candidate Label Pool (reference only): {labels_str}\n\n"
                f"Please comprehensively judge the emotion. You may prioritize the candidate label pool, or output the closest open-vocabulary emotion label if needed:\n"
                f"- Support evidence describes a similar emotional pattern, used for reference.\n"
                f"- Contrast evidence is used to distinguish easily confused emotions.\n"
                f"- Counter evidence is used to test whether the main hypothesis holds.\n\n"
                f"Output JSON with fields: prediction, confidence, reasoning."
            )},
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# 输出解析
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_q_output(raw_text: str) -> QueryOutput:
    out = QueryOutput(raw_text=raw_text)
    try:
        data = json.loads(_clean_json_text(raw_text))
        out.primary = data.get("primary", {})
        out.confusion = data.get("confusion", {})
        out.counter = data.get("counter", {})
        if not out.primary.get("query_text") or not out.confusion.get("query_text") or not out.counter.get("query_text"):
            out.valid = False
    except (json.JSONDecodeError, AttributeError):
        out.valid = False
    return out


def parse_s_output(raw_text: str, candidates: Dict[str, list]) -> SelectorOutput:
    out = SelectorOutput(raw_text=raw_text)
    try:
        data = json.loads(_clean_json_text(raw_text))
        for group in ["primary", "confusion", "counter"]:
            role_info = data.get(group, {})
            selected_id = role_info.get("id", "") if isinstance(role_info, dict) else str(role_info) if role_info else ""
            setattr(out, f"{group}_id", selected_id)

        valid_ids_by_group = {}
        for group in ["primary", "confusion", "counter"]:
            valid_ids_by_group[group] = {
                str(item.get("id", ""))
                for item in candidates.get(group, [])
                if item.get("id", "")
            }

        for group in ["primary", "confusion", "counter"]:
            sid = getattr(out, f"{group}_id", "")
            if not sid or sid not in valid_ids_by_group[group]:
                out.valid = False
    except (json.JSONDecodeError, AttributeError):
        out.valid = False
    return out


def _fuzzy_match_label(text: str, candidate_labels: List[str]) -> Optional[str]:
    """在文本中模糊匹配候选标签，返回最佳匹配或 None。"""
    if not text or not candidate_labels:
        return None
    text_lower = text.lower()
    _NEGATION_PREFIXES = ("not ", "no ", "non-", "非", "不", "没有", "无")
    best_label = None
    best_score = 0.0
    for label in candidate_labels:
        label_lower = str(label).strip().lower()
        if not label_lower:
            continue
        if text_lower == label_lower:
            return label
        if label_lower in text_lower or text_lower in label_lower:
            if label_lower in text_lower:
                idx = text_lower.index(label_lower)
                prefix = text_lower[:idx]
                if any(prefix.endswith(neg) for neg in _NEGATION_PREFIXES):
                    continue
            score = min(len(text_lower), len(label_lower)) / max(len(text_lower), len(label_lower), 1)
            if score > best_score:
                best_score = score
                best_label = label
    return best_label if best_score >= 0.3 else None


def parse_g_output(raw_text: str, candidate_labels: List[str]) -> GeneratorOutput:
    out = GeneratorOutput(raw_text=raw_text)

    # ── 尝试 1：JSON 解析 ──
    try:
        data = json.loads(_clean_json_text(raw_text))
        prediction = data.get("prediction", "")
        if isinstance(prediction, list):
            prediction = ", ".join(str(item).strip() for item in prediction if str(item).strip())
        elif isinstance(prediction, dict):
            prediction = prediction.get("label", "") or prediction.get("text", "") or ""
        out.prediction = str(prediction).strip()
        out.confidence = float(data.get("confidence", 0.0))
        out.reasoning = str(data.get("reasoning", ""))
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        pass

    # ── 尝试 2：对 prediction 做候选标签模糊匹配 ──
    if out.prediction:
        matched = _fuzzy_match_label(out.prediction, candidate_labels)
        if matched:
            out.prediction = matched

    # ── 尝试 3（fallback）：JSON 解析失败或 prediction 为空时，从原始文本中扫描候选标签 ──
    if not out.prediction and candidate_labels:
        matched = _fuzzy_match_label(raw_text, candidate_labels)
        if matched:
            out.prediction = matched
            out.reasoning = out.reasoning or "(extracted from raw text)"

    # ── 校验 ──
    if not out.prediction:
        out.valid = False
    if not (0.0 <= out.confidence <= 1.0):
        out.confidence = 0.5
    return out
