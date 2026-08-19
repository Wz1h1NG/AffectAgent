"""Role-specific multimodal prompts and robust structured-output parsers."""

import json
from typing import Dict, Iterable, List, Optional

from .schemas import (
    EmotionGeneratorOutput,
    EvidenceFilterOutput,
    QueryPlannerOutput,
)


GROUPS = ("support", "confusion", "counter")


def _resolve_video_patch_token(face_or_frame: str = "face") -> str:
    return "<FaceHere>" if "face" in face_or_frame else "<FrameHere>"


QUERY_PLANNER_SYSTEM_ZH = (
    "你是AffectAgent的多模态查询规划器。请同时观察字幕、视频和音频，并结合候选标签，"
    "生成支持、易混淆和反证三类认知检索查询。查询应描述可检索的语言、表情、动作、语气和"
    "情感线索，不能只输出抽象意图。严格输出JSON。"
)
QUERY_PLANNER_SYSTEM_EN = (
    "You are AffectAgent's multimodal Query Planner. Observe the subtitle, video, and audio "
    "together with the candidate labels, then formulate support, confusion, and counter "
    "cognitive retrieval queries. Describe retrievable linguistic, facial, behavioral, vocal, "
    "and affective cues rather than an abstract intent. Output JSON only."
)
QUERY_OUTPUT_SCHEMA = """{
  "support":   {"query_text": "...", "target_label": "..."},
  "confusion": {"query_text": "...", "confusing_label": "..."},
  "counter":   {"query_text": "...", "opposite_label": "..."}
}"""


def build_query_planner_messages(
    subtitle: str,
    candidate_labels: List[str],
    lang: str = "zh",
    face_or_frame: str = "face",
) -> List[Dict]:
    labels = ", ".join(candidate_labels)
    video_token = _resolve_video_patch_token(face_or_frame)
    if lang == "zh":
        content = (
            f'字幕: "{subtitle}"\n'
            f"视觉: <Video>{video_token}</Video>\n"
            f"听觉: <Audio><AudioHere></Audio>\n"
            f"候选标签: {labels}\n\n请输出如下JSON:\n{QUERY_OUTPUT_SCHEMA}"
        )
        system = QUERY_PLANNER_SYSTEM_ZH
    else:
        content = (
            f'Subtitle: "{subtitle}"\n'
            f"Visual: <Video>{video_token}</Video>\n"
            f"Audio: <Audio><AudioHere></Audio>\n"
            f"Candidate labels: {labels}\n\nOutput this JSON schema:\n{QUERY_OUTPUT_SCHEMA}"
        )
        system = QUERY_PLANNER_SYSTEM_EN
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def build_q_messages(
    subtitle: str,
    candidate_labels: List[str],
    lang: str = "zh",
    face_or_frame: str = "face",
) -> List[Dict]:
    """Deprecated function name retained for existing integrations."""

    return build_query_planner_messages(subtitle, candidate_labels, lang, face_or_frame)


EVIDENCE_FILTER_SYSTEM_ZH = (
    "你是AffectAgent的多模态证据过滤器。请用当前样本的字幕、视频和音频交叉核验三组认知证据，"
    "并从support、confusion、counter各保留一条最可靠证据。严格输出证据ID的JSON。"
)
EVIDENCE_FILTER_SYSTEM_EN = (
    "You are AffectAgent's multimodal Evidence Filter. Cross-verify each cognitive candidate "
    "against the current subtitle, video, and audio, retaining the most reliable item from each "
    "support, confusion, and counter group. Output evidence IDs as JSON only."
)


def _canonical_candidates(candidates: Dict[str, list]) -> Dict[str, list]:
    return {
        "support": candidates.get("support", candidates.get("primary", [])),
        "confusion": candidates.get("confusion", candidates.get("contrast", [])),
        "counter": candidates.get("counter", []),
    }


def _format_candidates_text(candidates: Dict[str, list], lang: str) -> str:
    chunks = []
    for group, items in _canonical_candidates(candidates).items():
        title = f"【{group}组】" if lang == "zh" else f"[{group.capitalize()} group]"
        rows = [title]
        for index, item in enumerate(items, 1):
            rows.append(f'{index}. id={item.get("id", "")}, "{item.get("text", "")}"')
        chunks.append("\n".join(rows))
    return "\n".join(chunks)


def build_evidence_filter_messages(
    subtitle: str,
    candidates: Dict[str, list],
    lang: str = "zh",
    face_or_frame: str = "face",
) -> List[Dict]:
    video_token = _resolve_video_patch_token(face_or_frame)
    candidates_text = _format_candidates_text(candidates, lang)
    schema = '{"support": {"id": "..."}, "confusion": {"id": "..."}, "counter": {"id": "..."}}'
    if lang == "zh":
        content = (
            f'当前样本字幕: "{subtitle}"\n'
            f"<Video>{video_token}</Video>\n<Audio><AudioHere></Audio>\n\n"
            f"候选认知证据:\n{candidates_text}\n\n每组保留一条，输出JSON:\n{schema}"
        )
        system = EVIDENCE_FILTER_SYSTEM_ZH
    else:
        content = (
            f'Current subtitle: "{subtitle}"\n'
            f"<Video>{video_token}</Video>\n<Audio><AudioHere></Audio>\n\n"
            f"Cognitive candidates:\n{candidates_text}\n\nRetain one per group and output JSON:\n{schema}"
        )
        system = EVIDENCE_FILTER_SYSTEM_EN
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


build_s_messages = build_evidence_filter_messages


EMOTION_GENERATOR_SYSTEM_ZH = (
    "你是AffectAgent的多模态情感生成器。结合原始文本、过滤后的认知证据以及经RAAF和MB-MoE"
    "增强的视听表示，输出最终情感标签和解释。严格输出JSON。"
)
EMOTION_GENERATOR_SYSTEM_EN = (
    "You are AffectAgent's multimodal Emotion Generator. Combine the raw text, filtered cognitive "
    "evidence, and the audiovisual representation enhanced by RAAF and MB-MoE to output the final "
    "emotion label and rationale. Output JSON only."
)


def _iter_evidence(value: object) -> Iterable[object]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def _item_value(item: object, key: str, default: str = "") -> str:
    if isinstance(item, dict):
        return str(item.get(key, default))
    return str(getattr(item, key, default))


def _format_evidence_text(evidence_items: Dict[str, object], lang: str) -> str:
    rows = []
    for role in GROUPS:
        values = evidence_items.get(role)
        # Accept the old implementation term ``contrast`` during migration.
        if role == "confusion" and values is None:
            values = evidence_items.get("contrast")
        for index, item in enumerate(_iter_evidence(values), 1):
            text = _item_value(item, "text")
            hint = _item_value(item, "label_hint")
            if lang == "zh":
                rows.append(f'[{role}证据 {index}{", " + hint if hint else ""}]\n"{text}"')
            else:
                rows.append(f'[{role.capitalize()} evidence {index}{", " + hint if hint else ""}]\n"{text}"')
    return "\n".join(rows)


def build_emotion_generator_messages(
    subtitle: str,
    evidence_items: Dict[str, object],
    candidate_labels: List[str],
    lang: str = "zh",
    face_or_frame: str = "face",
) -> List[Dict]:
    evidence = _format_evidence_text(evidence_items, lang)
    labels = ", ".join(candidate_labels)
    video_token = _resolve_video_patch_token(face_or_frame)
    if lang == "zh":
        content = (
            f"{evidence}\n[待分析样本]\n字幕: \"{subtitle}\"\n"
            f"视觉: <Video>{video_token}</Video>\n听觉: <Audio><AudioHere></Audio>\n"
            f"候选标签: {labels}\n\n输出包含prediction、confidence、reasoning的JSON。"
        )
        system = EMOTION_GENERATOR_SYSTEM_ZH
    else:
        content = (
            f"{evidence}\n[Current sample]\nSubtitle: \"{subtitle}\"\n"
            f"Visual: <Video>{video_token}</Video>\nAudio: <Audio><AudioHere></Audio>\n"
            f"Candidate labels: {labels}\n\nOutput JSON with prediction, confidence, and reasoning."
        )
        system = EMOTION_GENERATOR_SYSTEM_EN
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


build_g_messages = build_emotion_generator_messages


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def parse_query_planner_output(raw_text: str) -> QueryPlannerOutput:
    output = QueryPlannerOutput(raw_text=raw_text)
    try:
        data = json.loads(_clean_json_text(raw_text))
        output.support = data.get("support", data.get("primary", {}))
        output.confusion = data.get("confusion", data.get("contrast", {}))
        output.counter = data.get("counter", {})
        # Normalize pre-release field names while keeping old SFT outputs loadable.
        if "confusing_label" not in output.confusion and "contrast_label" in output.confusion:
            output.confusion["confusing_label"] = output.confusion["contrast_label"]
        if "opposite_label" not in output.counter and "counter_direction" in output.counter:
            output.counter["opposite_label"] = output.counter["counter_direction"]
        output.valid = all(
            isinstance(getattr(output, group), dict)
            and bool(getattr(output, group).get("query_text"))
            for group in GROUPS
        )
    except (json.JSONDecodeError, AttributeError, TypeError):
        output.valid = False
    return output


parse_q_output = parse_query_planner_output


def parse_evidence_filter_output(raw_text: str, candidates: Dict[str, list]) -> EvidenceFilterOutput:
    output = EvidenceFilterOutput(raw_text=raw_text)
    canonical = _canonical_candidates(candidates)
    try:
        data = json.loads(_clean_json_text(raw_text))
        for group in GROUPS:
            source = data.get(group, data.get("primary", {}) if group == "support" else {})
            selected_id = source.get("id", "") if isinstance(source, dict) else str(source or "")
            setattr(output, f"{group}_id", selected_id)
        valid_ids = {
            group: {str(item.get("id", "")) for item in canonical[group] if item.get("id")}
            for group in GROUPS
        }
        output.valid = all(
            bool(getattr(output, f"{group}_id"))
            and getattr(output, f"{group}_id") in valid_ids[group]
            for group in GROUPS
        )
    except (json.JSONDecodeError, AttributeError, TypeError):
        output.valid = False
    return output


parse_s_output = parse_evidence_filter_output


def _fuzzy_match_label(text: str, candidate_labels: List[str]) -> Optional[str]:
    if not text or not candidate_labels:
        return None
    normalized = text.strip().lower()
    best = None
    best_score = 0.0
    for label in candidate_labels:
        label_text = str(label).strip().lower()
        if not label_text:
            continue
        if normalized == label_text:
            return str(label)
        if label_text in normalized or normalized in label_text:
            score = min(len(normalized), len(label_text)) / max(len(normalized), len(label_text), 1)
            if score > best_score:
                best, best_score = str(label), score
    return best if best_score >= 0.3 else None


def parse_emotion_generator_output(raw_text: str, candidate_labels: List[str]) -> EmotionGeneratorOutput:
    output = EmotionGeneratorOutput(raw_text=raw_text)
    try:
        data = json.loads(_clean_json_text(raw_text))
        prediction = data.get("prediction", "")
        if isinstance(prediction, list):
            prediction = ", ".join(str(value).strip() for value in prediction if str(value).strip())
        elif isinstance(prediction, dict):
            prediction = prediction.get("label", prediction.get("text", ""))
        output.prediction = str(prediction).strip()
        output.confidence = float(data.get("confidence", 0.0))
        output.reasoning = str(data.get("reasoning", ""))
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        pass
    if output.prediction:
        output.prediction = _fuzzy_match_label(output.prediction, candidate_labels) or output.prediction
    elif candidate_labels:
        output.prediction = _fuzzy_match_label(raw_text, candidate_labels) or ""
    output.valid = bool(output.prediction)
    if not 0.0 <= output.confidence <= 1.0:
        output.confidence = 0.5
    return output


parse_g_output = parse_emotion_generator_output
