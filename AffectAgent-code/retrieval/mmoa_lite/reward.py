"""
reward.py — MMOA-Lite V3 情感分析 RL 奖励函数
实现 V3 方案的 7 维奖励分解 + 模块级 credit assignment：
  子奖励: r_task, r_type, r_counter, r_faith, r_mod, r_format, r_uar
  硬惩罚: p_hard (空选/解析失败/缺反证)
  模块归因:
    R_Q = 0.55*r_task + 0.35*r_type + 0.10*r_uar
    R_S = 0.70*r_task + 0.20*r_counter + 0.10*r_uar
    R_G = 0.35*r_task + 0.25*r_faith + 0.15*r_format + 0.15*r_mod + 0.10*r_uar
"""

import re
import torch
from typing import List, Dict, Optional
try:
    from .schemas import (
        RolloutResult, RewardBreakdown, FusionDiagnostics,
        QueryOutput, SelectorOutput, GeneratorOutput,
    )
except ImportError:
    from retrieval.mmoa_lite.schemas import (
        RolloutResult, RewardBreakdown, FusionDiagnostics,
        QueryOutput, SelectorOutput, GeneratorOutput,
    )


class EmotionRewardComputer:
    """V3 方案情感分析 RL 奖励计算器。"""

    # V3 总奖励权重
    W_TASK = 0.35
    W_TYPE = 0.15
    W_COUNTER = 0.10
    W_FAITH = 0.10
    W_MOD = 0.10
    W_FORMAT = 0.10
    W_UAR = 0.10

    def __init__(self, running_uar: Optional[Dict[str, float]] = None):
        """
        Args:
            running_uar: 每个类别的 running recall 字典 (可选，训练过程中动态更新)
        """
        self._class_correct = {}
        self._class_total = {}
        self._running_uar = running_uar or {}

    def reset_uar_stats(self):
        """每个 epoch 开始时重置 UAR 统计，避免历史数据稀释当前 epoch 的类别均衡信号。"""
        self._class_correct = {}
        self._class_total = {}

    @staticmethod
    def _normalize_label_text(text: str) -> str:
        if text is None:
            return ""
        normalized = str(text).strip().lower()
        normalized = re.sub(r"[\[\]\{\}\"'`]+", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip(" ,.;:，；、/|")

    @classmethod
    def _split_labels(cls, text: str) -> List[str]:
        normalized = cls._normalize_label_text(text)
        if not normalized:
            return []
        parts = re.split(r"[,，;/；、|]+", normalized)
        labels = [cls._normalize_label_text(part) for part in parts]
        labels = [label for label in labels if label]
        return labels if labels else [normalized]

    @classmethod
    def _tokenize_label(cls, label: str) -> List[str]:
        normalized = cls._normalize_label_text(label)
        if not normalized:
            return []
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized)
        return tokens if tokens else [normalized]

    @classmethod
    def _label_similarity(cls, lhs: str, rhs: str) -> float:
        left = cls._normalize_label_text(lhs)
        right = cls._normalize_label_text(rhs)
        if not left or not right:
            return 0.0
        if left == right or left in right or right in left:
            return 1.0
        left_tokens = set(cls._tokenize_label(left))
        right_tokens = set(cls._tokenize_label(right))
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens & right_tokens)
        if overlap == 0:
            return 0.0
        precision = overlap / len(left_tokens)
        recall = overlap / len(right_tokens)
        return 2 * precision * recall / (precision + recall)

    @classmethod
    def _best_label_overlap(cls, prediction: str, reference: str) -> float:
        pred_labels = cls._split_labels(prediction)
        ref_labels = cls._split_labels(reference)
        if not pred_labels or not ref_labels:
            return 0.0
        return max(
            cls._label_similarity(pred_label, ref_label)
            for pred_label in pred_labels
            for ref_label in ref_labels
        )

    @classmethod
    def _canonical_label(cls, text: str) -> str:
        labels = cls._split_labels(text)
        return labels[0] if labels else ""

    # ═══════════════════════════════════════════════════════════
    # 1. r_task — 主任务奖励
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def compute_r_task(prediction: str, ground_truth: str) -> float:
        if not prediction or not ground_truth:
            return 0.0
        return EmotionRewardComputer._best_label_overlap(prediction, ground_truth)

    # ═══════════════════════════════════════════════════════════
    # 2. r_type — Q 的检索类型纯度
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def compute_r_type(
        q_output: QueryOutput,
        candidates: Dict[str, list],
    ) -> float:
        """
        衡量通道A的检索结果是否与子查询类型对齐。
        - primary 候选中 target_label 占比
        - confusion 候选中双侧标签覆盖
        - counter 候选中反向标签占比
        """
        if not q_output.valid:
            return 0.0

        scores = []
        # primary: target_label 出现的比例
        target = q_output.primary.get("target_label", "")
        primary_cands = candidates.get("primary", [])
        if target and primary_cands:
            overlaps = [
                EmotionRewardComputer._best_label_overlap(target, c.get("label_hint", ""))
                for c in primary_cands
            ]
            scores.append(sum(overlaps) / len(overlaps))
        else:
            scores.append(0.0)

        # confusion: 对比双侧标签是否都出现
        contrast_label = q_output.confusion.get("contrast_label", "")
        confusion_cands = candidates.get("confusion", [])
        if contrast_label and confusion_cands:
            target_present = any(
                EmotionRewardComputer._best_label_overlap(target, c.get("label_hint", "")) >= 0.5
                for c in confusion_cands
            )
            contrast_present = any(
                EmotionRewardComputer._best_label_overlap(contrast_label, c.get("label_hint", "")) >= 0.5
                for c in confusion_cands
            )
            scores.append(1.0 if target_present and contrast_present else 0.5 if target_present or contrast_present else 0.0)
        else:
            scores.append(0.0)

        # counter: 反向标签占比
        counter_dir = q_output.counter.get("counter_direction", "")
        counter_cands = candidates.get("counter", [])
        if counter_dir and counter_cands:
            overlaps = [
                EmotionRewardComputer._best_label_overlap(counter_dir, c.get("label_hint", ""))
                for c in counter_cands
            ]
            scores.append(sum(overlaps) / len(overlaps))
        else:
            scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    # ═══════════════════════════════════════════════════════════
    # 3. r_counter — S 的反证覆盖质量
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def compute_r_counter(
        s_output: SelectorOutput,
        candidates: Dict[str, list],
    ) -> float:
        """
        检查 S 是否有效选出了反证（counter 组证据）。
        """
        if not s_output.valid:
            return 0.0

        counter_id = s_output.counter_id
        if not counter_id:
            return 0.0

        counter_cands = candidates.get("counter", [])
        for c in counter_cands:
            if c.get("id") == counter_id:
                return 1.0
        return 0.0

    # ═══════════════════════════════════════════════════════════
    # 4. r_faith — G 的推理忠实度
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def compute_r_faith(
        g_output: GeneratorOutput,
        selected_evidence: Dict[str, Optional[object]],
    ) -> float:
        """
        检查 G 的 reasoning 是否引用了各类证据。
        结合角色关键词提及和证据文本内容重叠两个维度评分：
          • 关键词 + 内容重叠 → 1/3
          • 仅关键词 或 仅内容重叠 → 0.5/3
        """
        if not g_output.valid or not g_output.reasoning:
            return 0.0

        reasoning_lower = g_output.reasoning.lower()
        role_keywords = {
            "support": ["support", "支持", "主假设", "一致", "similar", "相似"],
            "contrast": ["contrast", "对比", "区分", "混淆", "confus", "differ"],
            "counter": ["counter", "反面", "反证", "反面证据", "refut", "否定"],
        }

        score = 0.0
        for role, keywords in role_keywords.items():
            ev = selected_evidence.get(role)
            if ev is None:
                continue
            ev_text = ev.get("text", "") if isinstance(ev, dict) else getattr(ev, "text", "")

            # (a) 角色关键词提及
            keyword_hit = any(kw in reasoning_lower for kw in keywords)

            # (b) 证据文本内容重叠 (token-level overlap ≥ 15% 或子串命中)
            content_hit = False
            if ev_text:
                ev_toks = set(re.findall(r'[\w\u4e00-\u9fff]+', ev_text.lower()))
                reas_toks = set(re.findall(r'[\w\u4e00-\u9fff]+', reasoning_lower))
                if ev_toks:
                    overlap_ratio = len(ev_toks & reas_toks) / len(ev_toks)
                    content_hit = overlap_ratio >= 0.15
                if not content_hit:
                    # 子串 fallback：尝试多个片段
                    for start in range(0, min(len(ev_text), 60), 20):
                        snippet = ev_text[start:start + 20].lower().strip()
                        if snippet and snippet in reasoning_lower:
                            content_hit = True
                            break

            if keyword_hit and content_hit:
                score += 1.0 / 3.0
            elif keyword_hit or content_hit:
                score += 0.5 / 3.0

        return min(score, 1.0)

    # ═══════════════════════════════════════════════════════════
    # 5. r_mod — 多模态特征使用度
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def compute_r_mod(fusion_diag: Optional[FusionDiagnostics]) -> float:
        """
        基于 SupportFusion 门控诊断信息，衡量多模态特征是否被有效使用。
        """
        if fusion_diag is None:
            return 0.0

        score = 0.0
        # 视频门控均值 > 0.1 表示视频通道活跃
        if fusion_diag.video_gate_mean > 0.1:
            score += 0.5
        # 音频门控均值 > 0.1 表示音频通道活跃
        if fusion_diag.audio_gate_mean > 0.1:
            score += 0.5

        return score

    # ═══════════════════════════════════════════════════════════
    # 6. r_format — 格式合法性
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def compute_r_format(
        q_output: QueryOutput,
        s_output: SelectorOutput,
        g_output: GeneratorOutput,
    ) -> float:
        """三个 Agent 的 JSON 输出是否均合法，每个 1/3。"""
        score = 0.0
        if q_output.valid:
            score += 1.0 / 3.0
        if s_output.valid:
            score += 1.0 / 3.0
        if g_output.valid:
            score += 1.0 / 3.0
        return score

    # ═══════════════════════════════════════════════════════════
    # 7. r_uar — 类别均衡奖励
    # ═══════════════════════════════════════════════════════════

    def compute_r_uar(self, prediction: str, ground_truth: str) -> float:
        """
        基于 running UAR 的类别均衡奖励。
        如果当前类别的 recall 低于平均水平且预测正确，给额外奖励。
        """
        gt = self._canonical_label(ground_truth)
        if not gt:
            return 0.0
        pred_correct = self.compute_r_task(prediction, ground_truth) >= 0.5

        # 更新 running 统计
        self._class_total[gt] = self._class_total.get(gt, 0) + 1
        if pred_correct:
            self._class_correct[gt] = self._class_correct.get(gt, 0) + 1

        # 计算各类 recall
        recalls = {}
        for cls in self._class_total:
            total = self._class_total[cls]
            correct = self._class_correct.get(cls, 0)
            recalls[cls] = correct / total if total > 0 else 0.0

        if not recalls:
            return 0.0

        mean_recall = sum(recalls.values()) / len(recalls)
        current_recall = recalls.get(gt, 0.0)

        # 低 recall 类别的正确预测获得更多奖励
        if pred_correct and current_recall < mean_recall:
            return 1.0
        elif pred_correct:
            return 0.5
        return 0.0

    # ═══════════════════════════════════════════════════════════
    # 8. 硬惩罚
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def compute_p_hard(
        q_output: QueryOutput,
        s_output: SelectorOutput,
        g_output: GeneratorOutput,
    ) -> float:
        """
        硬惩罚项（从总奖励中扣除）。
        """
        penalty = 0.0

        # Q 解析失败
        if not q_output.valid:
            penalty += 0.3

        # S 解析失败或空选（parse_s_output 已保证 valid=True 时所有 ID 非空）
        if not s_output.valid:
            penalty += 0.3

        # G 解析失败
        if not g_output.valid:
            penalty += 0.3

        return min(penalty, 1.0)

    # ═══════════════════════════════════════════════════════════
    # 综合计算
    # ═══════════════════════════════════════════════════════════

    def compute_pipeline_rewards(self, result: RolloutResult) -> RewardBreakdown:
        """
        计算一次完整 rollout 的所有 V3 奖励分量，返回 RewardBreakdown。
        """
        sample = result.sample
        q_out = result.query_output or QueryOutput(valid=False)
        s_out = result.selector_output or SelectorOutput(valid=False)
        g_out = result.generator_output or GeneratorOutput(valid=False)

        # 各维度子奖励
        r_task = self.compute_r_task(g_out.prediction, sample.ground_truth)
        r_type = self.compute_r_type(q_out, result.candidates)
        r_counter = self.compute_r_counter(s_out, result.candidates)
        r_faith = self.compute_r_faith(g_out, result.selected_evidence)
        r_mod = self.compute_r_mod(result.fusion_diagnostics)
        r_format = self.compute_r_format(q_out, s_out, g_out)
        r_uar = self.compute_r_uar(g_out.prediction or "", sample.ground_truth)

        # 硬惩罚
        p_hard = self.compute_p_hard(q_out, s_out, g_out)

        # 总奖励
        raw_total = (
            self.W_TASK * r_task
            + self.W_TYPE * r_type
            + self.W_COUNTER * r_counter
            + self.W_FAITH * r_faith
            + self.W_MOD * r_mod
            + self.W_FORMAT * r_format
            + self.W_UAR * r_uar
            - p_hard
        )
        r_total = max(-1.0, min(1.0, raw_total))

        # 模块 credit assignment
        credit_q = 0.55 * r_task + 0.35 * r_type + 0.10 * r_uar
        credit_s = 0.70 * r_task + 0.20 * r_counter + 0.10 * r_uar
        # 当融合模块未激活时，将 r_mod 权重重分配给 r_task 和 r_faith
        fusion_active = (
            result.fusion_diagnostics is not None
            and result.fusion_diagnostics.fusion_partner_id != ""
        )
        if fusion_active:
            credit_g = 0.35 * r_task + 0.25 * r_faith + 0.15 * r_format + 0.15 * r_mod + 0.10 * r_uar
        else:
            credit_g = 0.425 * r_task + 0.325 * r_faith + 0.15 * r_format + 0.10 * r_uar

        bd = RewardBreakdown(
            r_task=r_task,
            r_type=r_type,
            r_counter=r_counter,
            r_faith=r_faith,
            r_mod=r_mod,
            r_format=r_format,
            r_uar=r_uar,
            p_hard=p_hard,
            r_total=r_total,
            credit_q=credit_q,
            credit_s=credit_s,
            credit_g=credit_g,
        )
        return bd

    def batch_compute_rewards(
        self, results: List[RolloutResult]
    ) -> List[RewardBreakdown]:
        """批量计算奖励。"""
        breakdowns = []
        for result in results:
            bd = self.compute_pipeline_rewards(result)
            result.rewards = bd
            breakdowns.append(bd)
        return breakdowns

    def to_reward_tensors(
        self, breakdowns: List[RewardBreakdown]
    ) -> Dict[str, torch.Tensor]:
        """
        将 RewardBreakdown 列表转换为 PPO step 所需的 reward tensor 列表。
        按模块 credit assignment 分配。
        """
        q_rewards = torch.tensor([b.credit_q for b in breakdowns]).unsqueeze(-1)
        s_rewards = torch.tensor([b.credit_s for b in breakdowns]).unsqueeze(-1)
        g_rewards = torch.tensor([b.credit_g for b in breakdowns]).unsqueeze(-1)
        return {"q": q_rewards, "s": s_rewards, "g": g_rewards}
