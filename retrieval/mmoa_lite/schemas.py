"""
schemas.py — 数据结构定义
用于 MMOA-Lite RL 联合训练流水线中各阶段的输入/输出/奖励的结构化表示。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class RolloutSample:
    """表示一个待处理样本，贯穿 Q→S→G 全流程。"""
    sample_id: str
    subtitle: str
    ground_truth: str
    candidate_labels: List[str]
    mme_4o_emotion: str = ""
    mm_ref: Optional[Dict] = None
    fusion_partner_id: Optional[str] = None
    partner_mm_ref: Optional[Dict] = None
    dataset: str = ""
    lang: str = "zh"


@dataclass
class QueryOutput:
    """Agent-Q 的结构化输出。"""
    primary: Dict[str, str] = field(default_factory=dict)
    confusion: Dict[str, str] = field(default_factory=dict)
    counter: Dict[str, str] = field(default_factory=dict)
    raw_text: str = ""
    valid: bool = True


@dataclass
class SelectorOutput:
    """Agent-S 的结构化输出。"""
    primary_id: str = ""
    confusion_id: str = ""
    counter_id: str = ""
    raw_text: str = ""
    valid: bool = True


@dataclass
class GeneratorOutput:
    """Agent-G 的结构化输出。"""
    prediction: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    raw_text: str = ""
    valid: bool = True


@dataclass
class EvidenceItem:
    """一条被选中的证据条目。"""
    evidence_id: str = ""
    global_id: Optional[int] = None
    text: str = ""
    role: str = ""            # "support" / "contrast" / "counter"
    subquery_type: str = ""   # "primary" / "confusion" / "counter"
    label_hint: str = ""      # 证据来源样本的情感标签提示


@dataclass
class FusionDiagnostics:
    """SupportFusion + MoE 诊断信息。"""
    fusion_partner_id: str = ""
    fusion_partner_label: str = ""
    fusion_partner_discrete_label: str = ""
    fusion_partner_source: str = "channel_B"
    video_gate_mean: float = 0.0
    audio_gate_mean: float = 0.0
    moe_experts_activated: List[int] = field(default_factory=list)
    moe_expert_weights: List[float] = field(default_factory=list)


@dataclass
class RewardBreakdown:
    """V3 方案的 7 维奖励分解 + 模块 credit assignment。"""
    # 子奖励分量
    r_task: float = 0.0       # 主任务：prediction == ground_truth
    r_type: float = 0.0       # Q: 检索结果与子查询类型对齐
    r_counter: float = 0.0    # S: 反证覆盖质量
    r_faith: float = 0.0      # G: reasoning 是否引用了证据
    r_mod: float = 0.0        # G: 多模态特征是否被有效使用
    r_format: float = 0.0     # 所有 Agent: JSON 格式合法性
    r_uar: float = 0.0        # 类别均衡奖励 (基于 running UAR)

    # 硬惩罚
    p_hard: float = 0.0       # 空选/解析失败/无非文本证据等

    # 总奖励 (clip to [-1, 1])
    r_total: float = 0.0

    # 模块 credit assignment
    credit_q: float = 0.0     # R_Q = 0.55*r_task + 0.35*r_type + 0.10*r_uar
    credit_s: float = 0.0     # R_S = 0.70*r_task + 0.20*r_counter + 0.10*r_uar
    credit_g: float = 0.0     # R_G = 0.35*r_task + 0.25*r_faith + 0.15*r_format + 0.15*r_mod + 0.10*r_uar


@dataclass
class RolloutResult:
    """一次完整 rollout 的所有中间和最终结果。"""
    sample: RolloutSample = None

    # Agent 输出
    query_output: QueryOutput = None
    selector_output: SelectorOutput = None
    generator_output: GeneratorOutput = None

    # 检索结果 (Channel A)
    candidates: Dict[str, list] = field(default_factory=dict)
    selected_evidence: Dict[str, Optional[EvidenceItem]] = field(default_factory=dict)

    # Channel B 融合结果
    fusion_diagnostics: Optional[FusionDiagnostics] = None

    # Token 序列 (用于 PPO 更新)
    q_query_ids: Optional[list] = None
    q_response_ids: Optional[list] = None
    s_query_ids: Optional[list] = None
    s_response_ids: Optional[list] = None
    g_query_ids: Optional[list] = None
    g_response_ids: Optional[list] = None

    # 运行时上下文 (用于按真实 rollout 条件重放 S/G 的 logprob)
    runtime_context: Dict[str, object] = field(default_factory=dict)

    # 奖励
    rewards: RewardBreakdown = None
