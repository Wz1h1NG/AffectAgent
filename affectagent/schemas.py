"""Structured data exchanged by the AffectAgent pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class RolloutSample:
    """One multimodal sample processed by Query Planner -> Filter -> Generator."""

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


@dataclass(init=False)
class QueryPlannerOutput:
    """The paper-defined support, confusion, and counter queries."""

    support: Dict[str, str]
    confusion: Dict[str, str]
    counter: Dict[str, str]
    raw_text: str
    valid: bool

    def __init__(
        self,
        support: Optional[Dict[str, str]] = None,
        confusion: Optional[Dict[str, str]] = None,
        counter: Optional[Dict[str, str]] = None,
        raw_text: str = "",
        valid: bool = True,
        primary: Optional[Dict[str, str]] = None,
    ):
        self.support = dict(support if support is not None else (primary or {}))
        self.confusion = dict(confusion or {})
        self.counter = dict(counter or {})
        self.raw_text = raw_text
        self.valid = valid

    @property
    def primary(self) -> Dict[str, str]:
        """Deprecated compatibility alias for :attr:`support`."""

        return self.support

    @primary.setter
    def primary(self, value: Dict[str, str]) -> None:
        self.support = value


QueryOutput = QueryPlannerOutput


@dataclass(init=False)
class EvidenceFilterOutput:
    """Identifiers retained by the Evidence Filter for each query group."""

    support_id: str
    confusion_id: str
    counter_id: str
    raw_text: str
    valid: bool

    def __init__(
        self,
        support_id: str = "",
        confusion_id: str = "",
        counter_id: str = "",
        raw_text: str = "",
        valid: bool = True,
        primary_id: str = "",
    ):
        self.support_id = support_id or primary_id
        self.confusion_id = confusion_id
        self.counter_id = counter_id
        self.raw_text = raw_text
        self.valid = valid

    @property
    def primary_id(self) -> str:
        """Deprecated compatibility alias for :attr:`support_id`."""

        return self.support_id

    @primary_id.setter
    def primary_id(self, value: str) -> None:
        self.support_id = value


SelectorOutput = EvidenceFilterOutput


@dataclass
class EmotionGeneratorOutput:
    prediction: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    raw_text: str = ""
    valid: bool = True


GeneratorOutput = EmotionGeneratorOutput


@dataclass
class EvidenceItem:
    evidence_id: str = ""
    global_id: Optional[int] = None
    text: str = ""
    role: str = ""  # support / confusion / counter
    subquery_type: str = ""
    label_hint: str = ""


EvidenceValue = Optional[Union[EvidenceItem, List[EvidenceItem]]]


@dataclass
class FusionDiagnostics:
    """Diagnostics emitted by RAAF and MB-MoE."""

    fusion_partner_id: str = ""
    fusion_partner_label: str = ""
    fusion_partner_discrete_label: str = ""
    fusion_partner_source: str = "perceptual_retrieval"
    video_gate_mean: float = 0.0
    audio_gate_mean: float = 0.0
    moe_experts_activated: List[int] = field(default_factory=list)
    moe_expert_weights: List[float] = field(default_factory=list)


@dataclass
class RewardBreakdown:
    """Paper equations (1)-(4) evaluated for one collaborative rollout."""

    score_full: float = 0.0
    score_label: float = 0.0
    score_rank: float = 0.0
    r_shared: float = 0.0
    r_planner: float = 0.0
    r_filter: float = 0.0
    r_generator: float = 0.0
    lambda_planner: float = 1.0
    lambda_filter: float = 1.0
    counterfactuals_complete: bool = False

    @property
    def credit_q(self) -> float:
        return self.r_planner

    @property
    def credit_s(self) -> float:
        return self.r_filter

    @property
    def credit_f(self) -> float:
        return self.r_filter

    @property
    def credit_g(self) -> float:
        return self.r_generator

    @property
    def r_total(self) -> float:
        return self.r_shared

    @property
    def r_task(self) -> float:
        return self.score_full


@dataclass
class RolloutResult:
    """Full and counterfactual outputs required by the paper reward."""

    sample: Optional[RolloutSample] = None
    query_output: Optional[QueryPlannerOutput] = None
    filter_output: Optional[EvidenceFilterOutput] = None
    generator_output: Optional[EmotionGeneratorOutput] = None
    label_baseline_output: Optional[EmotionGeneratorOutput] = None
    rank_baseline_output: Optional[EmotionGeneratorOutput] = None
    candidates: Dict[str, list] = field(default_factory=dict)
    selected_evidence: Dict[str, EvidenceValue] = field(default_factory=dict)
    fusion_diagnostics: Optional[FusionDiagnostics] = None
    q_query_ids: Optional[object] = None
    q_response_ids: Optional[object] = None
    f_query_ids: Optional[object] = None
    f_response_ids: Optional[object] = None
    g_query_ids: Optional[object] = None
    g_response_ids: Optional[object] = None
    runtime_context: Dict[str, object] = field(default_factory=dict)
    rewards: Optional[RewardBreakdown] = None

    @property
    def selector_output(self) -> Optional[EvidenceFilterOutput]:
        return self.filter_output

    @selector_output.setter
    def selector_output(self, value: Optional[EvidenceFilterOutput]) -> None:
        self.filter_output = value

    @property
    def s_query_ids(self):
        return self.f_query_ids

    @s_query_ids.setter
    def s_query_ids(self, value) -> None:
        self.f_query_ids = value

    @property
    def s_response_ids(self):
        return self.f_response_ids

    @s_response_ids.setter
    def s_response_ids(self, value) -> None:
        self.f_response_ids = value
