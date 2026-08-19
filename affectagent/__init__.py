"""Official implementation of AffectAgent with lazy public imports."""

from importlib import import_module


_EXPORTS = {
    "AffectAgentPipeline": ("affectagent.orchestrator", "AffectAgentPipeline"),
    "AffectiveRewardComputer": ("affectagent.reward", "AffectiveRewardComputer"),
    "DualChannelRetriever": ("affectagent.retriever_service", "DualChannelRetriever"),
    "EmotionGeneratorOutput": ("affectagent.schemas", "EmotionGeneratorOutput"),
    "EvidenceFilterOutput": ("affectagent.schemas", "EvidenceFilterOutput"),
    "MBMoE": ("affectagent.fusion_modules", "MBMoE"),
    "ModalityBalancingMoE": ("affectagent.fusion_modules", "ModalityBalancingMoE"),
    "QueryPlannerOutput": ("affectagent.schemas", "QueryPlannerOutput"),
    "RAAF": ("affectagent.fusion_modules", "RAAF"),
    "RetrievalAugmentedAdaptiveFusion": (
        "affectagent.fusion_modules",
        "RetrievalAugmentedAdaptiveFusion",
    ),
    "RolloutResult": ("affectagent.schemas", "RolloutResult"),
    "RolloutSample": ("affectagent.schemas", "RolloutSample"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
