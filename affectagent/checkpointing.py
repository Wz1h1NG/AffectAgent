"""Canonical AffectAgent checkpoint names with pre-release fallbacks."""

import os
from typing import Dict, Iterable, Optional


CHECKPOINT_FILES: Dict[str, tuple] = {
    "actor": ("affectgpt_trainable.pth",),
    "raaf": ("raaf.pth", "support_fusion.pth"),
    "mb_moe": ("mb_moe.pth", "modality_moe.pth"),
    "critic": ("critic.pth", "value_head.pth"),
}


def checkpoint_candidates(component: str) -> Iterable[str]:
    try:
        return CHECKPOINT_FILES[component]
    except KeyError as error:
        raise ValueError(f"Unknown checkpoint component: {component}") from error


def resolve_checkpoint_path(directory: str, component: str) -> Optional[str]:
    """Prefer the paper-named file and fall back to old release names."""

    for filename in checkpoint_candidates(component):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            return path
    return None
