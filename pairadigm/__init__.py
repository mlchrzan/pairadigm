"""
Pairadigm v1.0 — Concept-Guided Chain-of-Thought (CGCoT) pairwise annotation.

A Python library for systematic evaluation of text items along specific
conceptual dimensions through structured pairwise comparisons powered by LLMs,
validated by human annotations.
"""

__version__ = "1.0.1"
__author__ = "Michael Leon Chrzan"
__license__ = "Apache-2.0"

from .client import LLMClient
from .core import Pairadigm, pair_items, load_pairadigm, build_pairadigm, pair_from_ordinal

__all__ = [
    "Pairadigm",
    "LLMClient",
    "load_pairadigm",
    "build_pairadigm",
    "pair_from_ordinal",
    "pair_items",
]

# RewardModel is an optional heavy dependency; import lazily
def __getattr__(name: str):
    if name == "RewardModel":
        from .model import RewardModel  # noqa: PLC0415
        return RewardModel
    raise AttributeError(f"module 'pairadigm' has no attribute {name!r}")
