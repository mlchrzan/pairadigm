"""
Pairadigm: Concept-Guided Chain-of-Thought (CGCoT) pairwise annotation using Large Language Models.

A Python library for systematic evaluation of text items along specific conceptual dimensions 
through structured pairwise comparisons, powered by LLMs validated by human annotations.
"""

__version__ = "0.4.2"
__author__ = "Michael Leon Chrzan"
__license__ = "Apache-2.0"

from .core import Pairadigm, LLMClient, load_pairadigm, build_pairadigm, pair_items
from .model import RewardModel

__all__ = [
    "Pairadigm",
    "LLMClient",
    "load_pairadigm",
    "build_pairadigm",
    "pair_items",
    "RewardModel",
]
