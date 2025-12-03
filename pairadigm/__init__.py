"""
Pairadigm: Concept-Guided Chain-of-Thought (CGCoT) pairwise annotation using Large Language Models.

A Python library for systematic evaluation of text items along specific conceptual dimensions 
through structured pairwise comparisons, powered by LLMs.
"""

__version__ = "0.2.1"
__author__ = "M.L. Chrzan"
__license__ = "Apache-2.0"

from .core import Pairadigm, LLMClient, load_pairadigm, pair_items

__all__ = [
    "Pairadigm",
    "LLMClient",
    "load_pairadigm",
    "pair_items",
]
