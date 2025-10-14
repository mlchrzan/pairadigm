"""
Pairadigm: Concept-Guided Chain-of-Thought with Alternative Annotator Test

A Python package for researchers conducting pairwise annotation tasks with LLMs,
featuring systematic validation through the Alternative Annotator Test.
"""

from .core import Pairadigm
from .cgcot import (
    load_cgcot_prompts,
    generate_cgcot_breakdown,
    generate_breakdowns_parallel
)
from .pairing import (
    pair_items,
    generate_pairings_df
)
from .comparison import (
    pairwise_compare,
    generate_pairwise_annotations,
    generate_pairwise_annotations_parallel
)
from .scoring import (
    bradley_terry_scores,
    summarize_scores
)
from .validation import (
    check_transitivity,
    compare_annotators,
    alternate_annotator_test,
    bootstrap_confidence_interval,
    calculate_krippendorff_alpha
)
from .visualization import (
    plot_score_distribution,
    plot_comparison_network
)
from .llm_client import LLMClient, query_llm

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # Main class
    "Pairadigm",
    
    # CGCoT functions
    "load_cgcot_prompts",
    "generate_cgcot_breakdown",
    "generate_breakdowns_parallel",
    
    # Pairing functions
    "pair_items",
    "generate_pairings_df",
    
    # Comparison functions
    "pairwise_compare",
    "generate_pairwise_annotations",
    "generate_pairwise_annotations_parallel",
    
    # Scoring functions
    "bradley_terry_scores",
    "summarize_scores",
    
    # Validation functions
    "check_transitivity",
    "compare_annotators",
    "alternate_annotator_test",
    "bootstrap_confidence_interval",
    "calculate_krippendorff_alpha",
    
    # Visualization functions
    "plot_score_distribution",
    "plot_comparison_network",
    
    # LLM client
    "LLMClient",
    "query_llm",
]