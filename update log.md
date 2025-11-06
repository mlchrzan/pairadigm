## v1.3.0 - No-Tie-Die Release
- **Tie Annotations**: Introduced a new "Tie" annotation option to indicate no preference between two items.
- **New Visualization**: plot_epsilon_sensitivity() to visualize how varying the epsilon parameter affects Alt-Test Win Rate.
**Method Updates/Fixes for Ties**:
- `irr` now checks for Tie annotations and handles them correctly when calculating inter-rater reliability.
- `check_transitivity` accounts for Tie annotations in its logic of counting violations.
- `score_items` updated to use the Davidson model when Ties are present, instead of Bradley-Terry.
- `plot_comparison_network` gives a warning if Tie annotations are present, as they cannot be represented in a directed graph.


## v1.2.1
- **Multi-LLM Support**: Annotate with multiple LLM models simultaneously for comparison
- **Upload Human Annotations**: New `append_human_annotations()` method to add human judgments to existing analyses
- **Enhanced Validation**: 
  - Dawid-Skene model implementation for annotator reliability estimation
  - `dawid_skene_alt_test()` for weighted agreement testing
  - `dawid_skene_annotator_ranking()` to rank all annotators by reliability
  - `irr()` method for inter-rater reliability using Cohen's/Fleiss' Kappa or Krippendorff's Alpha
- **Improved Multi-Model Workflows**: Test all LLMs at once with `test_all_llms=True` parameter
- **Allowing for Ties**: Option to allow "Tie" as a valid comparison outcome in generating pairwise annotations
- **Better Error Handling**: Enhanced validation and clearer error messages

**Bug-Fix from version 1.1.0**: Fixed a bug in the `LLMClient` class where certain models did not properly handle the temperature parameter.

## v1.1.0
Added functionality for handling multiple LLM models in a single Pairadigm analysis.
