# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2025-12-14 - A Big Hug! 🤗
### Added 
- Early stopping functionality to RewardModel's finetuning process based on validation loss to prevent overfitting.
- Finetuning now returns the best model based on validation performance rather than the last epoch.
- RewardModel class now includes a `push_to_hub()` method to upload the finetuned model to Hugging Face Model Hub for easy sharing and deployment.
- Now includes support in LLMClient for calling inference via Hugging Face's Inference API, allowing users to leverage Hugging Face-hosted models seamlessly.

### Fixed
- Changed the `_prepped_pairadigm` function to correctly use the item text instead of the breakdown columns when creating pairs. Text is merged from the original data given to the pairadigm instance.
- Updated README.md

## [0.4.2] - 2025-12-08
### Added
- Updated the RewardModel class to create evaluation and test datasets in `prepare_data()` that includes both winning and losing items, allowing for internal assessment of model performance.
- Added a `test_model()` method to the RewardModel class to evaluate the finetuned model on a separate test dataset and report accuracy.

### Fixed
- Fixed a bug in `_prepare_pairadigm()` where the check for the decision column in `pairwise_df` was incorrect, which could lead to errors when creating pairs using the `margin` parameter.
- Updated package imports in `core.py`

## [0.4.1] - 2025-12-07

### Added
- Added support for Ollama LLMs in LLMClient (local models), including the `think` parameter.
  - Updated load_pairadigm() to handle loading Pairadigm objects with Ollama models without requiring API keys
- Progress monitoring when generating CGCoT breakdowns. 
- Create the `build_pairadigm()` function to run the full basic pipeline (breakdowns, pairings, annotation, and validation if human annotations are provided) all in one.
- Added a new `RewardModel` class for finetuning a model based on paired data in `model.py`

## [0.3.1] - 2025-11-12

### Added
- Allowing users to adjust the max_tokens and temperature parameters when generating breakdowns and pairwise annotations.
- Added progress monitoring for breakdown generation (both pre-paired and not)
- Added "base_url" parameter to LLMClient to support custom API endpoints for LLM providers (currently only OpenAI).
- Introduced a new "Tie" annotation option to indicate no preference between two items.
- plot_epsilon_sensitivity() to visualize how varying the epsilon parameter affects Alt-Test Win Rate.

### Fixed
- `irr` now checks for Tie annotations and handles them correctly when calculating inter-rater reliability.
- `check_transitivity` accounts for Tie annotations in its logic of counting violations.
- `score_items` updated to use the Davidson model when Ties are present, instead of Bradley-Terry.
- `plot_comparison_network` gives a warning if Tie annotations are present, as they cannot be represented in a directed graph.

## [0.2.1] - 2025-11-01

### Added
- Multi-LLM Support: Annotate with multiple LLM models simultaneously for comparison
- `append_human_annotations()` method to add human judgments to existing analyses
- Enhanced Validation:
  - Dawid-Skene model implementation for annotator reliability estimation
  - `dawid_skene_alt_test()` for weighted agreement testing
  - `dawid_skene_annotator_ranking()` to rank all annotators by reliability
  - `irr()` method for inter-rater reliability using Cohen's/Fleiss' Kappa or Krippendorff's Alpha
- Improved Multi-Model Workflows: Test all LLMs at once with `test_all_llms=True` parameter
- Allowing for Ties: Option to allow "Tie" as a valid comparison outcome in generating pairwise annotations
- Better Error Handling: Enhanced validation and clearer error messages

### Fixed
- Bug in `LLMClient` class where certain models did not properly handle the temperature parameter

## [0.1.0] - 2025-10-15

### Added
- Initial release
- Concept-Guided Chain-of-Thought (CGCoT) pairwise annotation
- Support for Google Gemini, OpenAI GPT, and Anthropic Claude models
- Automated pairwise comparison with parallel processing
- Bradley-Terry scoring for continuous evaluation
- AltTest for validation against human annotations
- Interactive visualizations with Plotly
- Save/load functionality for analysis persistence
