# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-04-18
### Updated
- **Robust Davidson Scoring**: Replaced the unstable iterative approach for estimating Davidson scores with a mathematically robust optimization method (`scipy.optimize.minimize`). This explicitly estimates both item strengths and the tie propensity parameter ($\tau$) efficiently. 
- **Reward Model Integrations**: Improved dynamic column fallback in `RewardModel.prepare_data()` to seamlessly support Davidson scores when present.

### Fixed
- **F-string Syntax Error**: Fixed an invalid string formulation containing literal backslashes inside an f-string evaluated in `pair_from_ordinal()`. 

## [1.0.0] - 2026-04-16 - 'Summer Body'
### Added
- **Safer Saving Logic**: Instead of using pickles, `pairadigm` now saves and loads data using individual parquet files, which are more robust and efficient. This also means that `pairadigm` objects are now much smaller and faster to load. It also saves the instance construction parameters in a `metadata.json` file, which is used to reconstruct the object when loading.
- **LLM API Cost Estimation**: Added `estimate_costs()` method in `Pairadigm` class to estimate token usage and cost using `tiktoken` or a heuristic fallback prior to large API operations.
- **Client Addition Workflows**: Enhanced `add_client()` logic to allow users to optionally generate breakdowns and pairwise comparisons exclusively for a newly added model.
- **Dawid-Skene Enhancements**: `dawid_skene_annotator_ranking()` now returns the confusion matrix in addition to ranking metrics. Added warnings for 3-class (tied) classifications.
- **Reasoning Models**: Introduced support for passing a reasoning level parameter to LLM clients (simplifying the OpenAI wrapper and removing redundant temperature tracking).
- **Base URLs Persistence**: Base URLs defined for `LLMClient` instances are now persisted within `persistence.py` metadata and seamlessly rehydrated during `load_pairadigm()`.

### Updated
- **Unified Breakdowns**: Consolidated `generate_breakdowns()` and `generate_breakdowns_from_paired()` into a single `generate_breakdowns()` method.
- **Module-Level Ordinal Logic**: Decoupled `pair_from_ordinal()` from the `Pairadigm` class into module-level logic. It now processes multi-annotator decision columns robustly.
- **Robust Extraction Regex**: Enhanced `pairwise_compare()` internal parsing logic to be far more forgiving for weaker or local models that deviate from strict formatting.
- **Documentation Overhaul**: Complete rewrite of docstrings across all public functions in `core.py`, providing rich parameters, examples, and researcher-friendly contexts.
- **Column Prefixing**: Standardized `human_` prefixes for manual annotations to eliminate collisions with the LLM `decision_` columns. 

### Fixed
- **AltTest Filtering Constraints**: Fixed logic in `alt_test()` where valid data entries were being erroneously dropped. 
- **Dawid-Skene Arguments**: Resolved a duplicate kwargs collision error within `dawid_skene_annotator_ranking()`.
- **Sparse Ordinal Data Bugs**: Addressed sparse annotation matrix routing issues when handling multi-annotator datasets without globally shared coverage.

## [0.5.4] - 2026-03-14 - 10/10 on the Splits
### Added 
- Users now have the ability to pass their own system_prompts and comparison_prompts if desired. 

### Updated
- `score_items()` now also can respect the splits created when generating pairings (or providing split data).

### Fixed 
- Syntax error where a print statement was misplaced in `score_items()` causing the method to not function.  
- Typo fixed: tempature → temperature in the signature, docstring, call site inside `generate_breakdowns_from_paired`, and the `build_pairadigm` call.
- Index mismatch: Changed results from a list to a dict, and replaced `r[0] for r in results`/`[r[1] for r in results]` with `result_df.index.map(...)` so non-zero-based or non-contiguous DataFrame indices are handled safely.
- Duplicate llm_annotator_cols entries: Added an `if decision_col not in self.llm_annotator_cols` guard before appending.

## [0.5.3] - 2026-03-14 - Split Personality 🖖🏽
### Added 
- `generate_pairings()` now supports item-level train/eval/test splits via a new `make_splits` parameter, preventing data leakage when pairs are used to train a `RewardModel`. When enabled, splits are generated at the item level (no item appears in more than one split), and resulting pairs are tagged with `item1_split` and `item2_split` columns.
  - `test_size` (default `0.15`) and `eval_size` (default `0.15`) control the proportion of items assigned to each held-out split.
  - Passing a non-default `test_size` or `eval_size` automatically enables `make_splits=True` with a warning.
  - `include_mixed_pairs` (default `False`) optionally appends a small number of intentional cross-split pairs, spread evenly across the train×eval, train×test, and eval×test combinations, useful for diagnosing generalisation gaps.
  - `num_mixed_pairs` (default `10`) controls the total number of cross-split pairs added when `include_mixed_pairs=True`.
- In accordance with the `generate_pairings()` update, the `RewardModel` class will now respect the data splits generated in `generate_pairings()`. It will also encourage users' data hygiene by asking them to either pass splits with their pairs - if just using the model without a `Pairadigm` - or warning them of the data leakage risk.
- `test_client_connections()` function in `Pairadigm` to verify API connectivity for all LLMClients.
- Progress monitoring when generating breakdowns from pre-paired data. 

### Updated
- The Davidson model in `score_items()` now uses NumPy broadcasting for efficiency and has progress monitoring. 
- If a user passes prior_breakdown_cols to the initial `Pairadigm` constructor, the constructor will also create the pairwise_df without needing to call `generator_pairings(breakdowns=True)` separately.

### Fixed 
- Fixed a logic error when creating a `Pairadigm` from paired data where `generate_breakdowns_from_paired()` needed item_id_col to be set but that wasn't enforced. Now if an `item_id_col` isn't set and `paired=True` a default one will be assigned (`item_id_DEFAULT`). 

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
