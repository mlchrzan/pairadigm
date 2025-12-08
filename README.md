# Pairadigm

A Python library for Concept-Guided Chain-of-Thought (CGCoT) pairwise annotation using Large Language Models (LLMs). Pairadigm enables systematic evaluation of text items along specific conceptual dimensions through structured pairwise comparisons.

## Overview

Pairadigm uses a multi-stage CGCoT prompting approach to break down complex concepts into analyzable components, then performs pairwise comparisons to rank items using the Bradley-Terry model. It supports multiple LLM providers (Google Gemini, OpenAI, Anthropic) and includes validation tools for comparing LLM annotations against human judgments. 

You can see a full example of the package in use in the `example.ipynb` notebook along with some dummy code below.

## Updates for version 0.4.1 - 2025-12-07

### Added
- **RewardModel Class**: Fine-tune ModernBERT (or other BERT-type model) for scalar construct measurement using reward modeling
  - Train models on pairwise comparison data
  - Score individual texts or batches on continuous scales
  - Support for custom dropout, max length, and device settings
  - Built-in score normalization to desired scales
  - Save/load trained models for reuse
- Support for Ollama LLMs (local models) with `think` parameter
- `build_pairadigm()` function to run full pipeline in one command
- Enhanced progress monitoring for CGCoT breakdown generation

## Installation

### Prerequisites

- Python 3.8+
- API keys for your chosen LLM provider(s)

### Setup
In the terminal, follow these steps:
1. Install the package:
```bash
pip install git+https://github.com/mlchrzan/pairadigm.git
```

2. Set up environment variables:
```bash
# Create a .env file in the project root
touch .env

# Add your API key(s) - choose based on your LLM provider
echo "GENAI_API_KEY=your_google_api_key_here" >> .env
# OR
echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
# OR
echo "ANTHROPIC_API_KEY=your_anthropic_api_key_here" >> .env
```

## Quick Start

Below are the basic workflows for using the package. You can find a full example of this in the jupyter notebook `example.ipynb`.

### Basic Workflow: Unpaired Items

WARNING: If loading .txt files into CGCOT Prompts, ensure the .txt files do NOT have double spaces as these will be interpreted as an additional prompt.

```python
import pandas as pd
from pairadigm import Pairadigm

# Load your data
df = pd.DataFrame({
    'id': ['item1', 'item2', 'item3'],
    'text': ['Text content 1', 'Text content 2', 'Text content 3']
})

# Define CGCoT prompts for your concept
cgcot_prompts = [
    "Analyze the following text for objectivity: {text}",
    "Based on the previous analysis: {previous_answers}\nIdentify any subjective language."
]

# Initialize Pairadigm
p = Pairadigm(
    data=df,
    item_id_name='id',
    text_name='text',
    cgcot_prompts=cgcot_prompts,
    model_name='gemini-2.0-flash-exp',
    target_concept='objectivity'
)

# Generate CGCoT breakdowns
p.generate_breakdowns(max_workers=4)

# Create pairings
p.generate_pairings(num_pairs_per_item=5, breakdowns=True)

# Generate pairwise annotations
p.generate_pairwise_annotations(max_workers=4)

# Compute Bradley-Terry scores
scored_df = p.score_items()

# Visualize results
p.plot_score_distribution()
p.plot_comparison_network()
```

### Using Multiple LLMs

```python
# Initialize with multiple models
p = Pairadigm(
    data=df,
    item_id_name='id',
    text_name='text',
    cgcot_prompts=cgcot_prompts,
    model_name=['gemini-2.0-flash-exp', 'gpt-4o', 'claude-sonnet-4'],
    target_concept='objectivity'
)

# View available clients
print(p.get_clients_info())

# Generate breakdowns with all models
p.generate_breakdowns(max_workers=4)

# Generate annotations with all models
p.generate_pairwise_annotations(max_workers=4)

# Score items for each model
scored_df_gemini = p.score_items(decision_col='decision_gemini-2.0-flash-exp')
scored_df_gpt = p.score_items(decision_col='decision_gpt-4o')
scored_df_claude = p.score_items(decision_col='decision_claude-sonnet-4')
```

### Working with Pre-Paired Data

```python
# Data with pre-existing pairs
paired_df = pd.DataFrame({
    'item1_id': ['a', 'b', 'c'],
    'item2_id': ['b', 'c', 'a'],
    'item1_text': ['Text A', 'Text B', 'Text C'],
    'item2_text': ['Text B', 'Text C', 'Text A']
})

p = Pairadigm(
    data=paired_df,
    paired=True,
    item_id_cols=['item1_id', 'item2_id'],
    item_text_cols=['item1_text', 'item2_text'],
    cgcot_prompts=cgcot_prompts,
    target_concept='political_bias'
)

# Generate breakdowns for paired items
p.generate_breakdowns_from_paired(max_workers=4)

# Continue with annotations and scoring...
p.generate_pairwise_annotations()
p.score_items()
```

### Adding Human Annotations

```python
# Create human annotation data
human_anns = pd.DataFrame({
    'item1': ['id1', 'id2'],
    'item2': ['id2', 'id3'],
    'annotator1': ['Text1', 'Text2'],
    'annotator2': ['Text2', 'Text1']
})

# Add to existing Pairadigm object
p.append_human_annotations(
    annotations=human_anns,
    decision_cols=['annotator1', 'annotator2']
)

# Or load from file
p.append_human_annotations(
    annotations='human_annotations.csv',
    annotator_names=['expert1', 'expert2']
)
```

### Validating Against Human Annotations

```python
# Data with human annotations
annotated_df = pd.DataFrame({
    'item1': ['a', 'b'],
    'item2': ['b', 'c'],
    'item1_text': ['Text A', 'Text B'],
    'item2_text': ['Text B', 'Text C'],
    'human1': ['Text1', 'Text2'],  # Human annotator choices
    'human2': ['Text1', 'Text1']
})

p = Pairadigm(
    data=annotated_df,
    paired=True,
    annotated=True,
    item_id_cols=['item1', 'item2'],
    item_text_cols=['item1_text', 'item2_text'],
    annotator_cols=['human1', 'human2'],
    cgcot_prompts=cgcot_prompts,
    target_concept='sentiment'
)

# Run LLM annotations
p.generate_breakdowns_from_paired()
p.generate_pairwise_annotations()

# Validate using ALT test
winning_rate, advantage_prob = p.alt_test(
    scoring_function='accuracy',
    epsilon=0.1,
    q_fdr=0.05
)

print(f"LLM winning rate: {winning_rate:.2%}")
print(f"Advantage probability: {advantage_prob:.2%}")

# Test all LLMs at once (if using multiple models)
results = p.alt_test(test_all_llms=True)
for model_name, (win_rate, adv_prob) in results.items():
    print(f"{model_name}: Win Rate={win_rate:.2%}, Advantage={adv_prob:.2%}")

# Check transitivity
transitivity_results = p.check_transitivity()
for annotator, (score, violations, total) in transitivity_results.items():
    print(f"{annotator}: {score:.2%} transitivity ({violations}/{total} violations)")

# Calculate inter-rater reliability
irr_results = p.irr(method='auto')
print(irr_results)

# Dawid-Skene validation (accounts for annotator reliability)
ds_results = p.dawid_skene_alt_test(
    alpha=0.05,
    use_by_correction=True
)
print(f"Dawid-Skene Winning Rate: {ds_results['winning_rate']:.2%}")

# Rank all annotators by reliability
ranking = p.dawid_skene_annotator_ranking(random_seed=42)
print(ranking[['annotator', 'reliability', 'rank', 'type']])
```

## CGCoT Prompts

CGCoT prompts are the backbone of Pairadigm's analysis. Design them to progressively analyze your target concept:

### Loading Prompts from File

```python
# prompts.txt format:
# What factual claims are made in this text? {text}
# Based on: {text} Are these claims supported by evidence?
# Does the language show emotional bias?

p.set_cgcot_prompts('prompts.txt')
```
WARNING: If loading .txt files into CGCOT Prompts, ensure the .txt files do NOT have double spaces as these will be interpreted as an additional prompt.

### Best Practices

1. **First prompt**: Identify relevant elements using `{text}` placeholder
2. **Middle prompts**: Build on `{previous_answers}` to deepen analysis
3. **Final prompt**: Synthesize findings related to target concept
4. Keep prompts focused and sequential

## Advanced Features

### Save and Load Analysis

```python
# Save your analysis
p.save('my_analysis.pkl')

# Load it later
from pairadigm import load_pairadigm
p = load_pairadigm('my_analysis.pkl')
```

### Fine-Tuning with RewardModel

```python
from pairadigm import RewardModel

# Prepare training data from pairwise comparisons
training_pairs = [
    ("Text with high score", "Text with low score", 1.0),
    ("Better text", "Worse text", 1.0),
    # ... more pairs
]

# Initialize and train reward model
reward_model = RewardModel(
    model_name="answerdotai/ModernBERT-large",
    dropout=0.1,
    max_length=384
)

train_loader = reward_model.prepare_data(training_pairs, batch_size=16)
reward_model.train(train_loader, epochs=3, learning_rate=2e-5)

# Score new texts
score = reward_model.score_text("New text to evaluate")
scores = reward_model.score_batch(["Text 1", "Text 2", "Text 3"])

# Normalize scores to desired scale (e.g., 1-9)
normalized = reward_model.normalize_scores(scores, scale_min=1.0, scale_max=9.0)

# Save trained model
reward_model.save('my_reward_model.pt')

# Load later
reward_model.load('my_reward_model.pt')
```

### Custom Scoring Functions

```python
def custom_similarity(pred, annotations):
    # Your custom scoring logic
    return score

winning_rate, advantage_prob = p.alt_test(
    scoring_function=custom_similarity
)
```

### Rate Limiting

```python
# Limit API calls to 10 per minute
p.generate_breakdowns(
    max_workers=4,
    rate_limit_per_minute=10
)
```

## API Reference

### Pairadigm Class

**Constructor Parameters:**
- `data`: Input DataFrame
- `item_id_name`: Column name for item IDs (unpaired data)
- `text_name`: Column name for item text (unpaired data)
- `paired`: Whether data is pre-paired
- `item_id_cols`: List of 2 ID columns (paired data)
- `item_text_cols`: List of 2 text columns (paired data)
- `annotated`: Whether data has human annotations
- `annotator_cols`: List of human annotation columns
- `llm_annotator_cols`: List of LLM annotation columns
- `prior_breakdown_cols`: List of existing breakdown columns
- `cgcot_prompts`: List of CGCoT prompt templates
- `model_name`: LLM model identifier(s) - can be string or list of strings
- `target_concept`: Concept being evaluated
- `api_key`: API key(s) for LLM service(s) - can be string or list
- `llm_clients`: Pre-initialized LLMClient(s) - alternative to model_name/api_key

**Key Methods:**
- `generate_breakdowns()`: Create CGCoT analyses for items
- `generate_breakdowns_from_paired()`: Create breakdowns for paired data
- `generate_pairings()`: Create pairwise combinations
- `generate_pairwise_annotations()`: Run LLM comparisons
- `append_human_annotations()`: Add human judgments to analysis
- `score_items()`: Compute Bradley-Terry scores
- `alt_test()`: Validate against human annotations
- `dawid_skene_alt_test()`: Validate with annotator reliability weighting
- `dawid_skene_annotator_ranking()`: Rank annotators by reliability
- `irr()`: Calculate inter-rater reliability
- `check_transitivity()`: Check annotation consistency
- `plot_score_distribution()`: Visualize score distribution
- `plot_comparison_network()`: Visualize comparison graph
- `get_clients_info()`: View information about LLM clients

## Example Datasets

The `data/` directory contains sample datasets to help you get started:

- `emobank.csv`: Full EmoBank dataset with emotional dimension ratings
- `emobank_sample.csv`: Smaller sample for quick testing
- `emobank_small_sample_simAnnotations.csv`: Sample with simulated annotations
- `cgcot_prompts/`: Example prompt files for arousal, dominance, and valence concepts

## Citation

If you use Pairadigm in your research, please cite:

```bibtex
@software{pairadigm2025,
  author = {Chrzan, M.L.},
  title = {pairadigm: Concept-Guided Chain-of-Thought Pairwise Annotation},
  year = {2025},
  url = {https://github.com/mlchrzan/pairadigm}
}
```

## License

Apache 2.0 License

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For questions and issues:
- Open an issue on GitHub
- Check the example notebooks in the repository
- Review the docstrings in `pairadigm.py`

## Upcoming Features
- Performance improvement for multiple models by parallelizing API calls across models, not just within models
- Enhanced validation metrics and visualizations (IN PROGRESS, recommendations welcome!)
    - Improved inter-rater reliability visualizations
    - Item evaluation metrics and visualizations 
- Conversion from Likert-scale annotation to pairwise
- Dawid-Skene item ground truth estimation with and without LLM annotators (NOT STARTED)
- Updated score_items to use the Dawid-Skene estimated ground truth (NOT STARTED)
- Update Dawid-Skene methods to generate multiple runs to examine stability (for now, we recommend examining variance independently over multiple seeds)
- Support for multiple concepts simultaneously (NOT STARTED)

# Previous Updates (see CHANGELOG.md for all)
## Updates for version 0.3.1 - 2025-11-12

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

## Updates for version 0.2.1 🎉

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

**Bug-Fix from version 0.1.0**: Fixed a bug in the `LLMClient` class where certain models did not properly handle the temperature parameter.

## Features

- **Multi-Provider LLM Support**: Works with Google Gemini, OpenAI GPT, and Anthropic Claude models
- **Multiple LLM Annotations**: Use multiple models simultaneously for comparison and consensus
- **Flexible Workflows**: Start with unpaired items, pre-paired data, or human-annotated comparisons
- **CGCoT Breakdowns**: Generate concept-specific analyses using customizable prompt chains
- **Automated Pairwise Comparison**: Parallel processing of comparisons with rate limiting
- **Bradley-Terry Scoring**: Convert pairwise preferences into continuous scores
- **Validation Tools**: 
  - ALT test for comparing LLM vs. human annotations
  - Dawid-Skene model for annotator reliability estimation
  - Inter-rater reliability (Cohen's/Fleiss' Kappa, Krippendorff's Alpha)
  - Transitivity checking for consistency validation
- **Interactive Visualizations**: Distribution plots and network graphs using Plotly
- **Save/Load Functionality**: Persist analysis state for reproducibility