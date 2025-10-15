# Pairadigm

A Python library for Concept-Guided Chain-of-Thought (CGCoT) pairwise annotation using Large Language Models (LLMs). Pairadigm enables systematic evaluation of text items along specific conceptual dimensions through structured pairwise comparisons.

## Overview

Pairadigm uses a multi-stage CGCoT prompting approach to break down complex concepts into analyzable components, then performs pairwise comparisons to rank items using the Bradley-Terry model. It supports multiple LLM providers (Google Gemini, OpenAI, Anthropic) and includes validation tools for comparing LLM annotations against human judgments.

## Features

- **Multi-Provider LLM Support**: Works with Google Gemini, OpenAI GPT, and Anthropic Claude models
- **Flexible Workflows**: Start with unpaired items, pre-paired data, or human-annotated comparisons
- **CGCoT Breakdowns**: Generate concept-specific analyses using customizable prompt chains
- **Automated Pairwise Comparison**: Parallel processing of comparisons with rate limiting
- **Bradley-Terry Scoring**: Convert pairwise preferences into continuous scores
- **Validation Tools**: 
  - ALT test for comparing LLM vs. human annotations
  - Transitivity checking for consistency validation
- **Interactive Visualizations**: Distribution plots and network graphs using Plotly
- **Save/Load Functionality**: Persist analysis state for reproducibility

## Installation

### Prerequisites

- Python 3.8+
- API keys for your chosen LLM provider(s)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mlchrzan/pairadigm.git
cd pairadigm
```

2. Install dependencies:
```bash
pip install pandas numpy scipy plotly networkx choix python-dotenv google-genai
# Add openai and/or anthropic if using those providers:
# pip install openai anthropic
```

3. Set up environment variables:
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

### Basic Workflow: Unpaired Items

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

# Check transitivity
transitivity_results = p.check_transitivity()
for annotator, (score, violations, total) in transitivity_results.items():
    print(f"{annotator}: {score:.2%} transitivity ({violations}/{total} violations)")
```

## CGCoT Prompts

CGCoT prompts are the backbone of Pairadigm's analysis. Design them to progressively analyze your target concept:

### Loading Prompts from File

```python
# prompts.txt format:
# What factual claims are made in this text? {text}
# ---
# Based on: {previous_answers}
# Are these claims supported by evidence?
# ---
# Does the language show emotional bias?

p.set_cgcot_prompts('prompts.txt')
```

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
- `cgcot_prompts`: List of CGCoT prompt templates
- `model_name`: LLM model identifier
- `target_concept`: Concept being evaluated

**Key Methods:**
- `generate_breakdowns()`: Create CGCoT analyses for items
- `generate_pairings()`: Create pairwise combinations
- `generate_pairwise_annotations()`: Run LLM comparisons
- `score_items()`: Compute Bradley-Terry scores
- `alt_test()`: Validate against human annotations
- `check_transitivity()`: Check annotation consistency
- `plot_score_distribution()`: Visualize score distribution
- `plot_comparison_network()`: Visualize comparison graph

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
  title = {Pairadigm: Concept-Guided Chain-of-Thought Pairwise Annotation},
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

- Multi-LLM comparison mode
- Simultaneous evaluation of multiple concepts
- Enhanced validation metrics
- Additional visualization options
