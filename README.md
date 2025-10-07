# Pairadigm

**Concept-Guided Chain-of-Thought (CGCoT) Prompting with Alternative Annotator Test**

Pairadigm is a Python package for researchers who want to leverage Large Language Models (LLMs) for pairwise annotation tasks while ensuring reliability through the Alternative Annotator Test. It combines CGCoT prompting with Bradley-Terry modeling to produce robust construct measurements.

## Features

- ✨ **Flexible Input**: Start with raw items, paired data, or pre-annotated comparisons
- 🧠 **CGCoT Breakdowns**: Concept-guided chain-of-thought prompting for nuanced analysis
- 🔗 **Smart Pairing**: Generates connected comparison graphs with guaranteed coverage
- ⚡ **Parallel Processing**: Fast annotation with concurrent API calls
- 📊 **Bradley-Terry Scoring**: Statistical modeling of pairwise preferences
- ✅ **Validation Suite**: Transitivity checks, inter-annotator agreement, and reliability metrics
- 🔬 **Alternative Annotator Test**: Systematic evaluation of LLM vs. human annotation quality
- 📈 **Visualization**: Interactive plots with Plotly and Matplotlib
- 🔌 **Multi-Backend**: Supports Google GenAI, OpenAI, and Anthropic

## Installation

```bash
pip install pairadigm
```

### With Optional Dependencies

```bash
# For OpenAI support
pip install pairadigm[openai]

# For Anthropic support
pip install pairadigm[anthropic]

# For development
pip install pairadigm[dev]
```

## Quick Start

```python
import pandas as pd
from pairadigm import Pairadigm
from pairadigm.cgcot import load_cgcot_prompts

# Load your data
data = pd.read_csv('items.csv')  # Must have 'id' and 'text' columns

# Load CGCoT prompts
prompts = load_cgcot_prompts('cgcot_prompts.txt')

# Initialize
pairadigm = Pairadigm(
    data=data,
    row_id='id',
    row_name='text',
    cgcot_prompts=prompts,
    model_name='gemini-2.0-flash-exp',
    target_concept='political bias'
)

# Complete workflow
pairadigm.generate_breakdowns()
pairadigm.generate_pairings(num_pairs_per_item=10)
pairadigm.annotate(parallel=True)
pairadigm.compute_scores()

# Get results
summary = pairadigm.summarize(text_col='text')
pairadigm.plot_scores()
pairadigm.export_results('results.xlsx')
```

## Alternative Annotator Test

Validate whether an LLM can replace human annotators:

```python
# Load human annotations
human_annotations = pd.read_csv('human_pairwise.csv')

# Run validation
validation = pairadigm.validate(
    check_llm_transitivity=True,
    human_annotations=human_annotations
)

# Run Alternative Annotator Test
test_results = pairadigm.run_alternate_annotator_test(
    human_annotations=human_annotations,
    alpha=0.05
)

if test_results['pass_test']:
    print("✓ LLM can be used for remaining annotations!")
else:
    print("✗ Continue with human annotators")
```

## CGCoT Prompt Format

Create a text file with sequential prompts (one per line):

```text
Analyze the following text for {concept}: {text}. List specific indicators.
Based on your previous analysis: {previous_answers}. Rate the level of {concept} (1-5).
Provide a final assessment considering context: {text}. Previous: {previous_answers}
```

## Flexible Workflows

### Scenario 1: Start with Raw Items
```python
pairadigm.generate_breakdowns()
pairadigm.generate_pairings()
pairadigm.annotate()
pairadigm.compute_scores()
```

### Scenario 2: Start with Existing Pairings
```python
existing_pairs = pd.read_csv('pairs.csv')
pairadigm.annotate(pairwise_df=existing_pairs)
pairadigm.compute_scores()
```

### Scenario 3: Start with Annotated Data
```python
annotated = pd.read_csv('annotated.csv')
pairadigm.compute_scores(pairwise_df=annotated)
pairadigm.validate(check_llm_transitivity=True)
```

## Validation Metrics

- **Transitivity Rate**: Measures logical consistency (A>B, B>C → A>C)
- **Cohen's Kappa**: Inter-annotator agreement (0.60+ recommended)
- **Agreement Rate**: Direct comparison of decisions (0.70+ recommended)
- **Score Correlation**: Spearman correlation of Bradley-Terry scores
- **Bootstrap Confidence Intervals**: Uncertainty quantification

## API Keys

Set environment variables:

```bash
# Google GenAI
export GENAI_API_KEY='your-key'

# OpenAI
export OPENAI_API_KEY='your-key'

# Anthropic
export ANTHROPIC_API_KEY='your-key'
```

Or use `.env` file:
```
GENAI_API_KEY=your-key
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
```

## Supported Models

- **Google**: `gemini-2.0-flash-exp`, `gemini-1.5-pro`, `gemini-1.5-flash`
- **OpenAI**: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Anthropic**: `claude-sonnet-4`, `claude-opus-4`, `claude-haiku-4`

## Data Format

### Input Data (Raw Items)
```csv
id,text
1,"Sample text for analysis..."
2,"Another text to evaluate..."
```

### Pairwise Comparisons
```csv
item1,item2,breakdown1,breakdown2
1,2,"Breakdown for item 1","Breakdown for item 2"
```

### Annotated Data
```csv
item1,item2,decision,justification
1,2,Text1,"Item 1 shows stronger evidence because..."
2,3,Text2,"Item 3 demonstrates higher levels..."
```

## Citation

If you use Pairadigm in your research, please cite:

```bibtex
@software{pairadigm2024,
  title={Pairadigm: Concept-Guided Chain-of-Thought with Alternative Annotator Test},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/pairadigm}
}
```

## Advanced Usage

### Custom Validation Thresholds

```python
from pairadigm.validation import make_recommendation

# Custom criteria
results = pairadigm.validate(human_annotations=human_data)
results['cohens_kappa'] = 0.65
results['agreement_rate'] = 0.72

recommendation = make_recommendation(results, alpha=0.05)
```

### Bootstrap Confidence Intervals

```python
from pairadigm.validation import bootstrap_confidence_interval

ci_results = bootstrap_confidence_interval(
    pairwise_df=pairadigm.pairwise_df,
    row_id='id',
    n_bootstrap=1000,
    confidence_level=0.95
)

# Access CIs for each item
for item_id, stats in ci_results.items():
    print(f"Item {item_id}: {stats['mean']:.3f} "
          f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")
```

### Krippendorff's Alpha for Multiple Annotators

```python
from pairadigm.validation import calculate_krippendorff_alpha

# Long-format dataframe: annotator, item, decision
annotations = pd.DataFrame({
    'annotator': ['A', 'A', 'B', 'B', 'C', 'C'],
    'item': ['1_vs_2', '2_vs_3', '1_vs_2', '2_vs_3', '1_vs_2', '2_vs_3'],
    'decision': ['Text1', 'Text2', 'Text1', 'Text1', 'Text2', 'Text2']
})

alpha = calculate_krippendorff_alpha(
    annotations_df=annotations,
    annotator_col='annotator',
    item_col='item',
    decision_col='decision'
)
print(f"Krippendorff's Alpha: {alpha:.3f}")
```

### Network Analysis

```python
# Visualize comparison network
pairadigm.plot_network()

# Access network statistics
import networkx as nx
from pairadigm.visualization import build_comparison_graph

G = build_comparison_graph(pairadigm.pairwise_df)
print(f"Network density: {nx.density(G):.3f}")
print(f"Number of components: {nx.number_weakly_connected_components(G)}")
```

## Troubleshooting

### Rate Limiting

If you encounter rate limit errors:

```python
# Reduce parallel workers
pairadigm.annotate(parallel=True, max_workers=4)

# Or use sequential processing with rate limiting
pairadigm.annotate(parallel=False, rate_limit_per_minute=10)
```

### Memory Issues with Large Datasets

Process in batches:

```python
batch_size = 500
for i in range(0, len(data), batch_size):
    batch = data.iloc[i:i+batch_size]
    batch_pairadigm = Pairadigm(batch, row_id='id', ...)
    # Process batch
```

### Transitivity Violations

High violation rates may indicate:
- Ambiguous concept definitions
- Inconsistent prompt responses
- Need for human annotation

```python
# Check violations
validation = pairadigm.validate()
violations = validation['llm_transitivity']['violations']

# Examine specific violations
for a, b, c in violations[:5]:
    print(f"Intransitive: {a} > {b} > {c}, but {c} > {a}")
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development Setup

```bash
git clone https://github.com/yourusername/pairadigm.git
cd pairadigm
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black pairadigm/
flake8 pairadigm/
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pairadigm tests/

# Run specific test file
pytest tests/test_validation.py
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Bradley-Terry modeling via [choix](https://github.com/lucasmaystre/choix)
- Built on pandas, numpy, scikit-learn, and scipy
- Visualization powered by Plotly and Matplotlib

## Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/pairadigm/issues)
- 📖 Docs: [Full Documentation](https://pairadigm.readthedocs.io)

## Changelog

### v0.1.0 (2024-XX-XX)
- Initial release
- CGCoT breakdown generation
- Pairwise comparison and annotation
- Bradley-Terry scoring
- Alternative Annotator Test
- Transitivity validation
- Multi-backend LLM support