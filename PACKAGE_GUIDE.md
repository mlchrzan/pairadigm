# Pairadigm Package Development Guide

## Package Structure Overview

The Pairadigm package is organized into modular components:

```
pairadigm/
├── __init__.py          # Package initialization and exports
├── core.py              # Main Pairadigm class
├── cgcot.py             # CGCoT breakdown generation
├── pairing.py           # Pairing generation utilities
├── comparison.py        # Pairwise comparison functions
├── scoring.py           # Bradley-Terry scoring
├── validation.py        # Transitivity & reliability checks
├── visualization.py     # Plotting functions
├── llm_client.py        # LLM API wrapper
└── utils.py             # Helper utilities
```

## Design Decisions

### Class vs. Functions

**Pairadigm Class** (`core.py`):
- Provides a stateful, high-level API
- Manages workflow state (data, pairings, scores)
- Best for interactive use and complete workflows
- Handles data transformations between steps

**Standalone Functions** (other modules):
- Provide low-level, composable operations
- Can be used independently
- Best for custom workflows and batch processing
- No state management overhead

### Why Both Approaches?

1. **Flexibility**: Users can choose their preferred style
2. **Composability**: Functions can be combined in custom ways
3. **Testing**: Functions are easier to unit test
4. **Extensibility**: Users can extend the class or use functions directly

## Key Features

### 1. Flexible Input Handling

The package accepts three types of input:

```python
# Scenario 1: Raw items only
pairadigm = Pairadigm(data=items_df, row_id='id', row_name='text')
pairadigm.generate_breakdowns()
pairadigm.generate_pairings()
pairadigm.annotate()

# Scenario 2: Pre-paired items
pairadigm = Pairadigm(data=items_df, row_id='id')
pairadigm.annotate(pairwise_df=existing_pairs)

# Scenario 3: Pre-annotated data
pairadigm = Pairadigm(data=items_df, row_id='id')
pairadigm.compute_scores(pairwise_df=annotated_pairs)
```

### 2. Transitivity Validation

Checks logical consistency: if A > B and B > C, then A should > C.

**Implementation** (`validation.py`):
- Builds preference graph from comparisons
- Checks all possible triples
- Reports violation rate and specific violations
- Can be used for both LLM and human annotations

### 3. Alternative Annotator Test

Systematic evaluation framework with four criteria:

1. **Inter-annotator agreement** (Cohen's Kappa ≥ 0.60)
2. **Raw agreement rate** (≥ 70%)
3. **LLM transitivity** (≥ 90%)
4. **Score correlation** (Spearman ≥ 0.70, if applicable)

**Usage**:
```python
test_results = pairadigm.run_alternate_annotator_test(
    human_annotations=human_df,
    alpha=0.05
)

if test_results['pass_test']:
    # Use LLM for remaining annotations
else:
    # Continue with human annotators or refine prompts
```

### 4. Multi-Backend LLM Support

The `LLMClient` class abstracts different LLM providers:

```python
# Automatically inferred from model name
client = LLMClient(model_name='gemini-2.0-flash-exp')
client = LLMClient(model_name='gpt-4o')
client = LLMClient(model_name='claude-sonnet-4')

# Or explicitly specify
client = LLMClient(model_name='my-model', provider='google')
```

## Installation Instructions

### For Users

```bash
# Basic installation
pip install pairadigm

# With OpenAI support
pip install pairadigm[openai]

# With Anthropic support
pip install pairadigm[anthropic]

# Development installation
pip install pairadigm[dev]
```

### For Developers

```bash
# Clone repository
git clone https://github.com/yourusername/pairadigm.git
cd pairadigm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest --cov=pairadigm tests/

# Format code
black pairadigm/
flake8 pairadigm/
```

## Adding New Features

### Adding a New Validation Metric

1. Add function to `validation.py`:

```python
def calculate_new_metric(pairwise_df: pd.DataFrame) -> float:
    """Calculate new validation metric."""
    # Implementation
    return metric_value
```

2. Import in `__init__.py`:

```python
from .validation import calculate_new_metric
__all__.append('calculate_new_metric')
```

3. Optionally add to `Pairadigm.validate()` method in `core.py`

### Adding a New Visualization

1. Add function to `visualization.py`:

```python
def plot_new_visualization(df: pd.DataFrame, **kwargs):
    """Create new visualization."""
    # Implementation using plotly or matplotlib
    fig.show()
```

2. Add method to `Pairadigm` class:

```python
def plot_new_viz(self, **kwargs):
    """Create new visualization."""
    if self.scored_df is None:
        raise ValueError("Must compute scores first")
    return plot_new_visualization(self.scored_df, **kwargs)
```

### Adding a New LLM Provider

1. Update `llm_client.py`:

```python
def _infer_provider(self, model_name: str) -> str:
    # Add new provider detection
    if 'new_provider' in model_name.lower():
        return 'new_provider'
    # ...

def _initialize_client(self):
    if self.provider == 'new_provider':
        from new_provider import Client
        return Client(api_key=self.api_key)
    # ...

def _generate_new_provider(self, prompt, system_message, temperature, max_tokens):
    # Implementation
    return response.text
```

2. Update environment variable handling in `_get_api_key()`

## Testing Guidelines

### Unit Tests

Place in `tests/` directory:

```python
# tests/test_new_feature.py
import pytest
from pairadigm import new_feature

def test_new_feature_basic():
    result = new_feature(input_data)
    assert result == expected_output

def test_new_feature_edge_case():
    with pytest.raises(ValueError):
        new_feature(invalid_input)
```

### Integration Tests

Test complete workflows:

```python
def test_complete_workflow():
    pairadigm = Pairadigm(data=test_data, ...)
    pairadigm.generate_breakdowns()
    pairadigm.generate_pairings()
    pairadigm.annotate()
    pairadigm.compute_scores()
    
    assert pairadigm.scored_df is not None
    assert 'Bradley_Terry_Score' in pairadigm.scored_df.columns
```

## Documentation Standards

### Docstring Format

Use NumPy-style docstrings:

```python
def function_name(param1: str, param2: int = 10) -> Dict:
    """
    Short one-line description.
    
    Longer description with more details about the function's
    purpose and behavior.
    
    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int, default=10
        Description of param2
        
    Returns
    -------
    Dict
        Description of return value
        
    Raises
    ------
    ValueError
        When invalid input is provided
        
    Examples
    --------
    >>> result = function_name("test", 20)
    >>> print(result)
    {'key': 'value'}
    
    Notes
    -----
    Additional implementation details or references.
    """
    pass
```

### Type Hints

Always include type hints:

```python
from typing import List, Dict, Optional, Tuple, Union

def process_data(
    df: pd.DataFrame,
    columns: List[str],
    options: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """Process dataframe."""
    pass
```

## Common Patterns

### Error Handling

```python
def function_with_validation(df: pd.DataFrame, required_col: str):
    """Function with proper validation."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    if required_col not in df.columns:
        raise ValueError(f"Column '{required_col}' not found in DataFrame")
    
    try:
        # Main logic
        result = process(df)
    except Exception as e:
        raise RuntimeError(f"Processing failed: {e}") from e
    
    return result
```

### Progress Reporting

```python
def process_items_with_progress(items: List, total: int):
    """Process items with progress updates."""
    for i, item in enumerate(items, 1):
        process(item)
        
        if i % 50 == 0:
            print(f"  Completed {i}/{total} items")
    
    print(f"  Completed all {total} items")
```

### Rate Limiting

```python
import time

def process_with_rate_limit(items: List, rate_limit_per_minute: int = 15):
    """Process items respecting rate limits."""
    sleep_time = 60.0 / rate_limit_per_minute
    
    for i, item in enumerate(items):
        result = process(item)
        
        if i < len(items) - 1:
            time.sleep(sleep_time)
```

## Release Checklist

Before releasing a new version:

- [ ] All tests pass (`pytest tests/`)
- [ ] Code formatted (`black pairadigm/`)
- [ ] Linting clean (`flake8 pairadigm/`)
- [ ] Type checking passes (`mypy pairadigm/`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `setup.py` and `__init__.py`
- [ ] Examples tested
- [ ] README.md accurate

## Troubleshooting

### Import Errors

If you get import errors after installation:

```bash
# Reinstall in editable mode
pip install -e .

# Check installation
pip show pairadigm

# Verify imports
python -c "import pairadigm; print(pairadigm.__version__)"
```

### Test Failures

```bash
# Run tests with verbose output
pytest tests/ -v -s

# Run specific test
pytest tests/test_core.py::TestPairadigmInitialization::test_valid_initialization -v

# Debug with pdb
pytest tests/ --pdb
```

### API Key Issues

```bash
# Check environment variables
echo $GENAI_API_KEY
echo $OPENAI_API_KEY

# Use .env file
cp .env.example .env
# Edit .env with your keys
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests
5. Ensure all tests pass
6. Format code (`black pairadigm/`)
7. Commit (`git commit -m 'Add amazing feature'`)
8. Push (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Questions?

- Open an issue on GitHub
- Email: your.email@example.com
- Check documentation at https://pairadigm.readthedocs.io