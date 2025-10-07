# Pairadigm Package - Complete Summary

## 📦 What's Included

### Core Modules (9 files)

1. **`__init__.py`** - Package initialization and public API
2. **`core.py`** - Main `Pairadigm` class with workflow management
3. **`cgcot.py`** - CGCoT breakdown generation with parallel processing
4. **`pairing.py`** - Smart pairing generation ensuring graph connectivity
5. **`comparison.py`** - Pairwise comparison and LLM annotation
6. **`scoring.py`** - Bradley-Terry modeling and score analysis
7. **`validation.py`** - Transitivity checks and Alternative Annotator Test
8. **`visualization.py`** - Interactive plots with Plotly and Matplotlib
9. **`llm_client.py`** - Multi-backend LLM support (Google, OpenAI, Anthropic)
10. **`utils.py`** - Helper utilities and data quality checks

### Support Files

- **`setup.py`** - Installation configuration
- **`requirements.txt`** - Dependencies
- **`README.md`** - Full documentation
- **`QUICKSTART.md`** - 5-minute getting started guide
- **`PACKAGE_GUIDE.md`** - Developer documentation
- **`.env.example`** - Environment variable template

### Tests & Examples

- **`tests/test_core.py`** - Comprehensive unit tests
- **`examples/complete_workflow.py`** - Full usage examples

## 🎯 Key Features

### 1. Flexible Input Handling ✓
Three workflow options:
- Raw items → breakdowns → pairings → annotations → scores
- Pre-paired items → annotations → scores  
- Pre-annotated data → scores & validation

### 2. CGCoT Breakdowns ✓
- Sequential prompt chaining for nuanced analysis
- Parallel processing for speed (8+ workers)
- Template-based prompts with placeholders
- Rate limiting to respect API quotas

### 3. Smart Pairing ✓
- Ensures graph connectivity (all items reachable)
- Guarantees minimum comparisons per item
- Optional stratified pairing by categories
- Balanced Incomplete Block Design (BIBD) support

### 4. Multi-Backend LLM Support ✓
- **Google GenAI**: Gemini models (primary)
- **OpenAI**: GPT models (optional)
- **Anthropic**: Claude models (optional)
- Automatic provider inference from model name
- Unified API across providers

### 5. Bradley-Terry Scoring ✓
- Statistical modeling of pairwise preferences
- Optional normalization to [0, 1]
- Robust ILSR algorithm with L2 regularization
- Score confidence intervals
- Ranking and comparison utilities

### 6. Transitivity Validation ✓ NEW!
- Checks logical consistency (A>B, B>C → A>C)
- Reports violation rate and specific violations
- Works for both LLM and human annotations
- Identifies problematic comparison triples

### 7. Alternative Annotator Test ✓ NEW!
Systematic framework with 4 criteria:
- **Agreement Rate** (≥70% recommended)
- **Cohen's Kappa** (≥0.60 recommended)
- **LLM Transitivity** (≥90% recommended)
- **Score Correlation** (≥0.70 if applicable)

Returns PASS/FAIL with detailed recommendations

### 8. Advanced Validation ✓ NEW!
- **Krippendorff's Alpha** for multiple annotators
- **Bootstrap confidence intervals** for uncertainty
- **Agreement matrices** for annotator comparison
- **Data quality checks** with diagnostics

### 9. Rich Visualizations ✓
- Score distribution histograms (Plotly/Matplotlib)
- Network graphs of pairwise comparisons
- Transitivity violation highlighting
- Score vs. feature scatter plots
- Annotator agreement heatmaps
- Top-N ranking bar charts

### 10. Comprehensive Utilities ✓
- Data validation and quality checks
- HTML report generation
- Excel export with multiple sheets
- Stratified sampling for validation
- Decision normalization
- Progress tracking

## 📊 Complete API Overview

### Main Class: `Pairadigm`

```python
from pairadigm import Pairadigm

pairadigm = Pairadigm(
    data: pd.DataFrame,              # Your items
    row_id: str,                     # ID column
    row_name: str = None,            # Text column
    cgcot_prompts: List[str] = None, # CGCoT prompts
    model_name: str = 'gemini-2.0-flash-exp',
    target_concept: str = None       # e.g., 'bias'
)

# Workflow methods
pairadigm.generate_breakdowns(max_workers=8)
pairadigm.generate_pairings(num_pairs_per_item=10)
pairadigm.annotate(parallel=True, max_workers=8)
pairadigm.compute_scores(normalize=True)

# Validation methods
pairadigm.validate(
    check_llm_transitivity=True,
    human_annotations=human_df
)
pairadigm.run_alternate_annotator_test(
    human_annotations=human_df,
    alpha=0.05
)

# Analysis methods
pairadigm.summarize(text_col='text')
pairadigm.plot_scores()
pairadigm.plot_network()
pairadigm.export_results('results.xlsx')
```

### Standalone Functions

```python
from pairadigm import (
    # CGCoT
    load_cgcot_prompts,
    generate_cgcot_breakdown,
    generate_breakdowns_parallel,
    
    # Pairing
    pair_items,
    generate_pairings_df,
    stratified_pairing,
    
    # Comparison
    pairwise_compare,
    generate_pairwise_annotations,
    generate_pairwise_annotations_parallel,
    
    # Scoring
    bradley_terry_scores,
    summarize_scores,
    rank_items,
    compare_score_distributions,
    
    # Validation
    check_transitivity,
    compare_annotators,
    alternate_annotator_test,
    bootstrap_confidence_interval,
    calculate_krippendorff_alpha,
    
    # Visualization
    plot_score_distribution,
    plot_comparison_network,
    plot_score_comparison,
    plot_ranking,
    plot_transitivity_violations,
    
    # LLM
    LLMClient,
    query_llm
)
```

## 🔧 Installation & Setup

### Install Package
```bash
# Development install
git clone https://github.com/yourusername/pairadigm.git
cd pairadigm
pip install -e .

# Or from PyPI (when published)
pip install pairadigm
```

### Setup Environment
```bash
# Copy template
cp .env.example .env

# Edit with your API key
echo "GENAI_API_KEY=your_key_here" >> .env
```

### Verify Installation
```python
import pairadigm
print(pairadigm.__version__)  # Should print: 0.1.0

# Test imports
from pairadigm import Pairadigm
from pairadigm.validation import check_transitivity
```

## 📝 Usage Patterns

### Pattern 1: Complete Workflow from Scratch
```python
pairadigm = Pairadigm(data=df, row_id='id', row_name='text', ...)
pairadigm.generate_breakdowns()
pairadigm.generate_pairings()
pairadigm.annotate()
pairadigm.compute_scores()
pairadigm.summarize(text_col='text')
```

### Pattern 2: Start with Existing Pairings
```python
pairadigm = Pairadigm(data=df, row_id='id', ...)
pairadigm.annotate(pairwise_df=existing_pairs)
pairadigm.compute_scores()
```

### Pattern 3: Validate Pre-Annotated Data
```python
pairadigm = Pairadigm(data=df, row_id='id', ...)
pairadigm.compute_scores(pairwise_df=annotated)
validation = pairadigm.validate(check_llm_transitivity=True)
```

### Pattern 4: Alternative Annotator Test
```python
# Annotate subset with both LLM and humans
validation_sample = df.sample(100)
pairadigm_llm = Pairadigm(validation_sample, ...)
pairadigm_llm.generate_breakdowns()
pairadigm_llm.generate_pairings()
pairadigm_llm.annotate()

# Compare with human annotations
test_results = pairadigm_llm.run_alternate_annotator_test(
    human_annotations=human_df
)

if test_results['pass_test']:
    # Use LLM for full dataset
    pairadigm_full = Pairadigm(df, ...)
    pairadigm_full.generate_breakdowns()
    pairadigm_full.generate_pairings()
    pairadigm_full.annotate()
```

### Pattern 5: Batch Processing
```python
results = []
for i in range(0, len(df), 500):
    batch = df.iloc[i:i+500]
    p = Pairadigm(batch, ...)
    p.generate_breakdowns()
    p.generate_pairings()
    p.annotate()
    p.compute_scores()
    results.append(p.scored_df)

final = pd.concat(results)
```

## 📈 Data Format Requirements

### Input Data (CSV/DataFrame)
```csv
id,text,category
1,"Text to analyze",A
2,"Another text",B
```

### CGCoT Prompts (TXT file)
```text
Analyze: {text}. Look for {concept} indicators.
Rate {concept} (1-5): {previous_answers}
Final assessment: {text}. Previous: {previous_answers}
```

### Pairwise Annotations (CSV/DataFrame)
```csv
item1,item2,decision,justification
1,2,Text1,"Item 1 shows stronger..."
2,3,Text2,"Item 3 demonstrates..."
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run with Coverage
```bash
pytest --cov=pairadigm tests/
```

### Run Specific Test
```bash
pytest tests/test_core.py::TestPairadigmInitialization -v
```

## 🎓 Example Research Workflow

### Research Question
*"Can an LLM reliably annotate news articles for political bias?"*

### Step 1: Prepare Data (Day 1)
```python
articles = pd.read_csv('news_articles.csv')
prompts = load_cgcot_prompts('political_bias_prompts.txt')
```

### Step 2: Pilot Test (Day 1-2)
```python
# Test on small sample
pilot = articles.sample(50)
p = Pairadigm(pilot, row_id='id', row_name='text', 
              cgcot_prompts=prompts, target_concept='political bias')
p.generate_breakdowns()
# Manually review breakdowns for quality
```

### Step 3: Human Annotations (Day 3-7)
```python
# Generate pairings for human annotation
p.generate_pairings(num_pairs_per_item=10)
p.pairwise_df.to_csv('for_human_annotation.csv')

# Humans annotate independently
# Load results
human_annotations = pd.read_csv('human_completed.csv')
```

### Step 4: LLM Annotations (Day 8)
```python
# Annotate same pairs with LLM
p.annotate(parallel=True)
```

### Step 5: Validation (Day 8)
```python
# Check human reliability
human_trans = check_transitivity(human_annotations, "Human")
print(f"Human transitivity: {human_trans['transitivity_rate']:.2%}")

# Run Alternative Annotator Test
test = p.run_alternate_annotator_test(human_annotations)
```

### Step 6: Decision (Day 8)
```python
if test['pass_test']:
    print("✓ Proceeding with LLM for full dataset")
    # Annotate remaining articles
    full_p = Pairadigm(articles, ...)
    full_p.generate_breakdowns()
    full_p.generate_pairings()
    full_p.annotate()
    full_p.compute_scores()
else:
    print("✗ Continue with human annotators")
    print(f"Issues: {test['recommendation']}")
```

### Step 7: Analysis (Day 9-10)
```python
# Analyze results
summary = full_p.summarize(text_col='text')
full_p.plot_scores(title='Political Bias in News Articles')

# Export for paper
full_p.export_results('final_results.xlsx')

# Stratified analysis
by_source = full_p.scored_df.groupby('source')['Bradley_Terry_Score'].mean()
```

## 🔬 Validation Thresholds

### Recommended Criteria for Alternative Annotator Test

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Cohen's Kappa | ≥ 0.60 | Moderate agreement |
| Agreement Rate | ≥ 70% | Acceptable consistency |
| LLM Transitivity | ≥ 90% | Logical consistency |
| Score Correlation | ≥ 0.70 | Similar rankings |

### Adjusting Thresholds
```python
# Stricter criteria for high-stakes tasks
test = pairadigm.run_alternate_annotator_test(human_df, alpha=0.01)

# Custom thresholds in validation.py
results['cohens_kappa'] = 0.65  # Your value
recommendation = make_recommendation(results, alpha=0.05)
```

## 📚 Dependencies

### Core
- pandas ≥ 1.3.0
- numpy ≥ 1.21.0
- scikit-learn ≥ 1.0.0
- scipy ≥ 1.7.0
- choix ≥ 0.3.5

### Visualization
- plotly ≥ 5.0.0
- matplotlib ≥ 3.4.0
- networkx ≥ 2.6.0

### LLM
- google-genai ≥ 0.1.0
- python-dotenv ≥ 0.19.0

### Optional
- openai ≥ 1.0.0 (for GPT models)
- anthropic ≥ 0.18.0 (for Claude models)

## 🚀 Performance Tips

### Speed Optimization
```python
# Maximize parallel workers (within API limits)
pairadigm.generate_breakdowns(max_workers=16)
pairadigm.annotate(parallel=True, max_workers=16)

# Use faster models for large datasets
model_name='gemini-2.0-flash-exp'  # Faster
# vs
model_name='gemini-1.5-pro'  # More accurate but slower
```

### Memory Optimization
```python
# Process in batches
batch_size = 500
for batch in np.array_split(df, len(df)//batch_size):
    process_batch(batch)
```

### Cost Optimization
```python
# Reduce comparisons per item
num_pairs_per_item=5  # vs 10

# Use fewer CGCoT prompts
prompts = prompts[:2]  # Use first 2 only

# Sample for pilot testing
df.sample(50)
```

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| API key not found | Check `.env` file location and variable names |
| Rate limit exceeded | Reduce `max_workers` or add `rate_limit_per_minute` |
| No valid comparisons | Check decision format (must be 'Text1' or 'Text2') |
| High transitivity violations | Refine CGCoT prompts for clarity |
| Low agreement with humans | Review concept definition and prompt specificity |
| Memory error | Use batch processing for large datasets |

## 📖 Citation

```bibtex
@software{pairadigm2024,
  title={Pairadigm: Concept-Guided Chain-of-Thought with Alternative Annotator Test},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/pairadigm},
  version={0.1.0}
}
```

## 📬 Support & Contact

- **Issues**: https://github.com/yourusername/pairadigm/issues
- **Discussions**: https://github.com/yourusername/pairadigm/discussions
- **Email**: your.email@example.com
- **Docs**: https://pairadigm.readthedocs.io

## 🎉 You're Ready!

All modules are complete and ready to use. Next steps:

1. ✅ Copy all artifacts to your `pairadigm/` directory
2. ✅ Run `pip install -e .` to install
3. ✅ Set up your `.env` file with API keys
4. ✅ Try the quickstart example
5. ✅ Run tests with `pytest`
6. ✅ Start annotating!

Happy researching! 🔬📊