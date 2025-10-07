# Pairadigm Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/mlchrzan/pairadigm.git
cd pairadigm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

## Setup API Keys

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API key(s):
```bash
# For Google Gemini (recommended)
GENAI_API_KEY=your_actual_key_here

# Optional: For other providers
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## Basic Usage (5 Minutes)

### Step 1: Prepare Your Data

Create a CSV file with your items:

```csv
id,text
1,"This is the first item to analyze"
2,"This is the second item to analyze"
3,"This is the third item to analyze"
```

### Step 2: Create CGCoT Prompts

Create a text file `cgcot_prompts.txt`:

```text
Analyze the following text for indicators of {concept}: {text}. List 3-5 specific indicators you observe.
Based on your previous analysis: {previous_answers}. Rate the overall level of {concept} on a scale of 1-5 and explain your rating.
Provide a final assessment of {concept} in the text, considering context and any implicit signals: {text}. Previous analysis: {previous_answers}
```

### Step 3: Run Complete Workflow

```python
import pandas as pd
from pairadigm import Pairadigm, load_cgcot_prompts

# Load data
data = pd.read_csv('your_data.csv')

# Load prompts
prompts = load_cgcot_prompts('cgcot_prompts.txt')

# Initialize Pairadigm
pairadigm = Pairadigm(
    data=data,
    row_id='id',
    row_name='text',
    cgcot_prompts=prompts,
    model_name='gemini-2.0-flash-exp',
    target_concept='objectivity'  # Change to your concept
)

# Run complete workflow
print("Step 1: Generating breakdowns...")
pairadigm.generate_breakdowns(max_workers=4)

print("Step 2: Creating pairings...")
pairadigm.generate_pairings(num_pairs_per_item=10)

print("Step 3: Annotating...")
pairadigm.annotate(parallel=True, max_workers=4)

print("Step 4: Computing scores...")
pairadigm.compute_scores(normalize=True)

# View results
summary = pairadigm.summarize(text_col='text')
pairadigm.plot_scores()
pairadigm.export_results('results.xlsx')
```

## Alternative Annotator Test (10 Minutes)

If you have human annotations to compare:

```python
# Load human annotations (same format as LLM annotations)
human_annotations = pd.read_csv('human_annotations.csv')
# Expected columns: item1, item2, decision

# Validate
validation = pairadigm.validate(
    check_llm_transitivity=True,
    human_annotations=human_annotations
)

# Run Alternative Annotator Test
test_results = pairadigm.run_alternate_annotator_test(
    human_annotations=human_annotations,
    alpha=0.05
)

# Check if LLM passed
if test_results['pass_test']:
    print("✓ LLM can be used for remaining annotations!")
else:
    print("✗ LLM did not pass. Recommendations:")
    print(test_results['recommendation'])
```

## Common Use Cases

### Case 1: You Have Raw Items Only

```python
pairadigm = Pairadigm(data=items_df, row_id='id', row_name='text', ...)
pairadigm.generate_breakdowns()
pairadigm.generate_pairings()
pairadigm.annotate()
pairadigm.compute_scores()
```

### Case 2: You Have Pre-Existing Pairings

```python
existing_pairs = pd.read_csv('existing_pairs.csv')
# Format: item1, item2, breakdown1, breakdown2

pairadigm = Pairadigm(data=items_df, row_id='id', ...)
pairadigm.annotate(pairwise_df=existing_pairs)
pairadigm.compute_scores()
```

### Case 3: You Have Pre-Annotated Data

```python
annotated_data = pd.read_csv('annotated.csv')
# Format: item1, item2, decision, justification

pairadigm = Pairadigm(data=items_df, row_id='id', ...)
pairadigm.compute_scores(pairwise_df=annotated_data)
pairadigm.validate(check_llm_transitivity=True)
```

## Customize Prompts for Your Task

### For Political Bias:
```text
Identify political bias indicators in this text: {text}. Note any loaded language, selective framing, or partisan perspectives.
Based on your analysis: {previous_answers}, rate the political bias from 1 (neutral) to 5 (highly biased).
Provide your final assessment of political bias: {text}. Analysis: {previous_answers}
```

### For Sentiment:
```text
Analyze the sentiment in this text: {text}. Identify positive, negative, and neutral elements.
Rate the overall sentiment: {previous_answers}. Use a scale from 1 (very negative) to 5 (very positive).
Final sentiment assessment for: {text}. Previous: {previous_answers}
```

### For Credibility:
```text
Evaluate credibility indicators in: {text}. Look for sources, evidence, expertise, and potential conflicts.
Rate credibility (1-5): {previous_answers}. Consider completeness and reliability.
Final credibility assessment: {text}. Analysis: {previous_answers}
```

## Tips for Success

### 1. Start Small
- Test with 10-20 items first
- Verify breakdowns are meaningful
- Check a few annotations manually

### 2. Iterate on Prompts
- If annotations seem random, refine your prompts
- Be more specific about what to look for
- Add examples in prompts if needed

### 3. Rate Limiting
```python
# For free-tier APIs, reduce parallel workers
pairadigm.generate_breakdowns(max_workers=2)
pairadigm.annotate(parallel=False, rate_limit_per_minute=10)
```

### 4. Batch Processing for Large Datasets
```python
batch_size = 500
for i in range(0, len(data), batch_size):
    batch_data = data.iloc[i:i+batch_size]
    batch_pairadigm = Pairadigm(batch_data, ...)
    # Process batch
```

## Troubleshooting

### "API key not found"
- Make sure `.env` file is in your working directory
- Check that variable names match exactly (e.g., `GENAI_API_KEY`)
- Try setting environment variables directly in terminal

### "No valid comparisons found"
- Check that your data has `CGCoT_Breakdown` column
- Verify decisions are 'Text1' or 'Text2' (not '1', '2')
- Look for ERROR decisions in your annotations

### "Rate limit exceeded"
- Reduce `max_workers` to 2-4
- Increase sleep time with `rate_limit_per_minute=5`
- Use sequential processing instead of parallel

### "Transitivity violations high"
- Review your CGCoT prompts for clarity
- Check if concept is well-defined
- Consider adding more specific instructions

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check [examples/](examples/) for more complex workflows
3. Review [PACKAGE_GUIDE.md](PACKAGE_GUIDE.md) for development
4. Run tests: `pytest tests/`

## Getting Help

- Open an issue on GitHub
- Check documentation at https://pairadigm.readthedocs.io
- Email: your.email@example.com

## Example Output

After running the workflow, you'll get:

1. **scored_df**: DataFrame with Bradley-Terry scores for each item
2. **Excel file**: Complete results with scores, pairings, and validations
3. **Plots**: Interactive visualizations of score distribution and comparison network
4. **Summary statistics**: Mean, median, std, and top/bottom items

Happy analyzing! 🎉