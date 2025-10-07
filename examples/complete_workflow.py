"""
Complete workflow example for Pairadigm package.

This example demonstrates:
1. Starting from raw items
2. Generating CGCoT breakdowns
3. Creating pairwise comparisons
4. Annotating with LLM
5. Computing Bradley-Terry scores
6. Validating with the Alternative Annotator Test
"""

import pandas as pd
from pairadigm import Pairadigm
from pairadigm.cgcot import load_cgcot_prompts

# ============================================================================
# SCENARIO 1: Complete workflow from raw items
# ============================================================================

# Load your data
data = pd.read_csv('your_data.csv')
# Expected columns: 'id', 'text', and any other metadata

# Load CGCoT prompts from file
cgcot_prompts = load_cgcot_prompts('cgcot_prompts.txt')

# Initialize Pairadigm
pairadigm = Pairadigm(
    data=data,
    row_id='id',
    row_name='text',
    cgcot_prompts=cgcot_prompts,
    model_name='gemini-2.0-flash-exp',
    target_concept='political bias'
)

# Step 1: Generate CGCoT breakdowns
print("Step 1: Generating CGCoT breakdowns...")
pairadigm.generate_breakdowns(max_workers=8)

# Step 2: Generate pairwise comparisons
print("Step 2: Generating pairwise comparisons...")
pairadigm.generate_pairings(num_pairs_per_item=10)

# Step 3: Annotate comparisons
print("Step 3: Annotating comparisons with LLM...")
pairadigm.annotate(parallel=True, max_workers=8)

# Step 4: Compute Bradley-Terry scores
print("Step 4: Computing Bradley-Terry scores...")
pairadigm.compute_scores(normalize=True)

# Step 5: Summarize results
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
summary = pairadigm.summarize(text_col='text')

# Step 6: Visualize
print("\nGenerating visualizations...")
pairadigm.plot_scores(title='Distribution of Political Bias Scores')
pairadigm.plot_network()

# Step 7: Export results
pairadigm.export_results('pairadigm_results.xlsx')

# ============================================================================
# SCENARIO 2: Alternative Annotator Test with human annotations
# ============================================================================

# Load human annotations (same format as LLM annotations)
human_annotations = pd.read_csv('human_annotations.csv')
# Expected columns: item1, item2, decision

# Validate LLM annotations
print("\n" + "="*60)
print("VALIDATION")
print("="*60)
validation = pairadigm.validate(
    check_llm_transitivity=True,
    human_annotations=human_annotations
)

# Run Alternative Annotator Test
print("\n" + "="*60)
print("ALTERNATIVE ANNOTATOR TEST")
print("="*60)
test_results = pairadigm.run_alternate_annotator_test(
    human_annotations=human_annotations,
    alpha=0.05
)

if test_results['pass_test']:
    print("\n✓ LLM passed the Alternative Annotator Test!")
    print("  You can use the LLM for remaining annotations.")
else:
    print("\n✗ LLM did not pass the Alternative Annotator Test.")
    print("  Consider refining prompts or using human annotators.")

# ============================================================================
# SCENARIO 3: Starting with pre-existing pairings
# ============================================================================

# Load pre-existing pairings
existing_pairings = pd.read_csv('existing_pairings.csv')
# Expected columns: item1, item2, breakdown1, breakdown2

pairadigm2 = Pairadigm(
    data=data,
    row_id='id',
    target_concept='objectivity',
    model_name='gpt-4o'
)

# Annotate existing pairings
pairadigm2.annotate(pairwise_df=existing_pairings)

# Compute scores
pairadigm2.compute_scores()

# ============================================================================
# SCENARIO 4: Starting with pre-annotated data
# ============================================================================

# Load pre-annotated data
annotated_data = pd.read_csv('annotated_comparisons.csv')
# Expected columns: item1, item2, decision, justification

pairadigm3 = Pairadigm(
    data=data,
    row_id='id',
    target_concept='sentiment'
)

# Just compute scores
pairadigm3.compute_scores(pairwise_df=annotated_data)

# Validate transitivity
validation = pairadigm3.validate(check_llm_transitivity=True)
print(f"Transitivity rate: {validation['llm_transitivity']['transitivity_rate']:.2%}")

# ============================================================================
# SCENARIO 5: Custom CGCoT prompts
# ============================================================================

custom_prompts = [
    """Analyze the following text for indicators of {concept}:
    Text: {text}
    
    List 3-5 specific indicators you observe:""",
    
    """Based on your previous analysis:
    {previous_answers}
    
    Rate the overall level of {concept} on a scale of 1-5 and explain:""",
    
    """Consider the context and any implicit biases:
    Text: {text}
    Previous analysis: {previous_answers}
    
    Provide a final assessment of {concept}:"""
]

pairadigm_custom = Pairadigm(
    data=data,
    row_id='id',
    row_name='text',
    cgcot_prompts=custom_prompts,
    model_name='claude-sonnet-4',
    target_concept='trustworthiness'
)

# ============================================================================
# SCENARIO 6: Batch processing for large datasets
# ============================================================================

def process_batch(data_batch, batch_num):
    """Process a single batch of data."""
    print(f"\nProcessing batch {batch_num}...")
    
    pairadigm_batch = Pairadigm(
        data=data_batch,
        row_id='id',
        row_name='text',
        cgcot_prompts=cgcot_prompts,
        model_name='gemini-2.0-flash-exp',
        target_concept='credibility'
    )
    
    pairadigm_batch.generate_breakdowns()
    pairadigm_batch.generate_pairings()
    pairadigm_batch.annotate(parallel=True)
    pairadigm_batch.compute_scores()
    
    return pairadigm_batch.scored_df

# Process in batches of 1000
batch_size = 1000
results = []

for i in range(0, len(data), batch_size):
    batch = data.iloc[i:i+batch_size]
    result = process_batch(batch, i//batch_size + 1)
    results.append(result)

# Combine results
final_results = pd.concat(results, ignore_index=True)
final_results.to_csv('batch_results.csv', index=False)

print("\n" + "="*60)
print("WORKFLOW COMPLETE")
print("="*60)