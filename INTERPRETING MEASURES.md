# Alternative Annotator Test (ALT-TEST) Interpretation Guide

## Overview

The ALT-TEST (Alternative Annotator Test) is a statistical method that evaluates whether an LLM annotator performs comparably to human annotators on a pairwise comparison task. It answers the question: **"Can the LLM substitute for human annotators?"**

## Key Metrics

### 1. Winning Rate (ω)

**Definition:** The proportion of human annotators that the LLM significantly outperforms or matches.

**Interpretation:**
- **ω ≥ 0.75**: Strong evidence that LLM can substitute for humans
- **0.50 ≤ ω < 0.75**: Moderate evidence; LLM performs comparably to many annotators
- **ω < 0.50**: Weak evidence; LLM underperforms relative to human consensus

**What it means:**
- ω = 1.0: LLM performs as well as or better than ALL human annotators
- ω = 0.5: LLM performs as well as half of the human annotators
- ω = 0.0: LLM performs worse than all human annotators

### 2. Advantage Probability

**Definition:** The average probability that the LLM's annotations align better with the consensus of other annotators than the held-out human annotator does.

**Interpretation:**
- **> 0.60**: LLM shows strong alignment with human consensus
- **0.50-0.60**: LLM shows moderate alignment
- **< 0.50**: LLM shows weak alignment; humans are more internally consistent

**What it means:**
This metric captures how often the LLM "agrees with the group" compared to individual humans agreeing with the group (excluding themselves).

## Statistical Testing

### Benjamini-Yekutieli (BY) Procedure

The test uses BY correction to control the False Discovery Rate (FDR) at α = 0.05, accounting for dependencies between tests (since annotators may overlap on items).

**What this controls:**
- Without correction, testing many annotators could lead to false positives
- BY correction ensures that approximately 95% of "wins" declared for the LLM are genuine

### Leave-One-Out Testing

For each human annotator:
1. That annotator is temporarily removed from the pool
2. LLM and the held-out annotator are compared on how well they agree with remaining annotators
3. A paired t-test determines if LLM significantly outperforms the human

**Why this matters:**
This approach simulates replacing each human with the LLM, revealing whether the LLM can maintain annotation quality.

## Practical Interpretation Examples

### Example 1: Strong Performance
```
Winning Rate (ω): 0.850
Advantage Probability: 0.620
Tested against 20 human annotators
```

**Interpretation:** The LLM significantly outperforms or matches 17 out of 20 human annotators (85%). On average, the LLM agrees with the group 62% of the time, compared to individual humans. **Recommendation:** LLM is a viable substitute for human annotation on this task.

### Example 2: Moderate Performance
```
Winning Rate (ω): 0.600
Advantage Probability: 0.540
Tested against 15 human annotators
```

**Interpretation:** The LLM matches or exceeds 9 out of 15 humans (60%). However, the advantage probability is only slightly above chance. **Recommendation:** LLM could supplement human annotation but should not fully replace humans. Consider using LLM for initial screening with human verification.

### Example 3: Poor Performance
```
Winning Rate (ω): 0.300
Advantage Probability: 0.450
Tested against 25 human annotators
```

**Interpretation:** The LLM only matches 7.5 out of 25 humans (30%), and actually performs worse than random chance on average. **Recommendation:** LLM is not suitable for this annotation task. Human annotators are necessary.

## Factors Affecting Results

### 1. Task Difficulty
- **Subjective tasks** (e.g., sentiment, offensiveness): Lower ω expected due to genuine disagreement among humans
- **Objective tasks** (e.g., factual comparison): Higher ω expected

### 2. Annotation Quality
- If human annotators are noisy or inconsistent, LLM may achieve artificially high ω
- Check Inter-Rater Reliability (IRR) among humans first

### 3. Number of Annotators
- More annotators provide more robust estimates
- Minimum recommended: 10+ human annotators
- Fewer annotators may yield unstable results

### 4. Minimum Overlap Requirements
- `min_humans_per_instance=2`: Each item must be annotated by at least 2 humans
- `min_instances_per_human=30`: Each human must annotate at least 30 items
- Stricter requirements increase reliability but reduce sample size

## Comparison with Other Metrics

| Metric | What It Measures | ALT-TEST Advantage |
|--------|------------------|-------------------|
| **Accuracy** | Overall agreement | Ignores annotator variability |
| **Cohen's Kappa** | Pairwise agreement | Limited to 2 annotators |
| **Fleiss' Kappa** | Multi-rater agreement | Assumes equal reliability |
| **ALT-TEST** | LLM vs. human consensus | Accounts for annotator-specific reliability |

## Best Practices

1. **Always report both metrics**: Winning rate AND advantage probability provide complementary information

2. **Context matters**: Compare results to baseline human IRR using `.irr()` method

3. **Multiple seeds**: Run analysis with different random seeds to check stability

4. **Inspect failures**: When ω < 0.75, examine specific cases where LLM disagreed with humans

5. **Task-specific thresholds**: Adjust acceptance criteria based on application:
   - **High-stakes decisions** (medical, legal): Require ω ≥ 0.90
   - **Content moderation**: ω ≥ 0.75 acceptable
   - **Exploratory research**: ω ≥ 0.60 may suffice

## Reporting Results

When publishing ALT-TEST results, include:

```python
results = pairadigm_obj.alt_test(
    epsilon=0.1,
    q_fdr=0.05,
    min_humans_per_instance=2,
    min_instances_per_human=30
)

print(f"""
ALT-TEST Results for {pairadigm_obj.target_concept}:
- Winning Rate: {results[0]:.3f}
- Advantage Probability: {results[1]:.3f}
- Model: {pairadigm_obj.model_names[0]}
- Human Annotators: {len(pairadigm_obj.annotator_cols)}
- Total Comparisons: {len(pairadigm_obj.pairwise_df)}
""")
```

## Troubleshooting

**Q: Why is my winning rate 0.0?**
- LLM may be systematically biased compared to humans
- Check if LLM uses a different decision boundary
- Verify annotation format consistency (Text1/Text2 vs 0/1)

**Q: Why is advantage probability close to 0.5?**
- Task may be inherently ambiguous
- Check human IRR with `.irr()` method
- LLM and humans may use different reasoning

**Q: Can I test multiple LLMs?**
- Yes! Use `test_all_llms=True` to compare multiple models
- Lower ω values indicate which LLM is most human-like

---

# Dawid-Skene Reliability Interpretation Guide

The reliabilities from the Dawid-Skene model should be interpreted as follows:

**Basic Interpretation**
The Dawid-Skene reliability scores represent each annotator's estimated accuracy based on the diagonal of their confusion matrix. Higher scores indicate more reliable annotators.

Score Range: 0.0 to 1.0
- 1.0 = Perfect reliability (always agrees with the consensus)
- 0.5 = Random guessing (for binary classification)
- < 0.5 = Worse than random (systematically disagrees)

## What the Score Means
The reliability score answers: "What proportion of the time does this annotator agree with the estimated true labels?"

For binary classification (Text1 vs Text2):
- A score of 0.85 means the annotator correctly identifies the consensus label ~85% of the time
- A score of 0.60 indicates moderate reliability with more errors
- A score of 0.50 suggests the annotator provides no useful signal

## Context-Dependent Interpretation
Typical Reliability Ranges:
- 0.80-1.00: Excellent - Highly trustworthy annotator
- 0.70-0.80: Good - Reliable with some disagreement
- 0.60-0.70: Fair - Moderate reliability, use caution
- 0.50-0.60: Poor - Barely better than chance
- < 0.50: Problematic - May be confused or contrarian

## Important Considerations
1. Relative Comparison: Compare annotators within your dataset. A score of 0.75 might be excellent if all annotators score 0.60-0.80, but concerning if others score 0.85-0.95.
2. Task Difficulty: Harder tasks naturally produce lower reliability scores across all annotators.
3. LLM vs Human: When comparing LLM and human annotators, expect some variation. The key question is whether the LLM's reliability is comparable to or exceeds human annotators.
4. Statistical Confidence: The Dawid-Skene model estimates these weights from the data. With fewer annotations per annotator, estimates are less stable.

## Using in Your Workflow
The dawid_skene_annotator_ranking() method provides these scores in a ranked DataFrame, making it easy to:
- Identify which annotators (human or LLM) are most reliable
- Decide whether to exclude low-reliability annotators
- Validate that your LLM annotator performs comparably to humans

## What the Reliability Score Means with Ties
With 3 classes (including ties), a reliability score of 0.80 means:
- The annotator correctly identifies which of the three categories (Text1, Tie, Text2) matches the consensus 80% of the time
- This includes correctly identifying ties when they exist

**Important Note**
If ties are systematic (e.g., truly ambiguous pairs where no preference exists), they should definitely be modeled as a separate class. If ties are rare noise or annotator uncertainty, you might handle them differently depending on your research question.