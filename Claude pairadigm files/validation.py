"""
Validation functions for checking annotation quality, transitivity, and inter-annotator agreement.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, kendalltau


def check_transitivity(
    pairwise_df: pd.DataFrame,
    annotator_name: str = "Annotator"
) -> Dict:
    """
    Check transitivity of pairwise comparisons: if A > B and B > C, then A > C.
    
    Parameters
    ----------
    pairwise_df : pd.DataFrame
        DataFrame with columns: item1, item2, decision
    annotator_name : str
        Name of annotator for reporting
        
    Returns
    -------
    Dict
        Transitivity metrics including:
        - transitivity_rate: proportion of transitive triples
        - violations: list of intransitive triples
        - total_triples: number of triples checked
    """
    # Build preference graph: winner -> list of losers
    preferences = defaultdict(set)
    
    for _, row in pairwise_df.iterrows():
        if row['decision'] == 'Text1':
            preferences[row['item1']].add(row['item2'])
        elif row['decision'] == 'Text2':
            preferences[row['item2']].add(row['item1'])
    
    # Check all triples (A, B, C)
    items = list(preferences.keys())
    violations = []
    total_triples = 0
    
    for a in items:
        for b in preferences[a]:  # A > B
            for c in preferences[b]:  # B > C
                total_triples += 1
                # Check if A > C (transitivity)
                if c not in preferences[a]:
                    # Check if we have explicit evidence of C > A (violation)
                    if a in preferences[c]:
                        violations.append((a, b, c))
    
    transitivity_rate = 1.0 - (len(violations) / total_triples) if total_triples > 0 else 1.0
    
    return {
        'annotator': annotator_name,
        'transitivity_rate': transitivity_rate,
        'violations': violations,
        'total_triples': total_triples,
        'num_violations': len(violations)
    }


def compare_annotators(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    annotator1_name: str = "Annotator 1",
    annotator2_name: str = "Annotator 2"
) -> Dict:
    """
    Compare two annotators' pairwise judgments.
    
    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames with columns: item1, item2, decision
    annotator1_name, annotator2_name : str
        Names for reporting
        
    Returns
    -------
    Dict
        Agreement metrics including Cohen's Kappa
    """
    # Merge on item pairs (order-agnostic)
    df1_sorted = df1.copy()
    df2_sorted = df2.copy()
    
    # Create canonical pair representation
    for df in [df1_sorted, df2_sorted]:
        df['pair'] = df.apply(
            lambda row: tuple(sorted([row['item1'], row['item2']])),
            axis=1
        )
    
    # Merge on pairs
    merged = df1_sorted.merge(
        df2_sorted,
        on='pair',
        suffixes=('_1', '_2'),
        how='inner'
    )
    
    if len(merged) == 0:
        raise ValueError("No overlapping pairs found between annotators")
    
    # Normalize decisions to comparable format
    def normalize_decision(row, suffix):
        decision = row[f'decision{suffix}']
        item1_orig = row[f'item1{suffix}']
        item1_sorted = row['pair'][0]
        
        # If original item1 matches sorted item1, keep decision as is
        # Otherwise, flip it
        if item1_orig == item1_sorted:
            return 1 if decision == 'Text1' else 2
        else:
            return 2 if decision == 'Text1' else 1
    
    merged['decision_norm_1'] = merged.apply(lambda r: normalize_decision(r, '_1'), axis=1)
    merged['decision_norm_2'] = merged.apply(lambda r: normalize_decision(r, '_2'), axis=1)
    
    # Calculate agreement
    agreements = (merged['decision_norm_1'] == merged['decision_norm_2']).sum()
    agreement_rate = agreements / len(merged)
    
    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(
        merged['decision_norm_1'],
        merged['decision_norm_2']
    )
    
    # Calculate which annotator shows more consistency
    disagreements = merged[merged['decision_norm_1'] != merged['decision_norm_2']]
    
    return {
        'annotator1': annotator1_name,
        'annotator2': annotator2_name,
        'total_pairs': len(merged),
        'agreements': agreements,
        'disagreements': len(disagreements),
        'agreement_rate': agreement_rate,
        'cohens_kappa': kappa,
        'kappa_interpretation': interpret_kappa(kappa)
    }


def interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's Kappa value."""
    if kappa < 0:
        return "Poor (worse than chance)"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


def compute_score_correlation(
    scores1: pd.Series,
    scores2: pd.Series,
    method: str = 'spearman'
) -> Dict:
    """
    Compute correlation between two sets of scores.
    
    Parameters
    ----------
    scores1, scores2 : pd.Series
        Score series to compare
    method : str
        'spearman' or 'kendall'
        
    Returns
    -------
    Dict
        Correlation coefficient and p-value
    """
    if method == 'spearman':
        corr, pval = spearmanr(scores1, scores2)
    elif method == 'kendall':
        corr, pval = kendalltau(scores1, scores2)
    else:
        raise ValueError("method must be 'spearman' or 'kendall'")
    
    return {
        'method': method,
        'correlation': corr,
        'p_value': pval,
        'significant': pval < 0.05
    }


def alternate_annotator_test(
    llm_pairwise_df: pd.DataFrame,
    human_pairwise_df: pd.DataFrame,
    llm_scores_df: pd.DataFrame,
    row_id: str,
    alpha: float = 0.05
) -> Dict:
    """
    Perform the Alternative Annotator Test to determine if LLM can replace human annotators.
    
    The test checks:
    1. Agreement rate between LLM and human annotations
    2. Inter-annotator reliability (Cohen's Kappa)
    3. Score correlation (if human scores available)
    4. Transitivity of both annotators
    
    Parameters
    ----------
    llm_pairwise_df : pd.DataFrame
        LLM pairwise annotations
    human_pairwise_df : pd.DataFrame
        Human pairwise annotations
    llm_scores_df : pd.DataFrame
        LLM Bradley-Terry scores
    row_id : str
        Column for item IDs
    alpha : float
        Significance level
        
    Returns
    -------
    Dict
        Test results and recommendation
    """
    results = {}
    
    # 1. Check transitivity for both annotators
    llm_transitivity = check_transitivity(llm_pairwise_df, "LLM")
    human_transitivity = check_transitivity(human_pairwise_df, "Human")
    
    results['llm_transitivity'] = llm_transitivity['transitivity_rate']
    results['human_transitivity'] = human_transitivity['transitivity_rate']
    
    # 2. Compare annotator agreement
    comparison = compare_annotators(
        llm_pairwise_df,
        human_pairwise_df,
        "LLM",
        "Human"
    )
    
    results['agreement_rate'] = comparison['agreement_rate']
    results['cohens_kappa'] = comparison['cohens_kappa']
    results['kappa_interpretation'] = comparison['kappa_interpretation']
    
    # 3. Check if human scores are available for correlation
    if 'Bradley_Terry_Score' in human_pairwise_df.columns:
        # Merge scores
        llm_scores = llm_scores_df[[row_id, 'Bradley_Terry_Score']].rename(
            columns={'Bradley_Terry_Score': 'llm_score'}
        )
        human_scores = human_pairwise_df[[row_id, 'Bradley_Terry_Score']].rename(
            columns={'Bradley_Terry_Score': 'human_score'}
        )
        merged_scores = llm_scores.merge(human_scores, on=row_id)
        
        correlation = compute_score_correlation(
            merged_scores['llm_score'],
            merged_scores['human_score']
        )
        results['score_correlation'] = correlation['correlation']
        results['score_p_value'] = correlation['p_value']
    else:
        results['score_correlation'] = None
    
    # 4. Make recommendation
    recommendation = make_recommendation(results, alpha)
    results['recommendation'] = recommendation
    results['pass_test'] = recommendation.startswith("PASS")
    
    return results


def make_recommendation(results: Dict, alpha: float) -> str:
    """
    Make recommendation based on Alternative Annotator Test results.
    
    Criteria for passing:
    - Cohen's Kappa >= 0.60 (moderate agreement)
    - Agreement rate >= 0.70
    - LLM transitivity >= 0.90
    - Score correlation >= 0.70 (if available)
    """
    kappa = results['cohens_kappa']
    agreement = results['agreement_rate']
    llm_trans = results['llm_transitivity']
    score_corr = results.get('score_correlation')
    
    issues = []
    
    if kappa < 0.60:
        issues.append(f"Cohen's Kappa too low ({kappa:.3f} < 0.60)")
    
    if agreement < 0.70:
        issues.append(f"Agreement rate too low ({agreement:.2%} < 70%)")
    
    if llm_trans < 0.90:
        issues.append(f"LLM transitivity too low ({llm_trans:.2%} < 90%)")
    
    if score_corr is not None and score_corr < 0.70:
        issues.append(f"Score correlation too low ({score_corr:.3f} < 0.70)")
    
    if not issues:
        return (
            "PASS: LLM can be used as an alternative annotator. "
            "High agreement and reliability metrics indicate the LLM produces "
            "annotations comparable to human annotators."
        )
    else:
        return (
            f"FAIL: LLM should NOT replace human annotators. Issues: {'; '.join(issues)}. "
            "Consider refining prompts or using LLM as a supplementary annotator only."
        )


def calculate_krippendorff_alpha(
    annotations_df: pd.DataFrame,
    annotator_col: str,
    item_col: str,
    decision_col: str
) -> float:
    """
    Calculate Krippendorff's Alpha for multiple annotators.
    
    Parameters
    ----------
    annotations_df : pd.DataFrame
        Long-format dataframe with one row per annotation
    annotator_col : str
        Column indicating annotator ID
    item_col : str
        Column indicating item/pair being annotated
    decision_col : str
        Column with annotation decision
        
    Returns
    -------
    float
        Krippendorff's Alpha coefficient
    """
    # Create reliability matrix (annotators x items)
    pivot = annotations_df.pivot(
        index=annotator_col,
        columns=item_col,
        values=decision_col
    )
    
    # Convert to numpy for calculation
    data = pivot.values
    n_annotators, n_items = data.shape
    
    # Calculate observed disagreement
    observed_disagreement = 0
    n_comparisons = 0
    
    for j in range(n_items):
        values = data[:, j]
        values = values[~pd.isna(values)]
        m = len(values)
        if m < 2:
            continue
        for i1 in range(m):
            for i2 in range(i1 + 1, m):
                observed_disagreement += (values[i1] != values[i2])
                n_comparisons += 1
    
    observed_disagreement /= n_comparisons if n_comparisons > 0 else 1
    
    # Calculate expected disagreement
    all_values = data.flatten()
    all_values = all_values[~pd.isna(all_values)]
    unique_values = np.unique(all_values)
    
    expected_disagreement = 0
    n_total = len(all_values)
    
    for v1 in unique_values:
        for v2 in unique_values:
            if v1 != v2:
                p1 = np.sum(all_values == v1) / n_total
                p2 = np.sum(all_values == v2) / n_total
                expected_disagreement += p1 * p2
    
    # Krippendorff's Alpha
    if expected_disagreement == 0:
        return 1.0
    
    alpha = 1 - (observed_disagreement / expected_disagreement)
    return alpha


def bootstrap_confidence_interval(
    pairwise_df: pd.DataFrame,
    row_id: str,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict:
    """
    Calculate bootstrap confidence intervals for Bradley-Terry scores.
    
    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise comparison data
    row_id : str
        Column for item IDs
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
        
    Returns
    -------
    Dict
        Mean scores and confidence intervals for each item
    """
    from .scoring import bradley_terry_scores
    
    # Get unique items
    items = pd.concat([
        pairwise_df['item1'],
        pairwise_df['item2']
    ]).unique()
    
    # Bootstrap sampling
    bootstrap_scores = defaultdict(list)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        sample_df = pairwise_df.sample(n=len(pairwise_df), replace=True)
        
        # Create temporary dataframe with items
        temp_df = pd.DataFrame({row_id: items})
        
        try:
            # Compute scores for this sample
            scored_df = bradley_terry_scores(
                original_df=temp_df,
                row_id=row_id,
                pairwise_df=sample_df,
                normalize=True
            )
            
            # Store scores
            for _, row in scored_df.iterrows():
                bootstrap_scores[row[row_id]].append(row['Bradley_Terry_Score'])
        except:
            continue  # Skip failed bootstrap samples
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    results = {}
    
    for item, scores in bootstrap_scores.items():
        if len(scores) > 0:
            results[item] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'ci_lower': np.percentile(scores, 100 * alpha / 2),
                'ci_upper': np.percentile(scores, 100 * (1 - alpha / 2))
            }
    
    return results