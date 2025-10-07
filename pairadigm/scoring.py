"""
Bradley-Terry model scoring and analysis functions.
"""

import pandas as pd
import numpy as np
import choix
from typing import Dict, Optional


def bradley_terry_scores(
    original_df: pd.DataFrame,
    row_id: str,
    pairwise_df: pd.DataFrame,
    normalize: bool = False
) -> pd.DataFrame:
    """
    Compute Bradley-Terry scores from pairwise comparison results.
    
    The Bradley-Terry model estimates the relative strength/quality of items
    based on pairwise comparisons. Higher scores indicate items that are
    preferred more often.
    
    Parameters
    ----------
    original_df : pd.DataFrame
        Original DataFrame with item metadata
    row_id : str
        Column name for unique item identifiers
    pairwise_df : pd.DataFrame
        DataFrame with pairwise comparison results
        Must have columns: ['item1', 'item2', 'decision']
    normalize : bool, default=False
        Whether to normalize scores to [0, 1] range
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added 'Bradley_Terry_Score' column
        
    Raises
    ------
    ValueError
        If no valid comparisons found or required columns missing
        
    Examples
    --------
    >>> items_df = pd.DataFrame({'id': [1, 2, 3]})
    >>> pairs_df = pd.DataFrame({
    ...     'item1': [1, 2], 
    ...     'item2': [2, 3], 
    ...     'decision': ['Text1', 'Text1']
    ... })
    >>> scored_df = bradley_terry_scores(items_df, 'id', pairs_df)
    >>> 'Bradley_Terry_Score' in scored_df.columns
    True
    
    Notes
    -----
    Uses the Iterative Luce Spectral Ranking (ILSR) algorithm from the
    choix library with L2 regularization (alpha=0.1) for stability.
    """
    # Validate inputs
    if row_id not in original_df.columns:
        raise ValueError(f"Column '{row_id}' not found in original_df")
    
    required_cols = ['item1', 'item2', 'decision']
    missing_cols = [col for col in required_cols if col not in pairwise_df.columns]
    if missing_cols:
        raise ValueError(f"pairwise_df missing required columns: {missing_cols}")
    
    # Filter out invalid decisions
    valid_df = pairwise_df[pairwise_df['decision'].isin(['Text1', 'Text2'])].copy()
    
    if valid_df.empty:
        raise ValueError("No valid comparisons found in pairwise_df.")
    
    print(f"Using {len(valid_df)} valid comparisons out of {len(pairwise_df)} total")
    
    # Create item index mapping
    item_to_idx = {item: idx for idx, item in enumerate(original_df[row_id].tolist())}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    n_items = len(item_to_idx)
    
    # Prepare comparisons for Bradley-Terry model
    # Format: list of (winner_idx, loser_idx) tuples
    comparisons = []
    
    for _, row in valid_df.iterrows():
        item1_idx = item_to_idx[row['item1']]
        item2_idx = item_to_idx[row['item2']]
        decision = row['decision']
        
        if decision == 'Text1':
            # item1 wins
            comparisons.append((item1_idx, item2_idx))
        elif decision == 'Text2':
            # item2 wins
            comparisons.append((item2_idx, item1_idx))
    
    if not comparisons:
        raise ValueError("No valid comparisons to compute Bradley-Terry scores.")
    
    print(f"Fitting Bradley-Terry model on {len(comparisons)} comparisons...")
    
    # Fit Bradley-Terry model using ILSR algorithm
    # alpha=0.1 provides L2 regularization for stability
    try:
        bt_scores = choix.ilsr_pairwise(
            n_items=n_items,
            data=comparisons,
            alpha=0.1
        )
    except Exception as e:
        raise RuntimeError(f"Failed to fit Bradley-Terry model: {e}")
    
    # Normalize scores to [0, 1] if requested
    if normalize:
        bt_min = bt_scores.min()
        bt_max = bt_scores.max()
        if bt_max > bt_min:
            bt_scores = (bt_scores - bt_min) / (bt_max - bt_min)
        else:
            # All scores are equal
            bt_scores = np.full_like(bt_scores, 0.5)
    
    # Add scores to original DataFrame
    original_df = original_df.copy()
    original_df['Bradley_Terry_Score'] = [
        bt_scores[item_to_idx[item_id]] 
        for item_id in original_df[row_id]
    ]
    
    # Print summary statistics
    scores = original_df['Bradley_Terry_Score']
    print(f"Bradley-Terry scores computed:")
    print(f"  Mean: {scores.mean():.3f}")
    print(f"  Std:  {scores.std():.3f}")
    print(f"  Range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    return original_df


def summarize_scores(
    df: pd.DataFrame,
    text_col: str,
    score_col: str = 'Bradley_Terry_Score'
) -> Dict:
    """
    Summarize Bradley-Terry scores with statistics and examples.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with scores
    text_col : str
        Column name for text that was scored
    score_col : str, default='Bradley_Terry_Score'
        Column name for scores
        
    Returns
    -------
    Dict
        Summary statistics including mean, median, std, quartiles
        
    Examples
    --------
    >>> summary = summarize_scores(scored_df, text_col='text')
    >>> print(f"Mean score: {summary['mean']:.3f}")
    """
    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found in DataFrame")
    
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")
    
    scores = df[score_col]
    
    # Calculate statistics
    summary = {
        'count': scores.count(),
        'mean': scores.mean(),
        'median': scores.median(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        'q25': scores.quantile(0.25),
        'q75': scores.quantile(0.75),
        'iqr': scores.quantile(0.75) - scores.quantile(0.25)
    }
    
    # Print detailed summary
    print("\n" + "="*60)
    print("BRADLEY-TERRY SCORE SUMMARY")
    print("="*60)
    print(f"Count:  {summary['count']}")
    print(f"Mean:   {summary['mean']:.3f}")
    print(f"Median: {summary['median']:.3f}")
    print(f"Std:    {summary['std']:.3f}")
    print(f"Range:  [{summary['min']:.3f}, {summary['max']:.3f}]")
    print(f"\nQuartiles:")
    print(f"  25th: {summary['q25']:.3f}")
    print(f"  50th: {summary['median']:.3f}")
    print(f"  75th: {summary['q75']:.3f}")
    
    # Show examples
    df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    
    top_item = df_sorted.iloc[0]
    bottom_item = df_sorted.iloc[-1]
    
    print(f"\n{'='*60}")
    print(f"HIGHEST SCORING ITEM (score: {top_item[score_col]:.3f})")
    print(f"{'='*60}")
    print(f"{top_item[text_col][:200]}...")
    
    print(f"\n{'='*60}")
    print(f"LOWEST SCORING ITEM (score: {bottom_item[score_col]:.3f})")
    print(f"{'='*60}")
    print(f"{bottom_item[text_col][:200]}...")
    print("="*60 + "\n")
    
    return summary


def score_confidence_intervals(
    df: pd.DataFrame,
    score_col: str = 'Bradley_Terry_Score',
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Calculate confidence intervals for scores using bootstrap.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with scores
    score_col : str
        Column with scores
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with CI columns added
        
    Notes
    -----
    This is a simplified version. For production use, consider the
    bootstrap_confidence_interval function in validation.py for
    more rigorous uncertainty quantification.
    """
    # Simple approximation using normal distribution
    scores = df[score_col]
    alpha = 1 - confidence_level
    z_score = 1.96  # For 95% CI
    
    # Assuming approximate normality (valid for large samples)
    margin_of_error = z_score * scores.std() / np.sqrt(len(scores))
    
    df = df.copy()
    df[f'{score_col}_CI_lower'] = scores - margin_of_error
    df[f'{score_col}_CI_upper'] = scores + margin_of_error
    
    return df


def rank_items(
    df: pd.DataFrame,
    score_col: str = 'Bradley_Terry_Score',
    ascending: bool = False
) -> pd.DataFrame:
    """
    Rank items by their scores.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with scores
    score_col : str
        Column with scores
    ascending : bool
        If True, rank lowest scores first
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'rank' column
    """
    df = df.copy()
    df['rank'] = df[score_col].rank(ascending=ascending, method='min')
    df = df.sort_values('rank')
    return df


def compare_score_distributions(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    score_col: str = 'Bradley_Terry_Score',
    label1: str = "Distribution 1",
    label2: str = "Distribution 2"
) -> Dict:
    """
    Compare two score distributions statistically.
    
    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames with scores to compare
    score_col : str
        Column with scores
    label1, label2 : str
        Labels for the distributions
        
    Returns
    -------
    Dict
        Statistical comparison results
    """
    from scipy import stats
    
    scores1 = df1[score_col].dropna()
    scores2 = df2[score_col].dropna()
    
    # T-test
    t_stat, t_pval = stats.ttest_ind(scores1, scores2)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pval = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(scores1, scores2)
    
    results = {
        'label1': label1,
        'label2': label2,
        'mean1': scores1.mean(),
        'mean2': scores2.mean(),
        'mean_diff': scores1.mean() - scores2.mean(),
        'std1': scores1.std(),
        'std2': scores2.std(),
        't_statistic': t_stat,
        't_pvalue': t_pval,
        'u_statistic': u_stat,
        'u_pvalue': u_pval,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'significant_difference': t_pval < 0.05
    }
    
    print(f"\n{'='*60}")
    print(f"COMPARING: {label1} vs {label2}")
    print(f"{'='*60}")
    print(f"{label1}:")
    print(f"  Mean: {results['mean1']:.3f} (SD: {results['std1']:.3f})")
    print(f"{label2}:")
    print(f"  Mean: {results['mean2']:.3f} (SD: {results['std2']:.3f})")
    print(f"\nDifference: {results['mean_diff']:.3f}")
    print(f"\nStatistical Tests:")
    print(f"  t-test:  t={t_stat:.3f}, p={t_pval:.4f}")
    print(f"  U-test:  U={u_stat:.1f}, p={u_pval:.4f}")
    print(f"  KS-test: D={ks_stat:.3f}, p={ks_pval:.4f}")
    print(f"\nSignificant difference: {results['significant_difference']}")
    print("="*60 + "\n")
    
    return results