import warnings
import numpy as np
import pandas as pd
import pytest
from pairadigm.validation import bradley_terry_alt_test

def make_simulated_pairwise_df(n_items=30):
    items = [f"item_{i}" for i in range(n_items)]
    # Latent scores for annotators
    # Human 1, 2, 3 have similar scores, LLM has slightly different scores
    np.random.seed(42)
    latent_scores = {
        "human_1": {items[i]: float(i) for i in range(n_items)},
        "human_2": {items[i]: float(i + np.random.normal(0, 2)) for i in range(n_items)},
        "human_3": {items[i]: float(i + np.random.normal(0, 2)) for i in range(n_items)},
        "llm": {items[i]: float(i + np.random.normal(0, 4)) for i in range(n_items)},
    }
    
    rows = []
    for i in range(n_items):
        for j in range(i + 1, n_items):
            item1 = items[i]
            item2 = items[j]
            row = {"item1": item1, "item2": item2}
            for ann in ["human_1", "human_2", "human_3", "llm"]:
                s1 = latent_scores[ann][item1]
                s2 = latent_scores[ann][item2]
                if s1 > s2:
                    row[ann] = "Text1"
                elif s1 < s2:
                    row[ann] = "Text2"
                else:
                    row[ann] = "Tie"
            rows.append(row)
    return pd.DataFrame(rows)

def test_hbt_basic():
    df = make_simulated_pairwise_df()
    
    # Test Wilcoxon (default)
    res = bradley_terry_alt_test(
        pairwise_df=df,
        annotator_cols=["human_1", "human_2", "human_3"],
        llm_decision_col="llm",
        metric="spearman",
        test_type="wilcoxon",
        verbose=True
    )
    
    # Assert return keys exist and are correct
    assert "human_human_correlations" in res
    assert "llm_human_correlations" in res
    assert "human_human_mean" in res
    assert "llm_human_mean" in res
    assert "test_statistic" in res
    assert "p_value" in res
    assert "advantage_probability" in res
    assert "interpretation" in res
    assert res["test_type"] == "wilcoxon"
    assert len(res["human_human_correlations"]) == 3
    assert len(res["llm_human_correlations"]) == 3

def test_hbt_paired_ttest():
    df = make_simulated_pairwise_df()
    
    # Test paired t-test
    res = bradley_terry_alt_test(
        pairwise_df=df,
        annotator_cols=["human_1", "human_2", "human_3"],
        llm_decision_col="llm",
        metric="kendall",
        test_type="paired_ttest",
        verbose=False
    )
    assert res["test_type"] == "paired_ttest"
    assert "p_value" in res

def test_hbt_bootstrapping():
    df = make_simulated_pairwise_df()
    
    res = bradley_terry_alt_test(
        pairwise_df=df,
        annotator_cols=["human_1", "human_2", "human_3"],
        llm_decision_col="llm",
        metric="spearman",
        n_bootstraps=100,
        random_seed=42,
        verbose=False
    )
    
    assert "bootstrap_p_value" in res
    assert "bootstrap_diff_ci" in res
    assert "bootstrap_adv_prob_ci" in res
    assert len(res["bootstrap_diffs"]) == 100

def test_hbt_deprecation_warnings():
    df = make_simulated_pairwise_df()
    
    # Test deprecated metric mapping
    with pytest.warns(UserWarning, match="Metric .* is deprecated"):
        res_pearson = bradley_terry_alt_test(
            pairwise_df=df,
            annotator_cols=["human_1", "human_2", "human_3"],
            llm_decision_col="llm",
            metric="pearson"
        )
    # Pearson should have been mapped to spearman
    assert "spearman" not in res_pearson  # since it's a dict for spearman directly (single metric returns dict directly, not dict of dicts unless 'all')
    
    # Test deprecated test types
    with pytest.warns(DeprecationWarning, match="test_type='mannwhitney' is deprecated"):
        res_mw = bradley_terry_alt_test(
            pairwise_df=df,
            annotator_cols=["human_1", "human_2", "human_3"],
            llm_decision_col="llm",
            test_type="mannwhitney"
        )
    assert res_mw["test_type"] == "wilcoxon"
    
    with pytest.warns(DeprecationWarning, match="test_type='ttest' is deprecated"):
        res_tt = bradley_terry_alt_test(
            pairwise_df=df,
            annotator_cols=["human_1", "human_2", "human_3"],
            llm_decision_col="llm",
            test_type="ttest"
        )
    assert res_tt["test_type"] == "paired_ttest"

def test_hbt_all_metrics():
    df = make_simulated_pairwise_df()
    res = bradley_terry_alt_test(
        pairwise_df=df,
        annotator_cols=["human_1", "human_2", "human_3"],
        llm_decision_col="llm",
        metric="all"
    )
    assert "spearman" in res
    assert "kendall" in res
    assert "human_human_mean" in res["spearman"]
    assert "human_human_mean" in res["kendall"]

def test_hbt_min_common_items():
    df = make_simulated_pairwise_df(n_items=30)
    # If min_common_items is larger than 30, it should warn and return empty
    with pytest.warns(UserWarning, match="No valid pivots with sufficient overlap"):
        res = bradley_terry_alt_test(
            pairwise_df=df,
            annotator_cols=["human_1", "human_2", "human_3"],
            llm_decision_col="llm",
            min_common_items=40
        )
    assert res == {}
