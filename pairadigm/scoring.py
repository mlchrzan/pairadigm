"""
Scoring functions for Pairadigm.

Provides Bradley-Terry and Davidson model fitting, score normalisation,
score column name resolution, and summary statistics.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import choix
import numpy as np
import pandas as pd
from scipy.optimize import minimize

WIN_1_VALUES = ["Text1", 0]
WIN_2_VALUES = ["Text2", 1]
TIE_VALUES = ["Tie", "tie", 2, 0.5]


# ---------------------------------------------------------------------------
# Column-name helpers
# ---------------------------------------------------------------------------

def get_score_col_name(
    decision_col: str = "decision",
    split: Optional[str] = None,
    model_name: str = "Bradley_Terry",
) -> str:
    """
    Return the canonical score column name produced by :func:`score_items`.

    Parameters
    ----------
    decision_col : str
        Name of the decision column (e.g., ``'decision'`` or
        ``'decision_gemini-2.0-flash-exp'``).
    split : str or None
        ``'full'`` or ``'split'`` (only relevant when
        ``generate_pairings(make_splits=True)`` was used).  ``None`` returns
        the single-score column name (no-split case).
    model_name : str
        Model prefix used in the column name (``'Bradley_Terry'`` or
        ``'Davidson'``).

    Returns
    -------
    str
        The column name as it would appear in ``scored_df``.
    """
    model_prefix = model_name.replace("-", "_")
    decision_suffix = (
        "" if decision_col == "decision"
        else f"_{decision_col.replace('decision_', '')}"
    )
    if split is None:
        return f"{model_prefix}_Score{decision_suffix}"
    return f"{model_prefix}_Score_{split}{decision_suffix}"


# ---------------------------------------------------------------------------
# Internal model fitters
# ---------------------------------------------------------------------------

def _fit_davidson_model(
    valid_df: pd.DataFrame,
    item_to_idx: dict,
    n_items: int,
    decision_col: str,
    model_label: str = "",
) -> Tuple[np.ndarray, str, list, float]:
    """
    Fit a Davidson model on a (possibly filtered) comparison DataFrame.

    Returns
    -------
    tuple[np.ndarray, str, list, float]
        ``(raw_scores, model_name_string, comparisons_list, estimated_tau)``
        where ``comparisons_list`` is an empty list for Davidson.
    """
    # Map items to integer ids for the subset of valid_df
    i_idx = valid_df["item1"].map(item_to_idx).to_numpy(dtype=np.int64)
    j_idx = valid_df["item2"].map(item_to_idx).to_numpy(dtype=np.int64)

    # Encode outcomes: 0=Text1, 1=Text2, 2=Tie
    y = np.where(
        valid_df[decision_col].isin(WIN_1_VALUES),
        0,
        np.where(valid_df[decision_col].isin(WIN_2_VALUES), 1, 2)
    ).astype(np.int64)

    # Optimization speedup: aggregate duplicate (i, j, outcome) rows
    packed = np.stack([i_idx, j_idx, y], axis=1)
    uniq, counts = np.unique(packed, axis=0, return_counts=True)
    i_u = uniq[:, 0]
    j_u = uniq[:, 1]
    y_u = uniq[:, 2]
    w_u = counts.astype(np.float64)

    n_free = n_items - 1
    # We estimate tie propensity (tau); add one parameter for log_tau
    x0 = np.zeros(n_free + 1, dtype=np.float64)
    x0[-1] = 0.0  # log_tau = 0 -> tau = 1

    eps = 1e-12

    def _unpack(params):
        scores = np.zeros(n_items, dtype=np.float64)
        if n_free > 0:
            scores[1:] = params[:n_free]
        tau = np.exp(params[-1])
        return scores, tau

    def objective_and_grad(params):
        scores, tau = _unpack(params)

        si = scores[i_u]
        sj = scores[j_u]

        # Davidson probabilities
        a = np.exp(si)
        b = np.exp(sj)
        g = np.exp(0.5 * (si + sj))  # sqrt(a*b)
        D = a + b + 2.0 * tau * g

        p1 = a / D
        p2 = b / D
        pt = (2.0 * tau * g) / D

        p1 = np.clip(p1, eps, 1.0)
        p2 = np.clip(p2, eps, 1.0)
        pt = np.clip(pt, eps, 1.0)

        p_obs = np.where(y_u == 0, p1, np.where(y_u == 1, p2, pt))
        nll = -(w_u * np.log(p_obs)).sum()

        # Analytic gradient
        grad_scores = np.zeros(n_items, dtype=np.float64)

        # Gradients of log probabilities
        dlogp1_dsi = 1.0 - (a + tau * g) / D
        dlogp1_dsj = - (b + tau * g) / D

        dlogp2_dsi = - (a + tau * g) / D
        dlogp2_dsj = 1.0 - (b + tau * g) / D

        dlogpt_dsi = 0.5 - (a + tau * g) / D
        dlogpt_dsj = 0.5 - (b + tau * g) / D

        dlogp_dsi = np.where(y_u == 0, dlogp1_dsi, np.where(y_u == 1, dlogp2_dsi, dlogpt_dsi))
        dlogp_dsj = np.where(y_u == 0, dlogp1_dsj, np.where(y_u == 1, dlogp2_dsj, dlogpt_dsj))

        contrib_i = -w_u * dlogp_dsi
        contrib_j = -w_u * dlogp_dsj
        np.add.at(grad_scores, i_u, contrib_i)
        np.add.at(grad_scores, j_u, contrib_j)

        if n_free > 0:
            grad_free = grad_scores[1:]
        else:
            grad_free = np.array([], dtype=np.float64)

        # Gradient for log_tau
        dlogp_dlogtau = np.where(y_u == 2, 1.0 - pt, -pt)
        grad_logtau = -(w_u * dlogp_dlogtau).sum()
        grad = np.concatenate([grad_free, np.array([grad_logtau])])

        return nll, grad

    print(f"[{model_label}] Fitting Davidson model (L-BFGS-B)...")
    
    result = minimize(
        fun=lambda p: objective_and_grad(p)[0],
        x0=x0,
        jac=lambda p: objective_and_grad(p)[1],
        method="L-BFGS-B",
        options={"maxiter": 1000},
    )

    scores_opt, tau_opt = _unpack(result.x)
    
    if not result.success:
        warnings.warn(f"Davidson model optimization didn't cleanly converge: {result.message}")

    return scores_opt, "Davidson", [], tau_opt


def _fit_bt_model(
    valid_df: pd.DataFrame,
    item_to_idx: dict,
    n_items: int,
    decision_col: str,
    model_label: str = "",
) -> Tuple[np.ndarray, str, list]:
    """
    Fit a Bradley-Terry model on a (possibly filtered) comparison DataFrame.

    Returns
    -------
    tuple[np.ndarray, str, list]
        ``(raw_scores, model_name_string, comparisons_list)``
        where ``comparisons_list`` is a list of ``(winner_idx, loser_idx)``
        tuples.
    """
    comparisons = []
    for item1, item2, decision in zip(valid_df["item1"], valid_df["item2"], valid_df[decision_col]):
        item1_idx = item_to_idx[item1]
        item2_idx = item_to_idx[item2]

        if decision in WIN_1_VALUES:
            comparisons.append((item1_idx, item2_idx))
        elif decision in WIN_2_VALUES:
            comparisons.append((item2_idx, item1_idx))

    if not comparisons:
        raise ValueError("No valid comparisons to compute Bradley-Terry scores.")

    print(f"[{model_label}] Fitting Bradley-Terry model...")
    scores = choix.ilsr_pairwise(n_items, comparisons, alpha=0.1)
    return scores, "Bradley-Terry", comparisons


def _compute_bt_se(
    log_strengths: np.ndarray,
    comparisons: list,
    n_items: int,
    normalization_scale: Union[str, Tuple[float, float]],
) -> np.ndarray:
    """
    Approximate SE for normalised BT scores via the Fisher Information Matrix
    diagonal and the delta method.

    Parameters
    ----------
    log_strengths : np.ndarray, shape (n_items,)
        Raw log-strength estimates from ``choix.ilsr_pairwise``.
    comparisons : list of (int, int)
        ``(winner_idx, loser_idx)`` pairs used to fit the model.
    n_items : int
    normalization_scale : str or tuple
        Same normalisation applied to the scores.

    Returns
    -------
    np.ndarray, shape (n_items,)
        Approximate SE on the **normalised** score scale.  NaN where
        insufficient data exist.
    """
    strengths = np.exp(log_strengths)  # exponentiate log-strengths

    # --- Count comparisons per pair ---
    n_matrix = np.zeros((n_items, n_items))
    for (i, j) in comparisons:
        n_matrix[i, j] += 1
        n_matrix[j, i] += 1  # symmetric total count

    # --- Diagonal of FIM in log-strength parameterisation ---
    # I(beta_i) = sum_j n_ij * p_ij * (1 - p_ij)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_matrix = strengths[:, None] / (strengths[:, None] + strengths[None, :])
    np.fill_diagonal(p_matrix, 0.0)
    fim_diag = (n_matrix * p_matrix * (1.0 - p_matrix)).sum(axis=1)

    # SE on log-strength
    with np.errstate(divide="ignore", invalid="ignore"):
        se_log = np.where(fim_diag > 0, 1.0 / np.sqrt(fim_diag), np.nan)

    # --- Propagate through normalisation via delta method ---
    # For zero-to-one:  norm(s) = (s - s_min) / (s_max - s_min)
    # d(norm)/d(log_strength) = s / (s_max - s_min)
    # SE_norm ≈ abs(d(norm)/d(log_strength)) * SE_log
    if isinstance(normalization_scale, tuple):
        scale_min, scale_max = normalization_scale
        raw_range = strengths.max() - strengths.min()
        if raw_range == 0:
            return np.zeros(n_items)
        jacobian = strengths * (scale_max - scale_min) / raw_range
    elif normalization_scale == "zero-to-one":
        raw_range = strengths.max() - strengths.min()
        if raw_range == 0:
            return np.zeros(n_items)
        jacobian = strengths / raw_range
    elif normalization_scale == "negative-one-to-one":
        raw_range = strengths.max() - strengths.min()
        if raw_range == 0:
            return np.zeros(n_items)
        jacobian = 2.0 * strengths / raw_range
    else:  # 'none'
        jacobian = strengths  # SE on raw log-strength scale

    return np.abs(jacobian) * se_log


def _normalize_bt_scores(
    scores: np.ndarray,
    normalization_scale: Union[str, Tuple[float, float]],
) -> np.ndarray:
    """Apply normalisation to a raw BT/Davidson score array."""
    if isinstance(normalization_scale, tuple):
        if len(normalization_scale) != 2:
            raise ValueError(
                "normalization_scale tuple must have exactly 2 elements: (min, max)"
            )
        scale_min, scale_max = normalization_scale
        if scale_min >= scale_max:
            raise ValueError("normalization_scale tuple requires min < max")
        return (
            scale_min
            + (scores - scores.min())
            / (scores.max() - scores.min())
            * (scale_max - scale_min)
        )
    elif normalization_scale == "zero-to-one":
        return (scores - scores.min()) / (scores.max() - scores.min())
    elif normalization_scale == "negative-one-to-one":
        return 2 * (scores - scores.min()) / (scores.max() - scores.min()) - 1
    elif normalization_scale == "none":
        return scores
    else:
        raise ValueError(
            "normalization_scale must be 'zero-to-one', 'negative-one-to-one', "
            "'none', or a (min, max) tuple"
        )


# ---------------------------------------------------------------------------
# Main scoring entry point
# ---------------------------------------------------------------------------

def score_items(
    pairwise_df: pd.DataFrame,
    data: pd.DataFrame,
    item_id_name: str,
    target_concept: str,
    text_name: Optional[str],
    paired: bool,
    item_id_cols: Optional[list],
    scored_df: Optional[pd.DataFrame] = None,
    normalization_scale: Union[str, Tuple[float, float]] = "zero-to-one",
    summarize: bool = True,
    decision_col: str = "decision",
    use_davidson: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Compute Bradley-Terry or Davidson scores from pairwise comparison results.

    When the pairwise DataFrame contains item-level split columns
    (``item1_split`` / ``item2_split``), two score columns are added:

    * ``<Model>_Score_full``  – from **all** valid comparisons.
    * ``<Model>_Score_split`` – from **within-split** comparisons only.

    When no split columns are present, only a single ``<Model>_Score`` column
    is added.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise comparison DataFrame (``Pairadigm.pairwise_df``).
    data : pd.DataFrame
        Original item DataFrame (``Pairadigm.data``).
    item_id_name : str
        Column name for unique item identifiers.
    target_concept : str
        Name of the concept being measured (used in print diagnostics).
    text_name : str or None
        Column name for item text (used in summarize output).
    paired : bool
        Whether the original data was in paired format.
    item_id_cols : list or None
        For paired data: ``['item1', 'item2']``.
    scored_df : pd.DataFrame or None
        Existing scored DataFrame to extend (for unpaired data).
    normalization_scale : str or tuple, default 'zero-to-one'
        How to normalise scores.
    summarize : bool, default True
        Whether to print summary statistics.
    decision_col : str, default 'decision'
        Name of the decision column.
    use_davidson : bool or None, default None
        Force Davidson model.  If None, auto-detects based on ties.

    Returns
    -------
    pd.DataFrame
        DataFrame with added score column(s).
    """
    if pairwise_df is None:
        raise ValueError(
            "No pairwise comparison results found. "
            "Run generate_pairwise_annotations() first."
        )

    if decision_col not in pairwise_df.columns:
        available = [c for c in pairwise_df.columns if c.startswith("decision")]
        raise ValueError(
            f"Decision column '{decision_col}' not found in pairwise_df. "
            f"Available decision columns: {available}"
        )

    # Auto-detect ties
    has_ties = pairwise_df[decision_col].isin(TIE_VALUES).any()

    if use_davidson is None:
        use_davidson = has_ties
        if has_ties:
            num_ties = pairwise_df[decision_col].isin(TIE_VALUES).sum()
            print(f"Detected {num_ties} ties in data. Using Davidson model.")

    # Filter valid decisions
    valid_values = WIN_1_VALUES + WIN_2_VALUES + (TIE_VALUES if use_davidson else [])
    valid_df = pairwise_df[pairwise_df[decision_col].isin(valid_values)]

    if len(valid_df) == 0:
        raise ValueError("No valid comparisons found to compute scores.")

    if len(valid_df) < len(pairwise_df):
        warnings.warn(
            f"Some rows filtered out due to invalid decision values. "
            f"Using {len(valid_df)}/{len(pairwise_df)} comparisons."
        )

    model_label = (
        decision_col.replace("decision_", "") if decision_col != "decision" else "default"
    )

    # Detect split columns
    has_splits = (
        "item1_split" in pairwise_df.columns
        and "item2_split" in pairwise_df.columns
    )

    # Build item → index mapping
    if paired and item_id_cols:
        item1_col, item2_col = item_id_cols
        all_items = pd.concat(
            [pairwise_df[item1_col], pairwise_df[item2_col]]
        ).unique().tolist()
        item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    else:
        item_to_idx = {item: idx for idx, item in enumerate(data[item_id_name].tolist())}

    n_items = len(item_to_idx)

    # --- Full model ---
    if use_davidson:
        bt_scores_full, model_name, comparisons_full, tau_full = _fit_davidson_model(
            valid_df, item_to_idx, n_items, decision_col,
            f"{model_label} [full]" if has_splits else model_label,
        )
    else:
        bt_scores_full, model_name, comparisons_full = _fit_bt_model(
            valid_df, item_to_idx, n_items, decision_col,
            f"{model_label} [full]" if has_splits else model_label,
        )
        tau_full = None
    
    raw_full = bt_scores_full.copy()  # keep raw log-strengths for SE
    bt_scores_full = _normalize_bt_scores(bt_scores_full, normalization_scale)

    # --- SE for full model (BT only) ---
    se_full: Optional[np.ndarray] = None
    if comparisons_full:  # empty for Davidson
        se_full = _compute_bt_se(raw_full, comparisons_full, n_items, normalization_scale)

    # --- Split model (optional) ---
    bt_scores_split = None
    se_split: Optional[np.ndarray] = None
    split_item_to_idx = None
    compute_split = False

    if has_splits:
        within_split_df = valid_df[valid_df["item1_split"] == valid_df["item2_split"]]
        if len(within_split_df) == 0:
            warnings.warn(
                "No within-split pairs found among valid comparisons. "
                "Split scores will not be computed."
            )
        else:
            split_items = sorted(
                set(within_split_df["item1"].tolist())
                | set(within_split_df["item2"].tolist())
            )
            split_item_to_idx = {item: idx for idx, item in enumerate(split_items)}
            n_split = len(split_item_to_idx)

            if use_davidson:
                raw_split, _, comparisons_split, tau_split = _fit_davidson_model(
                    within_split_df, split_item_to_idx, n_split,
                    decision_col, f"{model_label} [split]",
                )
            else:
                raw_split, _, comparisons_split = _fit_bt_model(
                    within_split_df, split_item_to_idx, n_split,
                    decision_col, f"{model_label} [split]",
                )
                tau_split = None
                
            bt_scores_split = _normalize_bt_scores(raw_split, normalization_scale)
            compute_split = True
            if comparisons_split:
                se_split = _compute_bt_se(raw_split, comparisons_split, n_split, normalization_scale)

    # --- Determine column names ---
    model_prefix = model_name.replace("-", "_")
    decision_suffix = (
        "" if decision_col == "decision"
        else f"_{decision_col.replace('decision_', '')}"
    )
    if has_splits:
        full_col_name  = f"{model_prefix}_Score_full{decision_suffix}"
        split_col_name = f"{model_prefix}_Score_split{decision_suffix}"
        full_se_col    = f"{model_prefix}_SE_full{decision_suffix}"
        split_se_col   = f"{model_prefix}_SE_split{decision_suffix}"
    else:
        full_col_name  = f"{model_prefix}_Score{decision_suffix}"
        full_se_col    = f"{model_prefix}_SE{decision_suffix}"

    # --- Build result DataFrame ---
    if paired:
        result = pd.DataFrame({"item_id": list(item_to_idx.keys())})
        result[full_col_name] = [bt_scores_full[item_to_idx[item]] for item in result["item_id"]]
        if se_full is not None:
            result[full_se_col] = [se_full[item_to_idx[item]] for item in result["item_id"]]
        if compute_split:
            result[split_col_name] = [
                bt_scores_split[split_item_to_idx[item]]
                if item in split_item_to_idx else float("nan")
                for item in result["item_id"]
            ]
            if se_split is not None:
                result[split_se_col] = [
                    se_split[split_item_to_idx[item]]
                    if item in split_item_to_idx else float("nan")
                    for item in result["item_id"]
                ]
    else:
        result = scored_df.copy() if scored_df is not None else data.copy()
        result[full_col_name] = [bt_scores_full[item_to_idx[u]] for u in result[item_id_name]]
        if se_full is not None:
            result[full_se_col] = [se_full[item_to_idx[u]] for u in result[item_id_name]]
        if compute_split:
            result[split_col_name] = [
                bt_scores_split[split_item_to_idx[u]]
                if u in split_item_to_idx else float("nan")
                for u in result[item_id_name]
            ]
            if se_split is not None:
                result[split_se_col] = [
                    se_split[split_item_to_idx[u]]
                    if u in split_item_to_idx else float("nan")
                    for u in result[item_id_name]
                ]

    # --- Diagnostics ---
    tag_full  = f"[{model_label} full]"  if has_splits else f"[{model_label}]"
    tag_split = f"[{model_label} split]" if has_splits else ""

    if use_davidson:
        n_ties = valid_df[decision_col].isin(TIE_VALUES).sum()
        print(f"{tag_full} Including {n_ties} tie decisions")
        if tau_full is not None:
            print(f"{tag_full} Estimated tie propensity (tau): {tau_full:.4f}")
    print(f"{tag_full} Mean {target_concept} score: {result[full_col_name].mean():.3f}")
    print(f"{tag_full} Std  {target_concept} score: {result[full_col_name].std():.3f}")

    if compute_split:
        tag_split = f"[{model_label} split]"
        n_within  = len(within_split_df)
        n_missing = result[split_col_name].isna().sum()
        print(f"{tag_split} {model_name} model fitted with {n_within} within-split comparisons")
        if use_davidson:
            n_ties_s = within_split_df[decision_col].isin(TIE_VALUES).sum()
            print(f"{tag_split} Including {n_ties_s} tie decisions")
            if tau_split is not None:
                print(f"{tag_split} Estimated tie propensity (tau): {tau_split:.4f}")
        print(f"{tag_split} Mean {target_concept} score: {result[split_col_name].mean():.3f}")
        print(f"{tag_split} Std  {target_concept} score: {result[split_col_name].std():.3f}")
        if n_missing > 0:
            print(f"{tag_split} {n_missing} item(s) had no within-split comparisons → NaN.")

    if summarize:
        for col_name in ([full_col_name, split_col_name] if compute_split else [full_col_name]):
            print(f"\nSummary statistics ({col_name}):")
            col_data = result[col_name].dropna()
            for k, v in {
                "mean":   col_data.mean(),
                "median": col_data.median(),
                "std":    col_data.std(),
                "min":    col_data.min(),
                "max":    col_data.max(),
                "count":  col_data.count(),
            }.items():
                print(f"  {k}: {v:.3f}")

    return result


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------

def summarize_scores(
    df: pd.DataFrame,
    target_concept: str,
    score_col: str = "Bradley_Terry_Score",
    text_col: Optional[str] = None,
) -> dict:
    """
    Print and return summary statistics for a score column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a score column.
    target_concept : str
        Name of the concept (used in print output).
    score_col : str
        Score column name.
    text_col : str or None
        If provided, prints the highest- and lowest-scoring texts.

    Returns
    -------
    dict
        ``{'mean', 'median', 'std', 'min', 'max', 'count'}``.
    """
    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found in DataFrame.")

    print(f"Score range: {df[score_col].min():.3f} to {df[score_col].max():.3f}")
    print(f"25th percentile: {df[score_col].quantile(0.25):.3f}")
    print(f"50th percentile (median): {df[score_col].quantile(0.50):.3f}")
    print(f"75th percentile: {df[score_col].quantile(0.75):.3f}")

    if text_col and text_col in df.columns:
        df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
        top   = df_sorted.iloc[0]
        bottom = df_sorted.iloc[-1]
        print(f"\nHighest scoring item on {target_concept} (score: {top[score_col]:.3f}):")
        print(top[text_col])
        print(f"\nLowest scoring item on {target_concept} (score: {bottom[score_col]:.3f}):")
        print(bottom[text_col])

    return {
        "mean":   df[score_col].mean(),
        "median": df[score_col].median(),
        "std":    df[score_col].std(),
        "min":    df[score_col].min(),
        "max":    df[score_col].max(),
        "count":  df[score_col].count(),
    }
