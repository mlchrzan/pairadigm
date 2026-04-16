"""
Validation helpers for Pairadigm.

Contains:
  - PerformanceWarning   : custom warning class for expensive operations (5c)
  - _dawid_skene_em      : consolidated EM algorithm (fixes 1e + 3b)
  - check_transitivity   : with O(N³) scale warning (5c) and dedup fix (1d)
  - prep_for_alt_test    : standardised to decision_col (7c)
  - alt_test             : with decision_col / llm_decision_col compat shim (7c)
  - dawid_skene_alt_test : uses consolidated EM (3b), M-step guard fix (1e)
  - dawid_skene_annotator_ranking : uses consolidated EM (3b)
  - irr                  : inter-rater reliability
  - icc                  : Intraclass Correlation Coefficient (9d)
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from ._stats import by_procedure, accuracy, neg_rmse, sim, ttest


# ---------------------------------------------------------------------------
# Custom warning classes
# ---------------------------------------------------------------------------

class PerformanceWarning(UserWarning):
    """Raised when an operation may be slow due to algorithmic complexity."""
    pass


# ---------------------------------------------------------------------------
# Consolidated Dawid-Skene EM (fixes 1e + 3b)
# ---------------------------------------------------------------------------

def _dawid_skene_em(
    labels: np.ndarray,
    num_classes: int,
    max_iter: int,
    tol: float,
    random_seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Single consolidated Dawid-Skene EM implementation.

    Handles missing annotations encoded as ``-1`` (the ``labels[i, j] >= 0``
    guard in the M-step — fix for bug **1e**).

    Parameters
    ----------
    labels : np.ndarray, shape (n_instances, n_annotators)
        Integer label matrix.  ``-1`` indicates a missing annotation.
    num_classes : int
        Number of distinct label classes.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance (L2 norm of label_probs change).
    random_seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    label_probs : np.ndarray, shape (n_instances, num_classes)
    annotator_reliability : np.ndarray, shape (n_annotators,)
        Mean diagonal of each annotator's confusion matrix.
    confusion : np.ndarray, shape (n_annotators, num_classes, num_classes)
        The confusion matrices for each annotator.
    converged_iter : int
        Iteration at which convergence was detected (or ``max_iter``).
    """
    n_instances, n_annotators = labels.shape
    if random_seed is not None:
        np.random.seed(random_seed)

    # Initialise with (valid) majority vote
    label_probs = np.zeros((n_instances, num_classes))
    for i in range(n_instances):
        valid = labels[i][labels[i] >= 0]
        if len(valid) > 0:
            majority = np.bincount(valid, minlength=num_classes)
            label_probs[i] = majority / majority.sum()
        else:
            label_probs[i] = 1.0 / num_classes

    # Initialise confusion matrices with small random noise
    confusion = (
        np.full((n_annotators, num_classes, num_classes), 1.0 / num_classes)
        + np.abs(np.random.randn(n_annotators, num_classes, num_classes) * 0.01)
    )
    # Normalise rows
    row_sums = confusion.sum(axis=2, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    confusion /= row_sums

    prev_label_probs = label_probs.copy()
    converged_iter = max_iter

    for iteration in range(max_iter):
        # --- E-step ---
        for i in range(n_instances):
            for c in range(num_classes):
                prob = 1.0
                for j in range(n_annotators):
                    observed = labels[i, j]
                    if observed >= 0:  # Skip missing annotations
                        prob *= confusion[j, c, observed]
                label_probs[i, c] = prob
            total = label_probs[i].sum()
            label_probs[i] = label_probs[i] / total if total > 0 else 1.0 / num_classes

        # --- M-step (vectorised numerator, with missing-annotation guard) ---
        for j in range(n_annotators):
            for c in range(num_classes):
                denom = sum(
                    label_probs[i, c]
                    for i in range(n_instances)
                    if labels[i, j] >= 0          # guard from fix 1e
                )
                for k in range(num_classes):
                    numer = sum(
                        label_probs[i, c]
                        for i in range(n_instances)
                        if labels[i, j] == k
                    )
                    confusion[j, c, k] = numer / denom if denom > 0 else 1.0 / num_classes

        # --- Convergence check ---
        delta = np.linalg.norm(label_probs - prev_label_probs)
        if iteration > 0 and delta < tol:
            converged_iter = iteration
            break
        prev_label_probs = label_probs.copy()

    # Annotator reliability = mean diagonal of confusion matrix
    reliability = np.array(
        [np.mean(np.diag(confusion[j])) for j in range(n_annotators)]
    )
    return label_probs, reliability, confusion, converged_iter


# ---------------------------------------------------------------------------
# check_transitivity (fix 1d + 5c)
# ---------------------------------------------------------------------------

def check_transitivity(
    pairwise_df: pd.DataFrame,
    annotator_cols: Optional[List[str]] = None,
    llm_annotator_cols: Optional[List[str]] = None,
    annotated: bool = False,
    llm_annotated: bool = False,
    all_annotator_cols: Optional[List[str]] = None,
) -> Dict[str, Tuple[float, int, int]]:
    """
    Check transitivity violations for annotators.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise comparison DataFrame.
    annotator_cols : list of str, optional
        Human annotator columns to check.  If ``all_annotator_cols`` is
        provided this parameter is ignored.
    llm_annotator_cols : list of str, optional
        LLM decision columns to check.
    annotated : bool
        Flag from Pairadigm instance.
    llm_annotated : bool
        Flag from Pairadigm instance.
    all_annotator_cols : list of str, optional
        Explicit list of columns to check (overrides auto-detection).

    Returns
    -------
    dict
        ``{annotator_col: (transitivity_score, n_violations, n_total_triples)}``
    """
    if pairwise_df is None:
        raise ValueError(
            "No pairwise comparison data found. "
            "Run generate_pairwise_annotations() first."
        )

    df = pairwise_df

    # Determine columns to check
    if all_annotator_cols is not None:
        raw_cols = all_annotator_cols if isinstance(all_annotator_cols, list) else [all_annotator_cols]
        for col in raw_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in pairwise DataFrame.")
    else:
        raw_cols = []
        if "decision" in df.columns:
            raw_cols.append("decision")
        if annotated and annotator_cols:
            raw_cols.extend(annotator_cols)
        if llm_annotated and llm_annotator_cols:
            raw_cols.extend(llm_annotator_cols)
        if not raw_cols:
            raise ValueError("No annotator columns found to check transitivity.")

    # Fix 1d: deduplicate while preserving order
    seen: set = set()
    cols_to_check: List[str] = []
    for c in raw_cols:
        if c not in seen:
            seen.add(c)
            cols_to_check.append(c)

    results: Dict[str, Tuple[float, int, int]] = {}
    items = list(set(df["item1"].unique()) | set(df["item2"].unique()))

    # 5c: O(N³) scale warning with proper PerformanceWarning category
    if len(items) > 200:
        warnings.warn(
            f"check_transitivity is O(n³); with {len(items)} items this may "
            "take a while. Consider sampling a subset of triples for large N.",
            PerformanceWarning,
            stacklevel=2,
        )

    for annotator_col in cols_to_check:
        violations = 0
        total_triples = 0

        comparisons: Dict[Tuple, float] = {}
        for _, row in df.iterrows():
            if pd.isna(row[annotator_col]):
                continue
            k1 = (row["item1"], row["item2"])
            k2 = (row["item2"], row["item1"])
            decision = row[annotator_col]
            if decision == "Text1":
                comparisons[k1], comparisons[k2] = 1, 0
            elif decision == "Text2":
                comparisons[k1], comparisons[k2] = 0, 1
            elif decision == "Tie":
                comparisons[k1] = comparisons[k2] = 0.5
            elif isinstance(decision, (int, float)) and decision in [0, 1]:
                comparisons[k1] = int(decision)
                comparisons[k2] = 1 - int(decision)

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                for k in range(j + 1, len(items)):
                    a, b, c = items[i], items[j], items[k]
                    ab, bc, ac = (a, b), (b, c), (a, c)
                    if not all(key in comparisons for key in [ab, bc, ac]):
                        continue
                    total_triples += 1
                    ab_d = comparisons[ab]
                    bc_d = comparisons[bc]
                    ac_d = comparisons[ac]

                    is_violation = False
                    if ab_d == 1 and bc_d == 1 and ac_d != 1:
                        is_violation = True
                    elif ab_d == 0 and bc_d == 0 and ac_d != 0:
                        is_violation = True
                    elif ab_d == 0.5 and bc_d == 0.5 and ac_d != 0.5:
                        is_violation = True
                    elif ab_d == 0.5 and bc_d == 1 and ac_d != 1:
                        is_violation = True
                    elif ab_d == 0.5 and bc_d == 0 and ac_d != 0:
                        is_violation = True
                    elif ab_d == 1 and bc_d == 0.5 and ac_d != 1:
                        is_violation = True
                    elif ab_d == 0 and bc_d == 0.5 and ac_d != 0:
                        is_violation = True

                    if is_violation:
                        violations += 1

        score = 1 - (violations / total_triples) if total_triples > 0 else 0.0
        results[annotator_col] = (score, violations, total_triples)

    return results


# ---------------------------------------------------------------------------
# prep_for_alt_test (standardised to decision_col — fix 7c)
# ---------------------------------------------------------------------------

def prep_for_alt_test(
    pairwise_df: pd.DataFrame,
    annotator_cols: List[str],
    item_id_cols: List[str],
    decision_col: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """
    Prepare annotation dicts for :func:`alt_test`.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise comparison DataFrame (``Pairadigm.pairwise_df``).
    annotator_cols : list of str
        Human annotator column names.
    item_id_cols : list of str
        ``['item1', 'item2']``
    decision_col : str, optional
        Which LLM decision column to use.  Auto-detected if only one exists.

    Returns
    -------
    (llm_annotations, humans_annotations)
    """
    if decision_col is None:
        dec_cols = [c for c in pairwise_df.columns if c.startswith("decision")]
        if len(dec_cols) == 0:
            raise ValueError(
                "No 'decision' columns found. Run generate_pairwise_annotations() first."
            )
        if len(dec_cols) == 1:
            decision_col = dec_cols[0]
        else:
            available = [c.replace("decision_", "") for c in dec_cols if c != "decision"]
            if "decision" in dec_cols:
                available.insert(0, "decision (default)")
            raise ValueError(
                f"Multiple LLM decision columns found: {dec_cols}. "
                f"Specify decision_col. Available models: {available}"
            )
    elif decision_col not in pairwise_df.columns:
        raise ValueError(
            f"Column '{decision_col}' not found in pairwise_df. "
            f"Available columns: {list(pairwise_df.columns)}"
        )

    llm_anns: Dict = {}
    human_anns: Dict[str, Dict] = {col: {} for col in annotator_cols}

    for idx, row in pairwise_df.iterrows():
        # The instance is the pair (represented by the row index).
        inst = idx
        # Collect LLM decision (independently of human annotations)
        decision = row[decision_col]
        if decision in ("Text1", 0, 0.0):
            llm_anns[inst] = 0
        elif decision in ("Text2", 1, 1.0):
            llm_anns[inst] = 1
        elif decision in ("Tie", 0.5):
            llm_anns[inst] = 0.5

        # Collect human annotations for every row regardless of LLM decision
        for col in annotator_cols:
            if col not in row or pd.isna(row[col]):
                continue
            hdec = row[col]
            if hdec in ("Text1", 0, 0.0):
                human_anns[col][inst] = 0
            elif hdec in ("Text2", 1, 1.0):
                human_anns[col][inst] = 1
            elif hdec in ("Tie", 0.5):
                human_anns[col][inst] = 0.5

    print(f"Using LLM decision column: {decision_col}")
    return llm_anns, human_anns


# ---------------------------------------------------------------------------
# alt_test (7c: standardised parameter name)
# ---------------------------------------------------------------------------

def alt_test(
    pairwise_df: pd.DataFrame,
    annotator_cols: List[str],
    item_id_cols: List[str],
    annotated: bool,
    llm_annotations: Optional[Dict] = None,
    humans_annotations: Optional[Dict] = None,
    scoring_function: Union[str, Callable] = "accuracy",
    epsilon: float = 0.1,
    q_fdr: float = 0.05,
    min_humans_per_instance: int = 2,
    min_instances_per_human: int = 30,
    decision_col: Optional[str] = None,
    # Backward-compat alias (7c)
    llm_decision_col: Optional[str] = None,
    test_all_llms: bool = False,
) -> Union[Tuple[float, float], Dict[str, Tuple[float, float]]]:
    """
    Perform the Alternative Annotator Test (AltTest).

    Parameters
    ----------
    pairwise_df : pd.DataFrame
    annotator_cols : list of str
    item_id_cols : list of str
    annotated : bool
    llm_annotations : dict, optional
    humans_annotations : dict, optional
    scoring_function : str or callable, default 'accuracy'
    epsilon : float, default 0.1
    q_fdr : float, default 0.05
    min_humans_per_instance : int, default 2
    min_instances_per_human : int, default 30
    decision_col : str, optional
        Preferred parameter name (standardised in v1.0).
    llm_decision_col : str, optional
        Deprecated alias for ``decision_col``.
    test_all_llms : bool, default False

    Returns
    -------
    (winning_rate, advantage_prob) or dict of model → (winning_rate, advantage_prob)
    """
    # 7c: backward-compat shim
    if llm_decision_col is not None and decision_col is None:
        warnings.warn(
            "llm_decision_col is deprecated; use decision_col instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        decision_col = llm_decision_col

    if not annotated:
        raise ValueError("Data must have human annotations to run the alt_test.")

    if pairwise_df is None:
        raise ValueError(
            "No pairwise comparison data found. Run generate_pairwise_annotations() first."
        )

    dec_cols = [c for c in pairwise_df.columns if c.startswith("decision")]
    if not dec_cols:
        raise ValueError("No 'decision' columns found. Run generate_pairwise_annotations() first.")

    if test_all_llms:
        cols_to_test = dec_cols
        print(f"Testing all {len(cols_to_test)} LLM decision columns: {cols_to_test}")
    else:
        if decision_col is None:
            if len(dec_cols) == 1:
                cols_to_test = [dec_cols[0]]
            else:
                available = [c.replace("decision_", "") for c in dec_cols if c != "decision"]
                if "decision" in dec_cols:
                    available.insert(0, "decision (default)")
                raise ValueError(
                    f"Multiple LLM decision columns found: {dec_cols}. "
                    f"Specify decision_col or set test_all_llms=True. "
                    f"Available models: {available}"
                )
        else:
            if decision_col not in pairwise_df.columns:
                raise ValueError(
                    f"Column '{decision_col}' not found in pairwise_df. "
                    f"Available columns: {list(pairwise_df.columns)}"
                )
            cols_to_test = [decision_col]

    if isinstance(scoring_function, str):
        scoring_function = {"accuracy": accuracy, "neg_rmse": neg_rmse}[scoring_function] \
            if scoring_function in ("accuracy", "neg_rmse") \
            else (_ for _ in ()).throw(ValueError("Unknown scoring function"))

    # Shared human annotations
    if humans_annotations is None:
        _, humans_annotations = prep_for_alt_test(
            pairwise_df, annotator_cols, item_id_cols, cols_to_test[0]
        )

    results: Dict[str, Tuple[float, float]] = {}

    for col in cols_to_test:
        if llm_annotations is None or test_all_llms:
            llm_anns, _ = prep_for_alt_test(pairwise_df, annotator_cols, item_id_cols, col)
        else:
            llm_anns = llm_annotations

        i_set: Dict = {}
        h_set: Dict = {}
        for h, anns in humans_annotations.items():
            i_set[h] = list(anns.keys())
            for inst, ann in anns.items():
                h_set.setdefault(inst, []).append(h)

        instances_to_keep = set()
        missing_llm = 0
        few_humans = 0
        for i in h_set:
            if i not in llm_anns:
                missing_llm += 1
            elif len(h_set[i]) < min_humans_per_instance:
                few_humans += 1
            else:
                instances_to_keep.add(i)

        if missing_llm > 0 or few_humans > 0:
            msgs = []
            if few_humans > 0:
                msgs.append(f"{few_humans} instances with < {min_humans_per_instance} annotators")
            if missing_llm > 0:
                msgs.append(f"{missing_llm} instances missing valid LLM decisions")
            print(f"[{col}] Dropped " + " and ".join(msgs) + ".")
        i_set = {h: [i for i in i_set[h] if i in instances_to_keep] for h in i_set}
        h_set = {i: h_set[i] for i in h_set if i in instances_to_keep}

        p_values, advantage_probs, humans_list = [], [], []
        for excluded_h in humans_annotations:
            instances = [i for i in i_set[excluded_h] if i in llm_anns]
            if len(instances) < min_instances_per_human:
                print(f"[{col}] Skipping {excluded_h} ({len(instances)} < {min_instances_per_human} instances).")
                continue

            llm_indicators, excl_indicators = [], []
            for inst in instances:
                h_ann   = humans_annotations[excluded_h][inst]
                l_ann   = llm_anns[inst]
                rest    = [humans_annotations[h][inst] for h in h_set[inst] if h != excluded_h]
                h_score = scoring_function(h_ann, rest)
                l_score = scoring_function(l_ann, rest)
                llm_indicators.append(1 if l_score >= h_score else 0)
                excl_indicators.append(1 if h_score >= l_score else 0)

            diff = [e - l for e, l in zip(excl_indicators, llm_indicators)]
            p_values.append(ttest(diff, epsilon))
            advantage_probs.append(float(np.mean(llm_indicators)))
            humans_list.append(excluded_h)

        rejected = by_procedure(p_values, q_fdr)
        adv_prob     = float(np.mean(advantage_probs))
        winning_rate = len(rejected) / len(humans_list) if humans_list else 0.0

        model_name = col.replace("decision_", "") if col != "decision" else "default"
        results[model_name] = (winning_rate, adv_prob)

        print(f"\n{'='*70}")
        print(f"ALT-TEST RESULTS - {model_name.upper()}")
        print(f"{'='*70}")
        print(f"Winning Rate (ω): {winning_rate:.3f}")
        print(f"Advantage Probability: {adv_prob:.3f}")
        print(f"Tested against {len(humans_list)} human annotators")
        print(f"{'='*70}\n")

    return list(results.values())[0] if len(results) == 1 else results


# ---------------------------------------------------------------------------
# dawid_skene_alt_test
# ---------------------------------------------------------------------------

def dawid_skene_alt_test(
    pairwise_df: pd.DataFrame,
    annotator_cols: List[str],
    annotated: bool,
    decision_col: Optional[str] = None,
    num_classes: int = 2,
    max_iter: int = 1000,
    tol: float = 1e-6,
    random_seed: Optional[int] = None,
    alpha: float = 0.05,
    use_by_correction: bool = True,
    test_all_llms: bool = False,
) -> Union[Dict, Dict[str, Dict]]:
    """Dawid-Skene version of the ALT-TEST (uses consolidated EM — fixes 1e + 3b)."""
    from statsmodels.stats.multitest import multipletests

    if not annotated:
        raise ValueError("Data must have human annotations to run the Dawid-Skene alt_test.")
    if pairwise_df is None:
        raise ValueError("No pairwise comparison data found.")

    dec_cols = [c for c in pairwise_df.columns if c.startswith("decision")]
    if not dec_cols:
        raise ValueError("No 'decision' columns found. Run generate_pairwise_annotations() first.")

    if test_all_llms:
        cols_to_test = dec_cols
    else:
        if decision_col is None:
            if len(dec_cols) == 1:
                cols_to_test = [dec_cols[0]]
            else:
                available = [c.replace("decision_", "") for c in dec_cols if c != "decision"]
                if "decision" in dec_cols:
                    available.insert(0, "decision (default)")
                raise ValueError(
                    f"Multiple LLM decision columns found: {dec_cols}. "
                    f"Specify decision_col or set test_all_llms=True. "
                    f"Available models: {available}"
                )
        else:
            if decision_col not in pairwise_df.columns:
                raise ValueError(f"Column '{decision_col}' not found in pairwise_df.")
            cols_to_test = [decision_col]

    all_results: Dict[str, Dict] = {}
    instances = pairwise_df.index.tolist()
    num_instances = len(instances)
    num_annotators = len(annotator_cols)

    # Build annotation matrix (shared across LLM columns)
    annotator_labels = np.full((num_instances, num_annotators), -1, dtype=int)
    for j, acol in enumerate(annotator_cols):
        for i in range(num_instances):
            val = pairwise_df.iloc[i][acol]
            if pd.isna(val):
                continue
            elif val in ("Text1", 0, 0.0):
                annotator_labels[i, j] = 0
            elif val in ("Text2", 1, 1.0):
                annotator_labels[i, j] = 1
            elif val in ("Tie", 0.5):
                annotator_labels[i, j] = 2

    if np.any(annotator_labels == 2) and num_classes < 3:
        warnings.warn(
            "Ties (class 2) detected in the data, but num_classes is set to less than 3. "
            "This will likely cause the Dawid-Skene algorithm to fail. "
            "Please explicitly set num_classes=3.",
            UserWarning
        )

    label_probs, annotator_weights, _, conv_iter = _dawid_skene_em(
        annotator_labels, num_classes, max_iter, tol, random_seed
    )

    for col in cols_to_test:
        llm_labels = np.full(num_instances, -1, dtype=int)
        for i in range(num_instances):
            val = pairwise_df.iloc[i][col]
            if val in ("Text1", 0, 0.0):
                llm_labels[i] = 0
            elif val in ("Text2", 1, 1.0):
                llm_labels[i] = 1
            elif val in ("Tie", 0.5):
                llm_labels[i] = 2

        margins = []
        for j in range(num_annotators):
            other = [k for k in range(num_annotators) if k != j]
            for i in range(num_instances):
                if annotator_labels[i, j] == -1 or llm_labels[i] == -1:
                    continue
                valid_other = [k for k in other if annotator_labels[i, k] != -1]
                if not valid_other:
                    continue
                valid_weights = annotator_weights[valid_other]
                llm_agr  = [int(llm_labels[i] == annotator_labels[i, k]) for k in valid_other]
                hum_agr  = [int(annotator_labels[i, j] == annotator_labels[i, k]) for k in valid_other]
                if sum(valid_weights) > 0:
                    llm_wacc = np.average(llm_agr, weights=valid_weights)
                    hum_wacc = np.average(hum_agr, weights=valid_weights)
                    margins.append((j, llm_wacc - hum_wacc))

        ann_margins = pd.DataFrame(margins, columns=["annotator", "margin"])
        p_values = []
        advantage_probabilities: Dict[str, Dict] = {}
        winning_count = 0

        for j in range(num_annotators):
            deltas = ann_margins[ann_margins["annotator"] == j]["margin"].values
            _, p   = ttest_1samp(deltas, popmean=0, alternative="greater")
            p_values.append(p)
            ann_name = annotator_cols[j]
            advantage_probabilities[ann_name] = {
                "mean_margin": float(np.mean(deltas)),
                "p_value": float(p),
            }

        if use_by_correction:
            reject, corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_by")
            for j, ann_name in enumerate(advantage_probabilities):
                advantage_probabilities[ann_name]["corrected_p_value"] = float(corrected[j])
                advantage_probabilities[ann_name]["reject_null"] = bool(reject[j])
                if reject[j]:
                    winning_count += 1
        else:
            for j, ann_name in enumerate(advantage_probabilities):
                advantage_probabilities[ann_name]["reject_null"] = p_values[j] < alpha
                if p_values[j] < alpha:
                    winning_count += 1

        winning_rate = winning_count / num_annotators
        model_name   = col.replace("decision_", "") if col != "decision" else "default"

        all_results[model_name] = {
            "annotator_weights":      annotator_weights,
            "label_probs":            label_probs,
            "margins":                margins,
            "advantage_probabilities": advantage_probabilities,
            "winning_rate":           winning_rate,
            "convergence_iteration":  conv_iter,
        }

        print("\n" + "=" * 70)
        print(f"DAWID-SKENE VALIDATION RESULTS - {model_name.upper()}")
        print("=" * 70)
        print(f"Converged at iteration: {conv_iter}")
        print("\nAnnotator Reliability Weights:")
        for j, acol in enumerate(annotator_cols):
            print(f"  {acol}: {annotator_weights[j]:.4f}")
        print("\nAdvantage Probabilities per Annotator:")
        for ann_name, stats in advantage_probabilities.items():
            print(f"\n{ann_name}:")
            print(f"  Mean margin: {stats['mean_margin']:.4f}")
            print(f"  p-value: {stats['p_value']:.4f}")
            if use_by_correction:
                print(f"  Corrected p-value: {stats['corrected_p_value']:.4f}")
            print(f"  Reject null: {stats['reject_null']}")
        print(f"\nOverall Winning Rate (ω): {winning_rate:.2f}")
        print("=" * 70 + "\n")

    return list(all_results.values())[0] if len(all_results) == 1 else all_results


# ---------------------------------------------------------------------------
# dawid_skene_annotator_ranking
# ---------------------------------------------------------------------------

def dawid_skene_annotator_ranking(
    pairwise_df: pd.DataFrame,
    annotator_cols: Optional[List[str]] = None,
    llm_annotator_cols: Optional[List[str]] = None,
    llm_annotated: bool = False,
    human_annotated: bool = False,
    num_classes: int = 2,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_seed: Optional[int] = None,
    return_confusion_matrices: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, np.ndarray]]]:
    """
    Rank annotators by Dawid-Skene reliability (uses consolidated EM — fix 3b).

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        DataFrame containing pairwise comparison annotations.
    annotator_cols : Optional[List[str]]
        List of column names containing human annotations.
    llm_annotator_cols : Optional[List[str]]
        List of column names containing LLM annotations.
    llm_annotated : bool
        Whether the data contains LLM annotations.
    human_annotated : bool
        Whether the data contains human annotations.
    num_classes : int
        Number of classes (default: 2).
    max_iter : int
        Maximum number of EM iterations (default: 100).
    tol : float
        Tolerance for EM convergence (default: 1e-6).
    random_seed : Optional[int]
        Random seed for reproducibility (default: None).
    return_confusion_matrices : bool
        Whether to return confusion matrices (default: False).

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, Dict[str, np.ndarray]]
        DataFrame containing annotator reliability rankings. If 
        ``return_confusion_matrices`` is True, also returns a dictionary 
        mapping annotator names to their confusion matrices.
    """
    if not llm_annotated and not human_annotated:
        raise ValueError("Data must have annotations to rank annotators.")
    if pairwise_df is None:
        raise ValueError("No pairwise comparison data found.")
    if random_seed is None:
        raise ValueError(
            "A seed is required for reproducibility of the EM algorithm. "
            "Recommended practice is to run results over multiple seeds."
        )

    _all_cols = []
    if human_annotated and annotator_cols:
        _all_cols.extend(annotator_cols)
    if llm_annotated:
        if llm_annotator_cols:
            _all_cols.extend(llm_annotator_cols)
        else:
            _all_cols.extend([c for c in pairwise_df.columns if c.startswith("decision")])
            
    if not _all_cols:
        # Fallback if boolean flags were missed
        if annotator_cols: _all_cols.extend(annotator_cols)
        if llm_annotator_cols: _all_cols.extend(llm_annotator_cols)
        
    if not _all_cols:
        raise ValueError("No annotator columns found to rank.")
        
    annotator_cols = list(dict.fromkeys(_all_cols)) # Remove duplicates while preserving order

    for col in annotator_cols:
        if col not in pairwise_df.columns:
            raise ValueError(f"Column '{col}' not found in pairwise_df.")

    n_instances  = len(pairwise_df)
    n_annotators = len(annotator_cols)
    print(f"Ranking {n_annotators} annotators across {n_instances} instances...")

    labels = np.full((n_instances, n_annotators), -1, dtype=int)
    for j, col in enumerate(annotator_cols):
        for i in range(n_instances):
            val = pairwise_df.iloc[i][col]
            if pd.isna(val):
                continue
            elif val in ("Text1", 0, 0.0):
                labels[i, j] = 0
            elif val in ("Text2", 1, 1.0):
                labels[i, j] = 1
            elif val in ("Tie", 0.5):
                labels[i, j] = 2

    if np.any(labels == 2) and num_classes < 3:
        warnings.warn(
            "Ties (class 2) detected in the data, but num_classes is set to less than 3. "
            "This will likely cause the Dawid-Skene algorithm to fail. "
            "Please explicitly set num_classes=3.",
            UserWarning
        )

    _, reliability, confusion, conv_iter = _dawid_skene_em(
        labels, num_classes, max_iter, tol, random_seed
    )

    rows = []
    for j, col in enumerate(annotator_cols):
        row = {
            "annotator":   col,
            "reliability": reliability[j],
            "type":        "LLM" if col.startswith("decision") else "Human",
        }
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df["rank"] = (
        results_df["reliability"].rank(ascending=False, method="min").astype(int)
    )
    results_df = results_df.sort_values("reliability", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("DAWID-SKENE ANNOTATOR RANKING")
    print("=" * 70)
    print(f"Converged at iteration: {conv_iter}")
    print("\nTop 5 Most Reliable Annotators:")
    print(results_df[["rank", "annotator", "reliability", "type"]].head())
    print("\n" + "=" * 70 + "\n")

    if return_confusion_matrices:
        confusion_matrices = {
            col: confusion[j] for j, col in enumerate(annotator_cols)
        }
        return results_df, confusion_matrices

    return results_df


# ---------------------------------------------------------------------------
# IRR
# ---------------------------------------------------------------------------

def irr(
    pairwise_df: pd.DataFrame,
    annotator_cols: Optional[List[str]],
    llm_annotator_cols: Optional[List[str]],
    annotated: bool,
    llm_annotated: bool,
    method: str = "auto",
    alpha_level: str = "nominal",
    min_overlap: int = 2,
) -> pd.DataFrame:
    """
    Calculate inter-rater reliability (IRR) between annotators.

    Returns a DataFrame with columns:
    ``group, method, score, n_annotators, n_items, interpretation, error``.
    """
    from sklearn.metrics import cohen_kappa_score

    if not annotated and not llm_annotated:
        raise ValueError("No annotations found.")
    if pairwise_df is None:
        raise ValueError("No pairwise comparison data found.")

    def _interpret(score: float) -> str:
        if score < 0:      return "Poor (worse than chance)"
        elif score < 0.20: return "Slight"
        elif score < 0.40: return "Fair"
        elif score < 0.60: return "Moderate"
        elif score < 0.80: return "Substantial"
        else:              return "Almost Perfect"

    def _cohens_kappa(a1, a2):
        mask = (~pd.isna(a1)) & (~pd.isna(a2))
        if mask.sum() < 2:
            raise ValueError("Insufficient overlapping annotations (need >= 2).")
        return cohen_kappa_score(a1[mask], a2[mask])

    def _fleiss_kappa(mat, n_cat=None):
        complete = ~np.isnan(mat).any(axis=1)
        if complete.sum() < 2:
            raise ValueError("Insufficient complete cases for Fleiss' Kappa.")
        M = mat[complete]
        n, k = M.shape
        cats = np.unique(M[~np.isnan(M)]) if n_cat is None else np.arange(n_cat)
        n_cat = len(cats)
        freq = np.zeros((n, n_cat))
        for idx, cat in enumerate(cats):
            freq[:, idx] = (M == cat).sum(axis=1)
        p_j     = freq.sum(axis=0) / (n * k)
        P_i     = ((freq ** 2).sum(axis=1) - k) / (k * (k - 1))
        P_bar   = P_i.mean()
        P_e_bar = (p_j ** 2).sum()
        return 1.0 if P_e_bar == 1 else (P_bar - P_e_bar) / (1 - P_e_bar)

    def _krippendorff(mat, level="nominal", n_cat=None):
        n_items, n_raters = mat.shape
        cats = np.unique(mat[~np.isnan(mat)]) if n_cat is None else np.arange(n_cat)
        n_cat = len(cats)
        c2i = {cat: i for i, cat in enumerate(cats)}
        coinc = np.zeros((n_cat, n_cat))
        for i in range(n_items):
            valid = mat[i][~np.isnan(mat[i])]
            nv = len(valid)
            if nv < 2:
                continue
            for c1 in valid:
                for c2 in valid:
                    coinc[c2i[c1], c2i[c2]] += 1 / (nv - 1)
        n_total = coinc.sum()
        if n_total == 0:
            raise ValueError("No valid pairs for Krippendorff's Alpha.")

        def delta(ci, cj):
            if level == "nominal":        return 1
            elif level == "ordinal":      return (ci - cj) ** 2
            elif level in ("interval", "ratio"):
                return (cats[ci] - cats[cj]) ** 2
            return 1

        D_o = sum(
            coinc[ci, cj] * delta(ci, cj)
            for ci in range(n_cat) for cj in range(n_cat) if ci != cj
        ) / n_total
        n_c = coinc.sum(axis=0) + coinc.sum(axis=1)
        D_e = sum(
            n_c[ci] * n_c[cj] * delta(ci, cj)
            for ci in range(n_cat) for cj in range(n_cat) if ci != cj
        ) / (n_total * (n_total - 1))
        return 1.0 if D_e == 0 else 1 - D_o / D_e

    def _calculate(cols, label):
        if len(cols) < 2:
            raise ValueError(f"Cannot calculate IRR for {label} with only 1 annotator.")
        unique_vals: set = set()
        for col in cols:
            unique_vals.update(pairwise_df[col].dropna().unique())
        has_ties  = any(v in ("Tie", "tie", 2) for v in unique_vals)
        n_cat     = 3 if has_ties else 2

        mat = np.full((len(pairwise_df), len(cols)), np.nan)
        for j, col in enumerate(cols):
            for i in range(len(pairwise_df)):
                val = pairwise_df.iloc[i][col]
                if pd.isna(val):          mat[i, j] = np.nan
                elif val in ("Text1", 0): mat[i, j] = 0
                elif val in ("Text2", 1): mat[i, j] = 1
                elif val in ("Tie", "tie", 2): mat[i, j] = 2

        valid = (~np.isnan(mat)).sum(axis=1) >= min_overlap
        if valid.sum() < 2:
            raise ValueError(f"Insufficient items with {min_overlap}+ annotators for {label}.")
        fmat = mat[valid]

        chosen = (
            ("cohens_kappa" if len(cols) == 2 else "krippendorff")
            if method == "auto" else method
        )
        if chosen == "cohens_kappa":
            if len(cols) != 2:
                raise ValueError("Cohen's Kappa requires exactly 2 annotators.")
            score = _cohens_kappa(fmat[:, 0], fmat[:, 1])
        elif chosen == "fleiss_kappa":
            score = _fleiss_kappa(fmat, n_cat=n_cat)
        elif chosen == "krippendorff":
            score = _krippendorff(fmat, level=alpha_level, n_cat=n_cat)
        else:
            raise ValueError(f"Unknown method: {chosen}")

        return {
            "method": chosen,
            "score": score,
            "n_annotators": len(cols),
            "n_items": int(valid.sum()),
            "interpretation": _interpret(score),
        }

    results: Dict[str, Any] = {}
    if annotated and annotator_cols:
        try:
            results["human"] = _calculate(annotator_cols, "human annotators")
        except ValueError as exc:
            results["human"] = {"error": str(exc)}
    if llm_annotated and llm_annotator_cols:
        try:
            results["llm"] = _calculate(llm_annotator_cols, "LLM annotators")
        except ValueError as exc:
            results["llm"] = {"error": str(exc)}

    all_cols = list(annotator_cols or []) + list(llm_annotator_cols or [])
    if len(all_cols) >= 2:
        try:
            results["all"] = _calculate(all_cols, "all annotators")
        except ValueError as exc:
            results["all"] = {"error": str(exc)}

    print("\n" + "=" * 70)
    print("INTER-RATER RELIABILITY RESULTS")
    print("=" * 70)
    for group in ("human", "llm", "all"):
        if group in results:
            print(f"\n{group.upper()} ANNOTATORS:")
            r = results[group]
            if "error" in r:
                print(f"  Error: {r['error']}")
            else:
                print(f"  Method: {r['method'].replace('_', ' ').title()}")
                print(f"  Score: {r['score']:.3f}")
                print(f"  Interpretation: {r['interpretation']}")
                print(f"  Annotators: {r['n_annotators']}")
                print(f"  Items: {r['n_items']}")
    print("=" * 70 + "\n")

    rows = []
    for group in ("human", "llm", "all"):
        if group in results:
            r = results[group]
            rows.append({
                "group":          group,
                "method":         r.get("method"),
                "score":          r.get("score"),
                "n_annotators":   r.get("n_annotators"),
                "n_items":        r.get("n_items"),
                "interpretation": r.get("interpretation"),
                "error":          r.get("error"),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ICC — Intraclass Correlation Coefficient (9d)
# ---------------------------------------------------------------------------

def icc(
    pairwise_df: pd.DataFrame,
    annotator_cols: List[str],
    llm_annotator_cols: List[str],
    annotated: bool,
    llm_annotated: bool,
    decision_mapping: Optional[Dict] = None,
    groups: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute Intraclass Correlation Coefficients (ICC) for annotator reliability.

    Uses the ICC(2,1) formulation: two-way random effects model, single
    measurement, absolute agreement.  Decisions are first mapped to a numeric
    scale (``Text1`` → 0, ``Text2`` → 1, ``Tie`` → 0.5) or via a custom
    ``decision_mapping``.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise comparison DataFrame (``Pairadigm.pairwise_df``).
    annotator_cols : list of str
        Human annotator column names.
    llm_annotator_cols : list of str
        LLM annotator column names.
    annotated : bool
    llm_annotated : bool
    decision_mapping : dict or None
        Custom mapping from decision values to numeric scores.
        Defaults to ``{'Text1': 0, 'Text2': 1, 'Tie': 0.5}``.
    groups : list of str or None
        Which annotator groups to include.  Options: ``'human'``, ``'llm'``,
        ``'all'``.  Defaults to all available.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    pd.DataFrame
        Rows for each requested group with columns:

        * ``group``
        * ``icc`` — point estimate
        * ``ci_lower``, ``ci_upper`` — ``(1-alpha)`` CI
        * ``f_stat``, ``df1``, ``df2``, ``p_value``
        * ``n_annotators``, ``n_items``
        * ``interpretation``
    """
    from scipy import stats as _stats

    _default_mapping = {"Text1": 0, 0: 0, "Text2": 1, 1: 1, "Tie": 0.5, "tie": 0.5, 2: 0.5}
    mapping = decision_mapping or _default_mapping

    def _interp_icc(v: float) -> str:
        if v < 0:    return "Poor (negative)"
        if v < 0.50: return "Poor"
        if v < 0.75: return "Moderate"
        if v < 0.90: return "Good"
        return "Excellent"

    def _compute_icc_group(cols: List[str]) -> dict:
        """Compute ICC(2,1) for a list of annotator columns."""
        available = [c for c in cols if c in pairwise_df.columns]
        if len(available) < 2:
            return {"error": f"Need >= 2 annotators, found {len(available)}."}

        # Build numeric matrix: rows = items (pairs), cols = raters
        mat = pairwise_df[available].applymap(
            lambda x: mapping.get(x, np.nan)
        ).values.astype(float)

        # Keep only rows with >= 2 non-NaN ratings
        row_valid = (~np.isnan(mat)).sum(axis=1) >= 2
        mat = mat[row_valid]
        if mat.shape[0] < 2:
            return {"error": "Fewer than 2 fully-rated items."}

        n, k = mat.shape  # items x raters

        # Grand mean (listwise) — using nanmean for robustness
        grand_mean = np.nanmean(mat)

        # Row means (per item)
        row_means = np.nanmean(mat, axis=1)

        # Column means (per rater)
        col_means = np.nanmean(mat, axis=0)

        # Total SS
        MSr_num = k * np.nansum((row_means - grand_mean) ** 2)
        MSr_df = n - 1

        MSc_num = n * np.sum((col_means - grand_mean) ** 2)
        MSc_df = k - 1

        SSe = np.nansum((mat - row_means[:, None] - col_means[None, :] + grand_mean) ** 2)
        MSe_df = (n - 1) * (k - 1)
        if MSe_df == 0:
            return {"error": "Insufficient degrees of freedom."}

        MSr = MSr_num / MSr_df
        MSc = MSc_num / MSc_df if MSc_df > 0 else np.nan
        MSe = SSe / MSe_df

        # ICC(2,1) absolute agreement
        icc_val = (MSr - MSe) / (MSr + (k - 1) * MSe + k * (MSc - MSe) / n)
        icc_val = float(np.clip(icc_val, -1.0, 1.0))

        # F-test
        F = MSr / MSe if MSe > 0 else np.nan
        p_val = float(_stats.f.sf(F, MSr_df, MSe_df)) if not np.isnan(F) else np.nan

        # Confidence interval for ICC using Fisher's F-transform
        F_lower = F / _stats.f.ppf(1 - alpha / 2, MSr_df, MSe_df)
        F_upper = F * _stats.f.ppf(1 - alpha / 2, MSe_df, MSr_df)
        ci_lower = float(np.clip((F_lower - 1) / (F_lower + k - 1), -1, 1))
        ci_upper = float(np.clip((F_upper - 1) / (F_upper + k - 1), -1, 1))

        return {
            "icc":           icc_val,
            "ci_lower":      ci_lower,
            "ci_upper":      ci_upper,
            "f_stat":        float(F) if not np.isnan(F) else None,
            "df1":           int(MSr_df),
            "df2":           int(MSe_df),
            "p_value":       p_val,
            "n_annotators":  len(available),
            "n_items":       int(n),
            "interpretation": _interp_icc(icc_val),
        }

    # --- Determine group configurations ---
    group_configs: Dict[str, List[str]] = {}
    if annotated and annotator_cols:
        group_configs["human"] = annotator_cols
    if llm_annotated and llm_annotator_cols:
        group_configs["llm"] = llm_annotator_cols
    if group_configs:
        group_configs["all"] = (
            (annotator_cols if annotated else [])
            + (llm_annotator_cols if llm_annotated else [])
        )

    if not group_configs:
        raise ValueError("No annotated data found. Add annotations before running ICC.")

    if groups is not None:
        unknown = set(groups) - set(group_configs)
        if unknown:
            raise ValueError(f"Unknown groups: {unknown}. Valid: {list(group_configs.keys())}")
        group_configs = {g: group_configs[g] for g in groups if g in group_configs}

    print("=" * 70)
    print("  ICC ANALYSIS (ICC(2,1) — two-way random, absolute agreement)")
    print("=" * 70)

    rows = []
    for group, cols in group_configs.items():
        result = _compute_icc_group(cols)
        if "error" not in result:
            print(f"\n[{group.upper()}]")
            print(f"  ICC(2,1): {result['icc']:.4f} "
                  f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}] "
                  f"95% CI")
            print(f"  F({result['df1']}, {result['df2']}) = {result['f_stat']:.3f}, "
                  f"p = {result['p_value']:.4f}")
            print(f"  Interpretation: {result['interpretation']}")
            print(f"  Raters: {result['n_annotators']}, Items: {result['n_items']}")
        else:
            print(f"\n[{group.upper()}] Skipped: {result['error']}")
        rows.append({"group": group, **result})

    print("=" * 70 + "\n")
    return pd.DataFrame(rows)
