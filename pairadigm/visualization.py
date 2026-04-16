"""
Visualisation helpers for Pairadigm.

Contains:
  - plot_score_distribution   — interactive Plotly histogram
  - plot_comparison_network   — directed comparison network (fix 1b: guard before lookup)
  - plot_epsilon_sensitivity  — winning-rate vs epsilon sweep
"""

from __future__ import annotations

import logging
import warnings
import textwrap
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score distribution
# ---------------------------------------------------------------------------

def plot_score_distribution(
    scored_df: pd.DataFrame,
    target_concept: str,
    score_col: Optional[str] = None,
    title: Optional[str] = None,
    nbins: int = 30,
    show_stats: bool = True,
    color: str = "skyblue",
    template: str = "plotly_white",
    return_fig: bool = False,
    compare_splits: bool = False,
    backend: str = "plotly",
) -> Optional[object]:
    """
    Plot a histogram of Pairadigm scores.

    Parameters
    ----------
    scored_df : pd.DataFrame
        DataFrame containing the score column (``Pairadigm.scored_df``).
    target_concept : str
        Name of the concept being measured (used in axis labels / title).
    score_col : str or None
        Column name for scores.  If ``None``, the method auto-selects:
        first looks for a ``*_Score_full`` column, then ``*_Score``.
    title : str or None
        Plot title.  Auto-generated if ``None``.
    nbins : int
        Number of histogram bins.
    show_stats : bool
        Whether to overlay mean/median lines and a statistics annotation
        (Plotly backend only).
    color : str
        Histogram bar colour (single-column Plotly or Seaborn mode).
    template : str
        Plotly template (Plotly backend only).
    return_fig : bool
        If ``True``, returns the figure object instead of calling ``.show()``.
    compare_splits : bool, default False
        If ``True`` and both ``*_Score_full`` and ``*_Score_split`` columns
        exist, plots both distributions side-by-side for comparison.
    backend : str, default 'plotly'
        Rendering backend: ``'plotly'`` (interactive) or ``'seaborn'`` (static).

    Returns
    -------
    Figure or None
    """
    if scored_df is None:
        raise ValueError("No scored DataFrame found. Run score_items() first.")

    if backend not in ("plotly", "seaborn"):
        raise ValueError("backend must be 'plotly' or 'seaborn'.")

    # --- Auto-detect score column ---
    if score_col is None:
        full_cols  = [c for c in scored_df.columns if c.endswith("_Score_full")]
        plain_cols = [c for c in scored_df.columns if c.endswith("_Score") and not c.endswith("_Score_split") and not c.endswith("_Score_full")]
        if full_cols:
            score_col = full_cols[0]
            logger.info("Auto-selected score column: %s", score_col)
        elif plain_cols:
            score_col = plain_cols[0]
        else:
            raise ValueError(
                "Could not auto-detect a score column. "
                f"Available columns: {list(scored_df.columns)}"
            )

    if score_col not in scored_df.columns:
        raise ValueError(
            f"Column '{score_col}' not found in scored DataFrame. "
            f"Available columns: {list(scored_df.columns)}"
        )

    if title is None:
        title = f"Distribution of {target_concept.title()} Scores"

    # --- Resolve split partner if compare_splits ---
    split_col: Optional[str] = None
    if compare_splits:
        # derive companion column name: *_Score_full -> *_Score_split
        candidate = score_col.replace("_Score_full", "_Score_split")
        if candidate in scored_df.columns:
            split_col = candidate
        else:
            warnings.warn(
                f"compare_splits=True but no matching split column found for '{score_col}'. "
                "Only the full-score distribution will be plotted.",
                UserWarning,
            )

    # ----------------------------------------------------------------
    # Plotly backend
    # ----------------------------------------------------------------
    if backend == "plotly":
        if split_col:
            # Melt to long form for side-by-side histogram
            df_long = pd.concat([
                scored_df[[score_col]].rename(columns={score_col: "score"}).assign(split="full"),
                scored_df[[split_col]].rename(columns={split_col: "score"}).assign(split="split"),
            ], ignore_index=True)
            fig = px.histogram(
                df_long,
                x="score", color="split",
                nbins=nbins,
                title=f"{title} (full vs split)",
                labels={"score": f"{target_concept.title()} Score"},
                marginal="box",
                barmode="overlay",
                opacity=0.7,
                template=template,
            )
        else:
            fig = px.histogram(
                scored_df,
                x=score_col,
                nbins=nbins,
                title=title,
                labels={score_col: f"{target_concept.title()} Score"},
                color_discrete_sequence=[color],
                marginal="box",
                template=template,
            )

            if show_stats:
                mean_score   = scored_df[score_col].mean()
                median_score = scored_df[score_col].median()
                fig.add_vline(
                    x=mean_score, line_dash="dash", line_color="red",
                    annotation_text=f"Mean: {mean_score:.3f}",
                    annotation_position="top right",
                )
                fig.add_vline(
                    x=median_score, line_dash="dot", line_color="orange",
                    annotation_text=f"Median: {median_score:.3f}",
                    annotation_position="top left",
                )
                stats_text = (
                    f"Mean: {mean_score:.3f}<br>"
                    f"Median: {median_score:.3f}<br>"
                    f"Std: {scored_df[score_col].std():.3f}<br>"
                    f"Count: {len(scored_df)}"
                )
                fig.add_annotation(
                    x=0.02, y=0.98, xref="paper", yref="paper",
                    text=stats_text, showarrow=False, font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="gray",
                    borderwidth=1, xanchor="left", yanchor="top",
                )

            fig.update_traces(
                hovertemplate=(
                    f"<b>{target_concept.title()} Score</b>: %{{x}}<br>"
                    "<b>Count</b>: %{y}<extra></extra>"
                )
            )

        fig.update_layout(yaxis_title="Frequency", bargap=0.02, hovermode="x unified")
        if return_fig:
            return fig
        fig.show()
        return None

    # ----------------------------------------------------------------
    # Seaborn backend
    # ----------------------------------------------------------------
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 5))
    if split_col:
        sns.histplot(
            data=scored_df, x=score_col, bins=nbins,
            label="full", ax=ax, color="steelblue", alpha=0.6,
        )
        sns.histplot(
            data=scored_df, x=split_col, bins=nbins,
            label="split", ax=ax, color="coral", alpha=0.6,
        )
        ax.legend(title="Split")
        ax.set_title(f"{title} (full vs split)")
    else:
        sns.histplot(data=scored_df, x=score_col, bins=nbins,
                     ax=ax, color=color, kde=True)
        ax.set_title(title)
        if show_stats:
            mean_score = scored_df[score_col].mean()
            ax.axvline(mean_score, color="red", linestyle="--", label=f"Mean={mean_score:.3f}")
            ax.legend()
    ax.set_xlabel(f"{target_concept.title()} Score")
    ax.set_ylabel("Count")
    plt.tight_layout()
    if return_fig:
        return fig
    plt.show()
    return None


# ---------------------------------------------------------------------------
# Comparison network (fix 1b: guard moved above dict lookup)
# ---------------------------------------------------------------------------

def plot_comparison_network(
    pairwise_df: pd.DataFrame,
    target_concept: str,
    centrality_measure: str = "pagerank",
    decision_col: str = "decision",
    return_fig: bool = False,
    scored_df: Optional[pd.DataFrame] = None,
    score_col: Optional[str] = None,
    item_id_name: Optional[str] = None,
    data_df: Optional[pd.DataFrame] = None,
    text_col: Optional[str] = None,
) -> Optional[go.Figure]:
    """
    Plot a directed pairwise-comparison network using Plotly.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise comparison DataFrame.
    target_concept : str
        Name of the measured concept (used in the plot title).
    centrality_measure : str, default 'pagerank'
        Node centrality measure: ``'pagerank'``, ``'in_degree'``,
        ``'out_degree'``, ``'betweenness'``, ``'eigenvector'``, ``'degree'``, ``'bradley_terry'``.
    decision_col : str, default 'decision'
        Column name for decisions.
    return_fig : bool
        If ``True``, returns the figure instead of calling ``.show()``.

    Returns
    -------
    plotly.graph_objects.Figure or None
    """
    import networkx as nx

    if pairwise_df is None:
        raise ValueError(
            "No pairwise comparison results found. "
            "Run generate_pairwise_annotations() first."
        )

    if decision_col not in pairwise_df.columns:
        raise ValueError(
            f"Decision column '{decision_col}' not found in pairwise_df."
        )

    # Fix 1b: validate centrality_measure BEFORE accessing the dict
    centrality_funcs = {
        "pagerank":   nx.pagerank,
        "in_degree":  nx.in_degree_centrality,
        "out_degree": nx.out_degree_centrality,
        "betweenness": nx.betweenness_centrality,
        "eigenvector": lambda G: nx.eigenvector_centrality(G, max_iter=1000),
        "degree":     nx.degree_centrality,
        "bradley_terry": None,  # Special case using scored_df
    }
    if centrality_measure not in centrality_funcs:
        raise ValueError(
            f"Unknown centrality measure: '{centrality_measure}'. "
            f"Valid options: {list(centrality_funcs)}"
        )

    # Warn about ties
    tie_values = ["Tie", "tie", 2, 0.5]
    if pairwise_df[decision_col].isin(tie_values).any():
        n_ties = pairwise_df[decision_col].isin(tie_values).sum()
        pct    = n_ties / len(pairwise_df) * 100
        warnings.warn(
            f"Network plot excludes {n_ties} tie decisions ({pct:.1f}%). "
            "Ties have no directional preference and cannot be shown as directed edges.",
            UserWarning,
        )

    # Build directed graph
    df = pairwise_df.copy()
    if decision_col != "decision":
        df["decision"] = df[decision_col]

    G = nx.DiGraph()
    for _, row in df.iterrows():
        if row["decision"] in ("Text1", 0):
            G.add_edge(row["item1"], row["item2"])
        elif row["decision"] in ("Text2", 1):
            G.add_edge(row["item2"], row["item1"])

    if len(G.nodes()) == 0:
        raise ValueError("No valid comparisons found to create network graph.")

    pos        = nx.spring_layout(G, seed=42)
    
    if centrality_measure == "bradley_terry":
        if scored_df is None:
            raise ValueError("scored_df must be provided to use 'bradley_terry' centrality.")
        
        if score_col is None:
            full_cols  = [c for c in scored_df.columns if c.endswith("_Score_full")]
            plain_cols = [c for c in scored_df.columns if c.endswith("_Score") and not c.endswith("_Score_split") and not c.endswith("_Score_full")]
            if full_cols:
                score_col = full_cols[0]
                if len(full_cols) > 1:
                    warnings.warn(
                        f"Multiple score columns found: {full_cols}. "
                        f"Using '{score_col}'.",
                        UserWarning,
                    )
            elif plain_cols:
                score_col = plain_cols[0]
                if len(plain_cols) > 1:
                    warnings.warn(
                        f"Multiple score columns found: {plain_cols}. "
                        f"Using '{score_col}'.",
                        UserWarning,
                    )
            else:
                raise ValueError("Could not auto-detect a score column in scored_df.")

        if score_col not in scored_df.columns:
            raise ValueError(f"Column '{score_col}' not found in scored DataFrame.")
            
        if item_id_name is not None and item_id_name in scored_df.columns:
            score_map = dict(zip(scored_df[item_id_name], scored_df[score_col]))
        else:
            # Fallback to index if item_id_name is not clear or missing
            score_map = scored_df[score_col].to_dict()
            
        # Default to 0 if a node is missing from scored_df
        node_color = [score_map.get(node, 0.0) for node in G.nodes()]
    else:
        centrality = centrality_funcs[centrality_measure](G)
        node_color = [centrality[node] for node in G.nodes()]

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none", mode="lines",
    )
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    colorbar_title = centrality_measure.replace("_", " ").title()

    hover_text = []
    if data_df is not None and text_col is not None and item_id_name is not None and item_id_name in data_df.columns and text_col in data_df.columns:
        text_map = dict(zip(data_df[item_id_name], data_df[text_col]))
        for n in G.nodes():
            raw_text = str(text_map.get(n, n))
            wrapped = "<br>".join(textwrap.wrap(raw_text, width=60))
            hover_text.append(f"<b>{n}</b><br><br>{wrapped}")
    else:
        for n in G.nodes():
            hover_text.append(str(n))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers", hoverinfo="text",
        marker=dict(
            showscale=True, colorscale="Viridis",
            color=node_color, size=10,
            colorbar=dict(title=colorbar_title),
            line_width=2,
        ),
        text=hover_text,
    )

    # Arrows layout annotations
    annotations = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        annotations.append(
            dict(
                ax=x0, ay=y0, axref='x', ayref='y',
                x=x1, y=y1, xref='x', yref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor='#888',
                standoff=10
            )
        )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"<br>Pairwise Comparison Network – {target_concept.title()}",
            showlegend=False, hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    if return_fig:
        return fig
    fig.show()
    return None


# ---------------------------------------------------------------------------
# Epsilon sensitivity (5d: redirects alt_test prints to logging inside loop)
# ---------------------------------------------------------------------------

def plot_epsilon_sensitivity(
    pairwise_df: pd.DataFrame,
    annotator_cols: List[str],
    item_id_cols: List[str],
    annotated: bool,
    llm_annotator_cols: List[str],
    epsilon_range: Tuple[float, float] = (-0.1, 0.25),
    epsilon_step: float = 0.01,
    test_all_llms: bool = True,
    figsize: Tuple[float, float] = (12, 7),
    style: str = "whitegrid",
    palette: str = "husl",
    show_annotations: bool = True,
    return_data: bool = False,
) -> Optional[Tuple]:
    """
    Plot winning rate as a function of epsilon for the ALT-TEST.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
    annotator_cols : list of str
    item_id_cols : list of str
    annotated : bool
    llm_annotator_cols : list of str
    epsilon_range : tuple
    epsilon_step : float
    test_all_llms : bool
    figsize : tuple
    style : str
    palette : str
    show_annotations : bool
    return_data : bool

    Returns
    -------
    None, or (fig, df_plot) if return_data=True
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    from .validation import alt_test as _alt_test

    if not annotated:
        raise ValueError("Data must have human annotations to perform epsilon sensitivity analysis.")
    if not annotator_cols:
        raise ValueError("No annotator columns found for human annotations.")
    if not llm_annotator_cols:
        raise ValueError("No LLM annotator columns found.")

    eps_values = np.arange(epsilon_range[0], epsilon_range[1] + epsilon_step, epsilon_step)
    print(f"Testing {len(eps_values)} epsilon values from {epsilon_range[0]} to {epsilon_range[1]}...")

    all_results: List = []
    for eps in eps_values:
        try:
            # 5d: suppress verbose stdout from alt_test; capture via logging
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = _alt_test(
                    pairwise_df=pairwise_df,
                    annotator_cols=annotator_cols,
                    item_id_cols=item_id_cols,
                    annotated=annotated,
                    epsilon=eps,
                    test_all_llms=test_all_llms,
                    decision_col=None if test_all_llms else None,
                )
            logger.debug(buf.getvalue())
            all_results.append(result)
        except Exception as exc:
            logger.warning(f"Failed at epsilon={eps:.3f}: {exc}")
            all_results.append(None)

    # Determine model names from first valid result
    if test_all_llms:
        model_names = (
            list(all_results[0].keys())
            if all_results[0] and isinstance(all_results[0], dict)
            else []
        )
    else:
        model_names = ["default"]

    rows: List[dict] = []
    for eps, result in zip(eps_values, all_results):
        if result is None:
            continue
        if test_all_llms and isinstance(result, dict):
            for model, (wr, _) in result.items():
                rows.append({"epsilon": eps, "model": model, "winning_rate": wr})
        else:
            wr = result[0] if isinstance(result, tuple) else result
            rows.append({"epsilon": eps, "model": "default", "winning_rate": wr})

    df_plot = pd.DataFrame(rows)

    sns.set_style(style)
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette(palette, n_colors=len(model_names))

    for idx, model in enumerate(model_names):
        df_m = df_plot[df_plot["model"] == model]
        if df_m.empty:
            continue
        ax.plot(
            df_m["epsilon"], df_m["winning_rate"],
            label=model, color=colors[idx], linewidth=2, marker="o",
            markersize=3, alpha=0.85,
        )

    if show_annotations:
        reference_lines = [
            (0.05, "Crowdworkers", "#e74c3c"),
            (0.10, "Trained annotators", "#3498db"),
            (0.20, "Experts", "#27ae60"),
        ]
        for eps_val, label, col_ in reference_lines:
            if epsilon_range[0] <= eps_val <= epsilon_range[1]:
                ax.axvline(x=eps_val, color=col_, linestyle=":", linewidth=2, alpha=0.6)
                y_pos = ax.get_ylim()[1] * 0.95
                ax.text(
                    eps_val, y_pos, f" {label}\n ε={eps_val}",
                    verticalalignment="top", horizontalalignment="left",
                    fontsize=9, color=col_, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=col_, alpha=0.8),
                )

    ax.set_xlabel("Epsilon (ε)", fontsize=12, weight="bold")
    ax.set_ylabel("Winning Rate (ω)", fontsize=12, weight="bold")
    ax.set_title(
        f"Epsilon Sensitivity Analysis: {' '.join(w.title() for w in ''.split())}\n"
        "Winning Rate vs. Epsilon Threshold",
        fontsize=14, weight="bold", pad=20,
    )
    ax.legend(title="Model", title_fontsize=11, fontsize=10, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(epsilon_range[0] - 0.01, epsilon_range[1] + 0.01)
    plt.tight_layout()

    if return_data:
        return fig, df_plot
    plt.show()
    return None
