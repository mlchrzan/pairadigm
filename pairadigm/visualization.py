"""
Visualization functions for Pairadigm results.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, List


def plot_score_distribution(
    results_df: pd.DataFrame,
    score_col: str = 'Bradley_Terry_Score',
    title: str = 'Distribution of Bradley-Terry Scores',
    package: str = 'plotly',
    bins: int = 30,
    color: str = 'skyblue'
):
    """
    Plot histogram of Bradley-Terry scores.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing the scores
    score_col : str, default='Bradley_Terry_Score'
        Column name for the Bradley-Terry scores
    title : str
        Title for the plot
    package : str, default='plotly'
        Plotting package to use ('plotly' or 'matplotlib')
    bins : int, default=30
        Number of histogram bins
    color : str, default='skyblue'
        Color for the histogram
        
    Examples
    --------
    >>> plot_score_distribution(scored_df, title='Political Bias Scores')
    """
    if score_col not in results_df.columns:
        raise ValueError(f"Column '{score_col}' not found in DataFrame")
    
    scores = results_df[score_col].dropna()
    
    if package == 'matplotlib':
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=bins, color=color, edgecolor='black', alpha=0.7)
        
        # Add mean line
        mean_score = scores.mean()
        plt.axvline(mean_score, color='green', linestyle='dashed', linewidth=2, 
                   label=f'Mean: {mean_score:.3f}')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Bradley-Terry Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    elif package == 'plotly':
        mean_score = scores.mean()
        
        fig = px.histogram(
            results_df,
            x=score_col,
            nbins=bins,
            title=title,
            labels={score_col: 'Bradley-Terry Score'},
            color_discrete_sequence=[color]
        )
        
        # Add mean line
        fig.add_vline(
            x=mean_score,
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text=f"Mean: {mean_score:.3f}",
            annotation_position="top right"
        )
        
        fig.update_layout(
            yaxis_title='Frequency',
            bargap=0.05,
            template='plotly_white',
            font=dict(size=12),
            showlegend=False
        )
        
        fig.show()
        
    else:
        raise ValueError(f"Unsupported package: {package}. Use 'plotly' or 'matplotlib'")


def plot_comparison_network(
    pairwise_df: pd.DataFrame,
    layout: str = 'spring',
    node_size_col: Optional[str] = None,
    title: str = 'Pairwise Comparison Network'
):
    """
    Plot network graph of pairwise comparisons.
    
    Nodes represent items, edges represent comparisons. Edge direction
    indicates which item was preferred.
    
    Parameters
    ----------
    pairwise_df : pd.DataFrame
        DataFrame containing pairwise comparison results
        Must have columns: ['item1', 'item2', 'decision']
    layout : str, default='spring'
        Network layout algorithm ('spring', 'circular', 'kamada_kawai')
    node_size_col : str, optional
        Column name for sizing nodes (e.g., 'Bradley_Terry_Score')
    title : str
        Title for the plot
        
    Examples
    --------
    >>> plot_comparison_network(pairwise_df)
    """
    required_cols = ['item1', 'item2', 'decision']
    missing_cols = [col for col in required_cols if col not in pairwise_df.columns]
    if missing_cols:
        raise ValueError(f"pairwise_df missing required columns: {missing_cols}")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges based on decisions
    for _, row in pairwise_df.iterrows():
        if row['decision'] == 'Text1':
            G.add_edge(row['item1'], row['item2'])
        elif row['decision'] == 'Text2':
            G.add_edge(row['item2'], row['item1'])
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42, k=0.5)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Calculate node colors based on centrality
    centrality = nx.degree_centrality(G)
    node_color = [centrality[node] for node in G.nodes()]
    
    # Prepare edge coordinates
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Prepare node coordinates
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # Create hover text
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        node_text.append(
            f'Item: {node}<br>'
            f'Wins: {out_degree}<br>'
            f'Losses: {in_degree}<br>'
            f'Win Rate: {out_degree/(in_degree+out_degree):.2%}' 
            if (in_degree + out_degree) > 0 else f'Item: {node}<br>No comparisons'
        )
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=15,
            colorbar=dict(
                title='Centrality',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )
    
    fig.show()


def plot_score_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    score_col: str = 'Bradley_Terry_Score',
    label1: str = 'Distribution 1',
    label2: str = 'Distribution 2',
    plot_type: str = 'box'
):
    """
    Compare two score distributions visually.
    
    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames with scores to compare
    score_col : str
        Column with scores
    label1, label2 : str
        Labels for the distributions
    plot_type : str, default='box'
        Type of plot ('box', 'violin', 'hist')
    """
    scores1 = df1[score_col].dropna()
    scores2 = df2[score_col].dropna()
    
    # Combine data
    combined_df = pd.DataFrame({
        'Score': pd.concat([scores1, scores2]),
        'Group': [label1] * len(scores1) + [label2] * len(scores2)
    })
    
    if plot_type == 'box':
        fig = px.box(
            combined_df,
            x='Group',
            y='Score',
            title=f'Score Comparison: {label1} vs {label2}',
            color='Group',
            color_discrete_sequence=['skyblue', 'lightcoral']
        )
    elif plot_type == 'violin':
        fig = px.violin(
            combined_df,
            x='Group',
            y='Score',
            title=f'Score Distribution: {label1} vs {label2}',
            color='Group',
            box=True,
            color_discrete_sequence=['skyblue', 'lightcoral']
        )
    elif plot_type == 'hist':
        fig = px.histogram(
            combined_df,
            x='Score',
            color='Group',
            title=f'Score Distribution: {label1} vs {label2}',
            barmode='overlay',
            opacity=0.7,
            color_discrete_sequence=['skyblue', 'lightcoral']
        )
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type}")
    
    fig.update_layout(template='plotly_white')
    fig.show()


def plot_score_vs_feature(
    df: pd.DataFrame,
    score_col: str = 'Bradley_Terry_Score',
    feature_col: str = None,
    plot_type: str = 'scatter'
):
    """
    Plot scores against another feature.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with scores and features
    score_col : str
        Column with scores
    feature_col : str
        Column with feature to plot against
    plot_type : str, default='scatter'
        Type of plot ('scatter', 'box', 'bar')
    """
    if feature_col is None:
        raise ValueError("feature_col must be specified")
    
    if feature_col not in df.columns:
        raise ValueError(f"Column '{feature_col}' not found in DataFrame")
    
    if plot_type == 'scatter':
        fig = px.scatter(
            df,
            x=feature_col,
            y=score_col,
            title=f'{score_col} vs {feature_col}',
            trendline='ols',
            opacity=0.6
        )
    elif plot_type == 'box':
        fig = px.box(
            df,
            x=feature_col,
            y=score_col,
            title=f'{score_col} by {feature_col}',
            color=feature_col
        )
    elif plot_type == 'bar':
        # Aggregate by feature
        agg_df = df.groupby(feature_col)[score_col].mean().reset_index()
        fig = px.bar(
            agg_df,
            x=feature_col,
            y=score_col,
            title=f'Mean {score_col} by {feature_col}'
        )
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type}")
    
    fig.update_layout(template='plotly_white')
    fig.show()


def plot_ranking(
    df: pd.DataFrame,
    score_col: str = 'Bradley_Terry_Score',
    label_col: Optional[str] = None,
    top_n: int = 20,
    title: str = 'Top Items by Score'
):
    """
    Plot horizontal bar chart of top-ranked items.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with scores
    score_col : str
        Column with scores
    label_col : str, optional
        Column for item labels (defaults to index)
    top_n : int, default=20
        Number of top items to show
    title : str
        Plot title
    """
    # Sort and get top N
    df_sorted = df.sort_values(score_col, ascending=False).head(top_n)
    
    if label_col is None:
        labels = df_sorted.index.astype(str)
    else:
        labels = df_sorted[label_col].astype(str)
    
    # Truncate long labels
    labels = [label[:50] + '...' if len(label) > 50 else label for label in labels]
    
    fig = go.Figure(go.Bar(
        x=df_sorted[score_col],
        y=labels,
        orientation='h',
        marker=dict(
            color=df_sorted[score_col],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Score')
        )
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=score_col,
        yaxis_title='Item',
        height=max(400, top_n * 25),
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    fig.show()


def plot_transitivity_violations(
    pairwise_df: pd.DataFrame,
    violations: List[tuple],
    title: str = 'Intransitive Triples'
):
    """
    Visualize intransitive triples in the comparison network.
    
    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise comparisons
    violations : List[tuple]
        List of (a, b, c) tuples representing violations where a>b, b>c, but c>a
    title : str
        Plot title
    """
    if not violations:
        print("No transitivity violations to plot")
        return
    
    # Create graph with violations highlighted
    G = nx.DiGraph()
    
    # Add all edges
    for _, row in pairwise_df.iterrows():
        if row['decision'] == 'Text1':
            G.add_edge(row['item1'], row['item2'], violation=False)
        elif row['decision'] == 'Text2':
            G.add_edge(row['item2'], row['item1'], violation=False)
    
    # Mark violation edges
    for a, b, c in violations:
        if G.has_edge(a, b):
            G[a][b]['violation'] = True
        if G.has_edge(b, c):
            G[b][c]['violation'] = True
        if G.has_edge(c, a):
            G[c][a]['violation'] = True
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Separate regular and violation edges
    regular_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('violation', False)]
    violation_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('violation', False)]
    
    # Plot with matplotlib for better control
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, alpha=0.9)
    
    # Draw regular edges
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, 
                          edge_color='gray', alpha=0.3, arrows=True)
    
    # Draw violation edges
    nx.draw_networkx_edges(G, pos, edgelist=violation_edges,
                          edge_color='red', width=2, alpha=0.8, arrows=True)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title(f'{title}\n{len(violations)} violations found', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_annotator_agreement_matrix(
    agreement_matrix: pd.DataFrame,
    title: str = 'Inter-Annotator Agreement Matrix'
):
    """
    Plot heatmap of annotator agreement.
    
    Parameters
    ----------
    agreement_matrix : pd.DataFrame
        Square matrix of agreement rates between annotators
    title : str
        Plot title
    """
    fig = go.Figure(data=go.Heatmap(
        z=agreement_matrix.values,
        x=agreement_matrix.columns,
        y=agreement_matrix.index,
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        text=agreement_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title='Agreement Rate')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Annotator',
        yaxis_title='Annotator',
        template='plotly_white'
    )
    
    fig.show()


def build_comparison_graph(pairwise_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build NetworkX directed graph from pairwise comparisons.
    
    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise comparisons with 'item1', 'item2', 'decision' columns
        
    Returns
    -------
    nx.DiGraph
        Directed graph of comparisons
    """
    G = nx.DiGraph()
    
    for _, row in pairwise_df.iterrows():
        if row['decision'] == 'Text1':
            G.add_edge(row['item1'], row['item2'])
        elif row['decision'] == 'Text2':
            G.add_edge(row['item2'], row['item1'])
    
    return G