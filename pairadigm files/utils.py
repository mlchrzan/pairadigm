"""
Helper utility functions for Pairadigm.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def validate_dataframe_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    df_name: str = "DataFrame"
) -> None:
    """
    Validate that DataFrame contains required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : List[str]
        List of required column names
    df_name : str
        Name for error messages
        
    Raises
    ------
    ValueError
        If required columns are missing
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def normalize_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize decision labels to consistent format.
    
    Converts various decision formats to 'Text1' or 'Text2'.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'decision' column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with normalized decisions
    """
    if 'decision' not in df.columns:
        raise ValueError("DataFrame must have 'decision' column")
    
    df = df.copy()
    
    # Mapping of various formats
    mapping = {
        '1': 'Text1',
        '2': 'Text2',
        'text1': 'Text1',
        'text2': 'Text2',
        'description 1': 'Text1',
        'description 2': 'Text2',
        'item1': 'Text1',
        'item2': 'Text2',
        'a': 'Text1',
        'b': 'Text2'
    }
    
    df['decision'] = df['decision'].astype(str).str.lower().str.strip()
    df['decision'] = df['decision'].map(mapping).fillna(df['decision'])
    
    return df


def calculate_agreement_matrix(
    annotators_dict: Dict[str, pd.DataFrame],
    pair_col: str = 'pair'
) -> pd.DataFrame:
    """
    Calculate pairwise agreement matrix between multiple annotators.
    
    Parameters
    ----------
    annotators_dict : Dict[str, pd.DataFrame]
        Dictionary mapping annotator names to their annotation DataFrames
    pair_col : str
        Column name for pair identifiers
        
    Returns
    -------
    pd.DataFrame
        Agreement matrix (annotators x annotators)
    """
    from .validation import compare_annotators
    
    annotators = list(annotators_dict.keys())
    n = len(annotators)
    agreement_matrix = pd.DataFrame(
        index=annotators,
        columns=annotators,
        dtype=float
    )
    
    for i, ann1 in enumerate(annotators):
        for j, ann2 in enumerate(annotators):
            if i == j:
                agreement_matrix.loc[ann1, ann2] = 1.0
            elif i < j:
                result = compare_annotators(
                    annotators_dict[ann1],
                    annotators_dict[ann2],
                    ann1,
                    ann2
                )
                agreement_matrix.loc[ann1, ann2] = result['agreement_rate']
                agreement_matrix.loc[ann2, ann1] = result['agreement_rate']
    
    return agreement_matrix


def identify_difficult_pairs(
    annotated_df: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Identify pairs with low inter-annotator agreement or close scores.
    
    Parameters
    ----------
    annotated_df : pd.DataFrame
        Annotated pairings
    threshold : float
        Agreement threshold below which pairs are considered difficult
        
    Returns
    -------
    pd.DataFrame
        Difficult pairs with metadata
    """
    # This is a placeholder - implement based on your specific needs
    # Could analyze justification length, decision confidence, etc.
    difficult = annotated_df[
        annotated_df['decision'] == 'ERROR'
    ].copy()
    
    return difficult


def export_for_human_annotation(
    pairings_df: pd.DataFrame,
    output_path: str,
    format: str = 'csv'
) -> None:
    """
    Export pairings in format suitable for human annotation.
    
    Parameters
    ----------
    pairings_df : pd.DataFrame
        Pairings to export
    output_path : str
        Output file path
    format : str
        Export format ('csv' or 'excel')
    """
    # Create clean export with just necessary columns
    export_df = pairings_df[['item1', 'item2', 'breakdown1', 'breakdown2']].copy()
    export_df['decision'] = ''
    export_df['notes'] = ''
    
    if format == 'csv':
        export_df.to_csv(output_path, index=False)
    elif format == 'excel':
        export_df.to_excel(output_path, index=False, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"✓ Exported {len(export_df)} pairs to {output_path}")


def merge_human_llm_annotations(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge human and LLM annotations for comparison.
    
    Parameters
    ----------
    human_df : pd.DataFrame
        Human annotations
    llm_df : pd.DataFrame
        LLM annotations
        
    Returns
    -------
    pd.DataFrame
        Merged annotations with both decisions
    """
    # Create canonical pair representation
    for df in [human_df, llm_df]:
        df['pair'] = df.apply(
            lambda row: tuple(sorted([row['item1'], row['item2']])),
            axis=1
        )
    
    merged = human_df.merge(
        llm_df,
        on='pair',
        suffixes=('_human', '_llm'),
        how='inner'
    )
    
    return merged


def sample_for_validation(
    df: pd.DataFrame,
    n_samples: int,
    stratify_col: Optional[str] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample items for validation annotation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    n_samples : int
        Number of samples to draw
    stratify_col : str, optional
        Column to stratify sampling by
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Sampled subset
    """
    if stratify_col and stratify_col in df.columns:
        sample = df.groupby(stratify_col, group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), n_samples // df[stratify_col].nunique()),
                random_state=random_state
            )
        )
    else:
        sample = df.sample(n=min(n_samples, len(df)), random_state=random_state)
    
    return sample.reset_index(drop=True)


def check_data_quality(df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Check data quality and return diagnostics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check
    verbose : bool
        Whether to print diagnostics
        
    Returns
    -------
    Dict
        Quality metrics
    """
    metrics = {
        'total_rows': len(df),
        'duplicate_rows': df.duplicated().sum(),
        'missing_values': df.isnull().sum().to_dict(),
        'column_types': df.dtypes.to_dict()
    }
    
    # Check for decision column
    if 'decision' in df.columns:
        metrics['decision_distribution'] = df['decision'].value_counts().to_dict()
        metrics['error_rate'] = (df['decision'] == 'ERROR').sum() / len(df)
    
    if verbose:
        print("="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        print(f"Total rows: {metrics['total_rows']}")
        print(f"Duplicate rows: {metrics['duplicate_rows']}")
        print(f"\nMissing values:")
        for col, count in metrics['missing_values'].items():
            if count > 0:
                print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
        
        if 'decision_distribution' in metrics:
            print(f"\nDecision distribution:")
            for decision, count in metrics['decision_distribution'].items():
                print(f"  {decision}: {count} ({count/len(df)*100:.1f}%)")
        
        print("="*60)
    
    return metrics


def create_annotation_report(
    pairadigm_instance,
    output_path: str = 'annotation_report.html'
) -> None:
    """
    Create an HTML report summarizing annotation results.
    
    Parameters
    ----------
    pairadigm_instance : Pairadigm
        Pairadigm instance with results
    output_path : str
        Output HTML file path
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pairadigm Annotation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .pass {{ color: #27ae60; font-weight: bold; }}
            .fail {{ color: #e74c3c; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Pairadigm Annotation Report</h1>
        <p><strong>Target Concept:</strong> {pairadigm_instance.target_concept}</p>
        <p><strong>Model:</strong> {pairadigm_instance.model_name}</p>
        
        <h2>Dataset Summary</h2>
        <div class="metric">
            <p><strong>Total Items:</strong> {len(pairadigm_instance.data)}</p>
            <p><strong>Total Comparisons:</strong> {len(pairadigm_instance.pairwise_df) if pairadigm_instance.pairwise_df is not None else 0}</p>
        </div>
    """
    
    if pairadigm_instance.scored_df is not None:
        scores = pairadigm_instance.scored_df['Bradley_Terry_Score']
        html += f"""
        <h2>Score Distribution</h2>
        <div class="metric">
            <p><strong>Mean:</strong> {scores.mean():.3f}</p>
            <p><strong>Median:</strong> {scores.median():.3f}</p>
            <p><strong>Std Dev:</strong> {scores.std():.3f}</p>
            <p><strong>Range:</strong> [{scores.min():.3f}, {scores.max():.3f}]</p>
        </div>
        """
    
    if pairadigm_instance.validation_results:
        html += "<h2>Validation Results</h2>"
        for key, value in pairadigm_instance.validation_results.items():
            html += f"<div class='metric'><strong>{key}:</strong> {value}</div>"
    
    html += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✓ Report saved to {output_path}")