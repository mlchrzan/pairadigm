"""
Pairing generation utilities for creating connected comparison graphs.
"""

import random
import itertools
import pandas as pd
from typing import List, Optional, Set, Dict


def pair_items(
    items: List,
    num_pairs_per_item: Optional[int] = 10,
    random_seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate a connected subset of pairwise comparisons.
    
    Creates pairs ensuring:
    1. Graph connectivity (all items reachable from any item)
    2. Minimum number of comparisons per item
    3. Balanced comparison distribution
    
    Parameters
    ----------
    items : List
        Items to compare
    num_pairs_per_item : int, optional
        Minimum pairs per item. If None, uses adaptive formula
    random_seed : int, optional
        For reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['item1', 'item2'] representing pairings
        
    Examples
    --------
    >>> items = [1, 2, 3, 4, 5]
    >>> pairs = pair_items(items, num_pairs_per_item=3)
    >>> len(pairs) >= 5 * 3 / 2  # At least min_pairs * n / 2
    True
    
    Notes
    -----
    The function uses a two-phase approach:
    1. Creates a spanning chain for connectivity
    2. Adds random pairs until minimum coverage is met
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    n = len(items)
    
    if n < 2:
        return pd.DataFrame(columns=['item1', 'item2'])
    
    # Determine minimum pairs per item
    if num_pairs_per_item is None:
        # Adaptive: more items = relatively fewer pairs needed
        min_pairs = max(3, min(6, int(n ** 0.5)))
    else:
        min_pairs = num_pairs_per_item
    
    # Generate all possible pairs
    all_pairs = set(itertools.combinations(items, 2))
    chosen_pairs: Set[tuple] = set()
    covered: Dict = {item: set() for item in items}
    
    # Phase 1: Create spanning chain for connectivity
    # This ensures graph is connected
    shuffled_items = items.copy()
    random.shuffle(shuffled_items)
    
    for i in range(n - 1):
        pair = tuple(sorted((shuffled_items[i], shuffled_items[i + 1])))
        chosen_pairs.add(pair)
        covered[shuffled_items[i]].add(shuffled_items[i + 1])
        covered[shuffled_items[i + 1]].add(shuffled_items[i])
    
    # Phase 2: Add pairs until minimum coverage
    additional_pairs = list(all_pairs - chosen_pairs)
    random.shuffle(additional_pairs)
    
    for a, b in additional_pairs:
        # Add if either item needs more pairs
        if len(covered[a]) < min_pairs or len(covered[b]) < min_pairs:
            chosen_pairs.add((a, b))
            covered[a].add(b)
            covered[b].add(a)
            
            # Early exit if all items have enough pairs
            if all(len(covered[item]) >= min_pairs for item in items):
                break
    
    # Convert to DataFrame
    pairs_list = [{'item1': a, 'item2': b} for a, b in chosen_pairs]
    df = pd.DataFrame(pairs_list)
    
    return df


def generate_pairings_df(
    df: pd.DataFrame,
    row_id: str,
    num_pairs_per_item: int = 10,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate pairings for items in a DataFrame with associated breakdowns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing items to pair
    row_id : str
        Column name with unique item identifiers
    num_pairs_per_item : int, optional
        Minimum pairs per item
    random_seed : int, optional
        For reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with pairings and associated breakdowns
        Columns: ['item1', 'item2', 'breakdown1', 'breakdown2']
        
    Raises
    ------
    ValueError
        If required columns are missing
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3],
    ...     'CGCoT_Breakdown': ['breakdown A', 'breakdown B', 'breakdown C']
    ... })
    >>> pairs = generate_pairings_df(df, row_id='id', num_pairs_per_item=2)
    >>> 'breakdown1' in pairs.columns
    True
    """
    if row_id not in df.columns:
        raise ValueError(f"Column '{row_id}' not found in DataFrame")
    
    if "CGCoT_Breakdown" not in df.columns:
        raise ValueError(
            "Column 'CGCoT_Breakdown' not found in DataFrame. "
            "Generate breakdowns first using generate_breakdowns()"
        )
    
    # Generate pairings based on item IDs
    uuid_pairings = pair_items(
        df[row_id].tolist(),
        num_pairs_per_item=num_pairs_per_item,
        random_seed=random_seed
    )
    
    # Map IDs to breakdowns
    uuid_to_breakdown = dict(zip(df[row_id], df['CGCoT_Breakdown']))
    
    uuid_pairings['breakdown1'] = uuid_pairings['item1'].map(uuid_to_breakdown)
    uuid_pairings['breakdown2'] = uuid_pairings['item2'].map(uuid_to_breakdown)
    
    # Check for any missing breakdowns
    missing_breakdowns = (
        uuid_pairings['breakdown1'].isna().sum() + 
        uuid_pairings['breakdown2'].isna().sum()
    )
    
    if missing_breakdowns > 0:
        raise ValueError(
            f"Found {missing_breakdowns} missing breakdowns. "
            "Ensure all items have CGCoT breakdowns."
        )
    
    return uuid_pairings


def validate_pairing_coverage(
    pairings_df: pd.DataFrame,
    all_items: List
) -> Dict:
    """
    Validate that all items are adequately covered in pairings.
    
    Parameters
    ----------
    pairings_df : pd.DataFrame
        DataFrame with 'item1' and 'item2' columns
    all_items : List
        List of all items that should be covered
        
    Returns
    -------
    Dict
        Coverage statistics including min/max/mean pairs per item
        
    Examples
    --------
    >>> pairs = pd.DataFrame({'item1': [1, 2], 'item2': [2, 3]})
    >>> stats = validate_pairing_coverage(pairs, [1, 2, 3])
    >>> stats['coverage_rate']
    1.0
    """
    # Count pairs per item
    item_counts = {}
    for item in all_items:
        count = (
            (pairings_df['item1'] == item).sum() +
            (pairings_df['item2'] == item).sum()
        )
        item_counts[item] = count
    
    covered_items = [item for item, count in item_counts.items() if count > 0]
    
    stats = {
        'total_pairs': len(pairings_df),
        'total_items': len(all_items),
        'covered_items': len(covered_items),
        'coverage_rate': len(covered_items) / len(all_items),
        'min_pairs_per_item': min(item_counts.values()) if item_counts else 0,
        'max_pairs_per_item': max(item_counts.values()) if item_counts else 0,
        'mean_pairs_per_item': sum(item_counts.values()) / len(all_items) if all_items else 0,
        'uncovered_items': [item for item in all_items if item_counts.get(item, 0) == 0]
    }
    
    return stats


def stratified_pairing(
    df: pd.DataFrame,
    row_id: str,
    stratify_col: str,
    num_pairs_per_item: int = 10,
    within_strata_ratio: float = 0.7,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate pairings with stratification to ensure balanced comparisons.
    
    Creates pairs both within and across strata to ensure:
    1. Items are compared to similar items (within-strata)
    2. Items are also compared across groups (cross-strata)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing items
    row_id : str
        Column name with unique identifiers
    stratify_col : str
        Column to stratify by (e.g., 'category', 'source')
    num_pairs_per_item : int
        Target pairs per item
    within_strata_ratio : float
        Proportion of pairs within same stratum (0-1)
    random_seed : int
        For reproducibility
        
    Returns
    -------
    pd.DataFrame
        Stratified pairings with breakdowns
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'id': [1, 2, 3, 4],
    ...     'category': ['A', 'A', 'B', 'B'],
    ...     'CGCoT_Breakdown': ['...', '...', '...', '...']
    ... })
    >>> pairs = stratified_pairing(df, 'id', 'category', num_pairs_per_item=2)
    """
    if stratify_col not in df.columns:
        raise ValueError(f"Column '{stratify_col}' not found in DataFrame")
    
    if not 0 <= within_strata_ratio <= 1:
        raise ValueError("within_strata_ratio must be between 0 and 1")
    
    random.seed(random_seed)
    
    # Calculate target pairs
    within_pairs_target = int(num_pairs_per_item * within_strata_ratio)
    cross_pairs_target = num_pairs_per_item - within_pairs_target
    
    all_pairings = []
    
    # Generate within-strata pairs
    for stratum in df[stratify_col].unique():
        stratum_items = df[df[stratify_col] == stratum][row_id].tolist()
        
        if len(stratum_items) > 1:
            stratum_pairs = pair_items(
                stratum_items,
                num_pairs_per_item=within_pairs_target,
                random_seed=random_seed
            )
            all_pairings.append(stratum_pairs)
    
    # Generate cross-strata pairs
    if cross_pairs_target > 0:
        all_items = df[row_id].tolist()
        cross_pairs = pair_items(
            all_items,
            num_pairs_per_item=cross_pairs_target,
            random_seed=random_seed + 1
        )
        
        # Filter to only cross-strata pairs
        item_to_stratum = dict(zip(df[row_id], df[stratify_col]))
        cross_pairs['stratum1'] = cross_pairs['item1'].map(item_to_stratum)
        cross_pairs['stratum2'] = cross_pairs['item2'].map(item_to_stratum)
        cross_pairs = cross_pairs[cross_pairs['stratum1'] != cross_pairs['stratum2']]
        cross_pairs = cross_pairs[['item1', 'item2']]
        
        all_pairings.append(cross_pairs)
    
    # Combine all pairings
    combined_pairings = pd.concat(all_pairings, ignore_index=True)
    combined_pairings = combined_pairings.drop_duplicates()
    
    # Add breakdowns
    if 'CGCoT_Breakdown' in df.columns:
        uuid_to_breakdown = dict(zip(df[row_id], df['CGCoT_Breakdown']))
        combined_pairings['breakdown1'] = combined_pairings['item1'].map(uuid_to_breakdown)
        combined_pairings['breakdown2'] = combined_pairings['item2'].map(uuid_to_breakdown)
    
    return combined_pairings


def balanced_incomplete_block_design(
    items: List,
    block_size: int = 3,
    replications: int = 2
) -> pd.DataFrame:
    """
    Generate pairings using Balanced Incomplete Block Design (BIBD).
    
    Useful for creating highly balanced comparison designs where each
    item is compared a fixed number of times.
    
    Parameters
    ----------
    items : List
        Items to compare
    block_size : int
        Number of items per block
    replications : int
        Number of times each item appears
        
    Returns
    -------
    pd.DataFrame
        Pairings from BIBD design
        
    Notes
    -----
    This creates a more balanced design than random pairing but may not
    always be possible for all parameter combinations.
    """
    # Simple implementation: create blocks and extract pairs
    n = len(items)
    blocks = []
    
    # Create replications
    for r in range(replications):
        random.seed(42 + r)
        shuffled = items.copy()
        random.shuffle(shuffled)
        
        # Create blocks of specified size
        for i in range(0, n, block_size):
            block = shuffled[i:i + block_size]
            if len(block) >= 2:
                blocks.append(block)
    
    # Extract all pairs from blocks
    all_pairs = set()
    for block in blocks:
        for pair in itertools.combinations(sorted(block), 2):
            all_pairs.add(pair)
    
    df = pd.DataFrame(list(all_pairs), columns=['item1', 'item2'])
    return df