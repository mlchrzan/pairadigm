"""
Unit tests for Pairadigm core functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pairadigm import Pairadigm
from pairadigm.validation import check_transitivity, compare_annotators
from pairadigm.pairing import pair_items
from pairadigm.scoring import bradley_terry_scores


class TestPairadigmInitialization:
    """Test Pairadigm class initialization."""
    
    def test_valid_initialization(self):
        """Test valid initialization."""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'text': ['Text A', 'Text B', 'Text C']
        })
        
        pairadigm = Pairadigm(
            data=data,
            row_id='id',
            row_name='text'
        )
        
        assert len(pairadigm.data) == 3
        assert pairadigm.row_id == 'id'
        assert pairadigm.row_name == 'text'
    
    def test_invalid_dataframe(self):
        """Test initialization with invalid DataFrame."""
        with pytest.raises(TypeError):
            Pairadigm(data="not a dataframe", row_id='id')
    
    def test_missing_row_id(self):
        """Test initialization with missing row_id column."""
        data = pd.DataFrame({'other_col': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            Pairadigm(data=data, row_id='id')


class TestPairing:
    """Test pairing functionality."""
    
    def test_pair_items_basic(self):
        """Test basic pair generation."""
        items = [1, 2, 3, 4, 5]
        pairs_df = pair_items(items, num_pairs_per_item=3, random_seed=42)
        
        assert len(pairs_df) > 0
        assert 'item1' in pairs_df.columns
        assert 'item2' in pairs_df.columns
        
        # Check all items appear
        all_items = set(pairs_df['item1']).union(set(pairs_df['item2']))
        assert all_items == set(items)
    
    def test_pair_items_connectivity(self):
        """Test that pairing creates connected graph."""
        items = list(range(10))
        pairs_df = pair_items(items, num_pairs_per_item=5, random_seed=42)
        
        # Build adjacency list
        graph = {item: set() for item in items}
        for _, row in pairs_df.iterrows():
            graph[row['item1']].add(row['item2'])
            graph[row['item2']].add(row['item1'])
        
        # Check connectivity using BFS
        visited = {items[0]}
        queue = [items[0]]
        
        while queue:
            current = queue.pop(0)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        assert len(visited) == len(items), "Graph is not connected"


class TestTransitivity:
    """Test transitivity checking."""
    
    def test_perfect_transitivity(self):
        """Test with perfectly transitive comparisons."""
        df = pd.DataFrame({
            'item1': [1, 2, 1],
            'item2': [2, 3, 3],
            'decision': ['Text1', 'Text1', 'Text1']  # 1>2, 2>3, 1>3
        })
        
        result = check_transitivity(df)
        
        assert result['transitivity_rate'] == 1.0
        assert len(result['violations']) == 0
    
    def test_intransitive_triple(self):
        """Test with intransitive triple."""
        df = pd.DataFrame({
            'item1': [1, 2, 3],
            'item2': [2, 3, 1],
            'decision': ['Text1', 'Text1', 'Text1']  # 1>2, 2>3, 3>1 (cycle)
        })
        
        result = check_transitivity(df)
        
        assert result['transitivity_rate'] < 1.0
        assert len(result['violations']) > 0


class TestAnnotatorComparison:
    """Test annotator comparison functionality."""
    
    def test_perfect_agreement(self):
        """Test with perfect agreement."""
        df1 = pd.DataFrame({
            'item1': [1, 2, 3],
            'item2': [2, 3, 4],
            'decision': ['Text1', 'Text2', 'Text1']
        })
        
        df2 = df1.copy()
        
        result = compare_annotators(df1, df2, "Ann1", "Ann2")
        
        assert result['agreement_rate'] == 1.0
        assert result['cohens_kappa'] == 1.0
    
    def test_partial_agreement(self):
        """Test with partial agreement."""
        df1 = pd.DataFrame({
            'item1': [1, 2, 3, 4],
            'item2': [2, 3, 4, 5],
            'decision': ['Text1', 'Text2', 'Text1', 'Text2']
        })
        
        df2 = pd.DataFrame({
            'item1': [1, 2, 3, 4],
            'item2': [2, 3, 4, 5],
            'decision': ['Text1', 'Text2', 'Text2', 'Text1']  # 2 disagreements
        })
        
        result = compare_annotators(df1, df2, "Ann1", "Ann2")
        
        assert 0 < result['agreement_rate'] < 1.0
        assert result['disagreements'] == 2


class TestBradleyTerry:
    """Test Bradley-Terry scoring."""
    
    def test_basic_scoring(self):
        """Test basic Bradley-Terry scoring."""
        # Create simple comparison data
        items_df = pd.DataFrame({'id': [1, 2, 3]})
        
        pairwise_df = pd.DataFrame({
            'item1': [1, 2, 1],
            'item2': [2, 3, 3],
            'decision': ['Text1', 'Text1', 'Text1']  # 1 beats all
        })
        
        scored_df = bradley_terry_scores(
            original_df=items_df,
            row_id='id',
            pairwise_df=pairwise_df,
            normalize=True
        )
        
        assert 'Bradley_Terry_Score' in scored_df.columns
        assert len(scored_df) == 3
        
        # Item 1 should have highest score
        item1_score = scored_df[scored_df['id'] == 1]['Bradley_Terry_Score'].values[0]
        item3_score = scored_df[scored_df['id'] == 3]['Bradley_Terry_Score'].values[0]
        assert item1_score > item3_score
    
    def test_normalized_scores(self):
        """Test that normalized scores are in [0, 1]."""
        items_df = pd.DataFrame({'id': list(range(5))})
        
        pairwise_df = pd.DataFrame({
            'item1': [0, 1, 2, 3, 0],
            'item2': [1, 2, 3, 4, 4],
            'decision': ['Text1', 'Text1', 'Text2', 'Text2', 'Text1']
        })
        
        scored_df = bradley_terry_scores(
            original_df=items_df,
            row_id='id',
            pairwise_df=pairwise_df,
            normalize=True
        )
        
        scores = scored_df['Bradley_Terry_Score']
        assert scores.min() >= 0
        assert scores.max() <= 1


class TestDataValidation:
    """Test data validation utilities."""
    
    def test_missing_columns(self):
        """Test detection of missing columns."""
        from pairadigm.utils import validate_dataframe_columns
        
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        
        with pytest.raises(ValueError):
            validate_dataframe_columns(df, ['col1', 'col3'])
    
    def test_valid_columns(self):
        """Test with all required columns present."""
        from pairadigm.utils import validate_dataframe_columns
        
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        
        # Should not raise
        validate_dataframe_columns(df, ['col1', 'col2'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])