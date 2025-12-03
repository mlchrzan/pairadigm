"""
Simple tests to verify package installation and basic functionality.
Run with: pytest tests/test_basic.py
"""

import pytest
import pandas as pd


def test_import():
    """Test that the package can be imported."""
    from pairadigm import Pairadigm, LLMClient, load_pairadigm, pair_items
    assert Pairadigm is not None
    assert LLMClient is not None
    assert load_pairadigm is not None
    assert pair_items is not None


def test_version():
    """Test that version is accessible."""
    import pairadigm
    assert hasattr(pairadigm, '__version__')
    assert pairadigm.__version__ == "0.2.1"


def test_pair_items():
    """Test the pair_items utility function."""
    from pairadigm import pair_items
    
    items = ['item1', 'item2', 'item3', 'item4', 'item5']
    result = pair_items(items, num_pairs_per_item=3, random_seed=42)
    
    assert isinstance(result, pd.DataFrame)
    assert 'item1' in result.columns
    assert 'item2' in result.columns
    assert len(result) > 0


def test_pairadigm_initialization():
    """Test basic Pairadigm initialization without API calls."""
    from pairadigm import Pairadigm
    import os
    
    # Set a dummy API key for testing initialization
    os.environ['GENAI_API_KEY'] = 'test_key_12345'
    
    # Create simple test data
    test_data = pd.DataFrame({
        'id': ['1', '2', '3'],
        'text': ['Sample text 1', 'Sample text 2', 'Sample text 3']
    })
    
    # Initialize without cgcot_prompts (should give warning but not error)
    try:
        p = Pairadigm(
            data=test_data,
            item_id_name='id',
            text_name='text',
            target_concept='test_concept',
            cgcot_prompts=None
        )
        # Should work but with a warning
        assert p is not None
    except ValueError as e:
        # Expected if target_concept validation fails
        assert "target_concept" in str(e) or "cgcot_prompts" in str(e)
    
    # Clean up
    if 'GENAI_API_KEY' in os.environ:
        del os.environ['GENAI_API_KEY']


def test_llm_client_provider_inference():
    """Test that LLMClient can infer providers from model names."""
    from pairadigm import LLMClient
    
    # Test provider inference (without actually initializing API clients)
    client = LLMClient.__new__(LLMClient)
    
    assert client._infer_provider('gemini-2.0-flash-exp') == 'google'
    assert client._infer_provider('gpt-4o') == 'openai'
    assert client._infer_provider('claude-sonnet-4') == 'anthropic'
    
    with pytest.raises(ValueError):
        client._infer_provider('unknown-model')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
