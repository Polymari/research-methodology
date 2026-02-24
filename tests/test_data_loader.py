import pytest
import pandas as pd
from unittest.mock import patch
from src.data_loader import load_theorems

# We mock pd.read_parquet so we don't actually download gigabytes during testing
@patch('src.data_loader.pd.read_parquet')
def test_load_theorems(mock_read_parquet):
    # Setup mock return value
    mock_df = pd.DataFrame({'id': [1, 2], 'text': ['theorem 1', 'theorem 2']})
    mock_read_parquet.return_value = mock_df
    
    # Call function
    df = load_theorems()
    
    # Assertions
    mock_read_parquet.assert_called_once_with("hf://datasets/uw-math-ai/theorem-search-dataset/theorem.parquet")
    assert len(df) == 2
    assert 'text' in df.columns
