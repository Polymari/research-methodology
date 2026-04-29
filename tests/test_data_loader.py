import pandas as pd
from unittest.mock import patch, MagicMock


@patch("src.data_loader.load_dataset")
def test_load_theorems(mock_load_dataset):
    """Test that load_theorems loads the correct HuggingFace config."""
    from src.data_loader import load_theorems

    mock_df = pd.DataFrame({"theorem_id": [1, 2], "body": ["thm 1", "thm 2"]})
    mock_ds = MagicMock()
    mock_ds["train"].to_pandas.return_value = mock_df
    mock_load_dataset.return_value = mock_ds

    df = load_theorems()

    mock_load_dataset.assert_called_once_with(
        "uw-math-ai/theorem-search-dataset", "theorem"
    )
    assert len(df) == 2
    assert "theorem_id" in df.columns


@patch("src.data_loader.load_dataset")
def test_load_slogans(mock_load_dataset):
    """Test that load_slogans loads the correct HuggingFace config."""
    from src.data_loader import load_slogans

    mock_df = pd.DataFrame({"theorem_id": [1, 2], "slogan": ["desc 1", "desc 2"]})
    mock_ds = MagicMock()
    mock_ds["train"].to_pandas.return_value = mock_df
    mock_load_dataset.return_value = mock_ds

    df = load_slogans()

    mock_load_dataset.assert_called_once_with(
        "uw-math-ai/theorem-search-dataset", "theorem_slogan"
    )
    assert len(df) == 2
    assert "slogan" in df.columns


def test_parse_arxiv_id():
    """Test arXiv ID extraction from URLs."""
    from src.data_loader import parse_arxiv_id

    assert parse_arxiv_id("https://arxiv.org/abs/2310.15076") == "2310.15076"
    assert parse_arxiv_id("https://arxiv.org/abs/1504.06467") == "1504.06467"
    assert parse_arxiv_id("") == ""
    assert parse_arxiv_id(None) == ""
    assert parse_arxiv_id("https://example.com") == ""


def test_load_test_queries_with_ground_truth():
    """Test ground-truth linking with mock data."""
    from src.data_loader import load_test_queries_with_ground_truth

    # Mock theorem corpus
    df_theorems = pd.DataFrame(
        {
            "theorem_id": [100, 101, 102, 200],
            "paper_id": ["2310.15076", "2310.15076", "2310.15076", "9999.99999"],
            "name": ["Theorem 3.1", "Lemma 3.8", "Theorem 1.1", "Theorem 1.1"],
            "body": ["body1", "body2", "body3", "body4"],
        }
    )

    # Patch load_test_queries to return test data
    mock_test = pd.DataFrame(
        {
            "query": ["find theorem A", "find lemma B", "missing paper query"],
            "theorem number": ["Theorem 3.1", "Lemma 3.8", "Theorem 5.5"],
            "paper title": ["Paper A", "Paper A", "Paper C"],
            "link to paper on arxiv": [
                "https://arxiv.org/abs/2310.15076",
                "https://arxiv.org/abs/2310.15076",
                "https://arxiv.org/abs/0000.00000",
            ],
        }
    )

    with patch("src.data_loader.load_test_queries", return_value=mock_test):
        results = load_test_queries_with_ground_truth(df_theorems)

    assert len(results) == 3

    # Q0: exact match
    assert results[0]["is_evaluable"] is True
    assert results[0]["is_exact"] is True
    assert results[0]["gt_theorem_id"] == 100

    # Q1: exact match
    assert results[1]["is_evaluable"] is True
    assert results[1]["is_exact"] is True
    assert results[1]["gt_theorem_id"] == 101

    # Q2: paper not in corpus
    assert results[2]["is_evaluable"] is False
    assert results[2]["gt_theorem_id"] is None
