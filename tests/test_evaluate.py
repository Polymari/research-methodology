import pytest
from src.evaluate import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_hit_at_k,
    calculate_mrr_at_k,
)


@pytest.fixture
def sample_data():
    retrieved = [[1, 2, 3, 4, 5], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24]]
    gt = [
        [3, 5],  # Query 1
        [15, 12, 10],  # Query 2
        [],  # Query 3
    ]
    return retrieved, gt


def test_calculate_recall_at_k(sample_data):
    retrieved, gt = sample_data
    recall = calculate_recall_at_k(retrieved, gt, k=5)
    assert abs(recall - (1.0 + 2 / 3 + 0.0) / 3) < 1e-5


def test_calculate_precision_at_k(sample_data):
    retrieved, gt = sample_data
    # Q1: 3, 5 are in gt -> 2/5
    # Q2: 10, 12 are in gt -> 2/5
    # Q3: gt empty -> 0
    p = calculate_precision_at_k(retrieved, gt, k=5)
    assert abs(p - (0.4 + 0.4 + 0.0) / 3) < 1e-5

    # Q1: 3 is at pos 3 -> P@3 = 1/3
    # Q2: 10 is at pos 1, 12 is at pos 3 -> P@3 = 2/3
    # Q3: -> 0
    p3 = calculate_precision_at_k(retrieved, gt, k=3)
    assert abs(p3 - (1 / 3 + 2 / 3 + 0.0) / 3) < 1e-5


def test_calculate_hit_at_k(sample_data):
    retrieved, gt = sample_data
    # Q1: hit (doc 3)
    # Q2: hit (doc 10)
    # Q3: no hit
    h = calculate_hit_at_k(retrieved, gt, k=5)
    assert abs(h - (1.0 + 1.0 + 0.0) / 3) < 1e-5

    # K=1
    # Q1: retrieved[0]=1, not in gt -> 0
    # Q2: retrieved[0]=10, in gt -> 1
    # Q3: 0
    h1 = calculate_hit_at_k(retrieved, gt, k=1)
    assert abs(h1 - (0.0 + 1.0 + 0.0) / 3) < 1e-5


def test_calculate_mrr_at_k(sample_data):
    retrieved, gt = sample_data
    # Q1: doc 3 is at index 2 (rank 3) -> 1/3
    # Q2: doc 10 is at index 0 (rank 1) -> 1/1
    # Q3: 0
    mrr = calculate_mrr_at_k(retrieved, gt, k=5)
    assert abs(mrr - (1 / 3 + 1.0 + 0.0) / 3) < 1e-5

    # Q1: doc 3 is rank 3. If k=2, no hit -> 0
    mrr2 = calculate_mrr_at_k(retrieved, gt, k=2)
    assert abs(mrr2 - (0.0 + 1.0 + 0.0) / 3) < 1e-5
