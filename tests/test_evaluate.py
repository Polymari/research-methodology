import pytest
from src.evaluate import calculate_recall_at_k

def test_calculate_recall_at_k():
    retrieved = [
        [1, 2, 3, 4, 5], 
        [10, 11, 12, 13, 14],
        [20, 21, 22, 23, 24]
    ]
    gt = [
        [3, 5],      # 2 found in top 5, gt size = 2 -> 2/2 = 1.0
        [15, 12, 10],# 2 found in top 5 (10, 12), gt size = 3 -> 2/3 = 0.666...
        []           # empty ground truth -> handled as 0.0 or ignored. Our impl uses 0.0
    ]
    
    # recall = (1.0 + 0.666 + 0.0) / 3 = 1.666 / 3 = 0.555...
    recall = calculate_recall_at_k(retrieved, gt, k=5)
    
    assert abs(recall - (1.0 + 2/3 + 0.0) / 3) < 1e-5

def test_calculate_recall_at_k_k_cutoff():
    retrieved = [[1, 2, 3, 4, 5]]
    gt = [[5]]
    
    # K=5 -> document 5 is at rank 5, recall = 1.0
    r5 = calculate_recall_at_k(retrieved, gt, k=5)
    assert r5 == 1.0
    
    # K=4 -> document 5 is excluded, recall = 0.0
    r4 = calculate_recall_at_k(retrieved, gt, k=4)
    assert r4 == 0.0
