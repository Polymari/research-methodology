import numpy as np

def calculate_recall_at_k(retrieved_indices, ground_truth_indices, k=10):
    """
    Calculates Recall@K.
    
    Args:
        retrieved_indices: List or 2D array of shape (num_queries, top_n) containing the indices of retrieved documents.
        ground_truth_indices: List of lists containing the indices of the ground truth documents for each query.
        k: The threshold for calculating recall (e.g., top 10).
    """
    recalls = []
    for retrieved, gt in zip(retrieved_indices, ground_truth_indices):
        retrieved_k = retrieved[:k]
        
        gt_set = set(gt)
        if len(gt_set) > 0:
            intersection = set(retrieved_k).intersection(gt_set)
            recalls.append(len(intersection) / len(gt_set))
        else:
            recalls.append(0.0) # If there are no ground truth docs, recall is 0 (or could be ignored)
            
    return np.mean(recalls)

if __name__ == "__main__":
    # Dummy mock evaluation test
    retrieved = [[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]]
    gt = [[3], [15, 12]]
    
    # query 1: retrieved doc 3 in top 5. (1/1) = 1.0!
    # query 2: retrieved doc 12 in top 5. (1/2) = 0.5!
    # average recall = 0.75
    
    mean_recall = calculate_recall_at_k(retrieved, gt, k=5)
    print(f"Mean Recall@5: {mean_recall:.4f}")
