import numpy as np


def calculate_precision_at_k(retrieved_indices, ground_truth_indices, k=10):
    """
    Precision@K: fraction of top-k results that are relevant.
    """
    precisions = []
    for retrieved, gt in zip(retrieved_indices, ground_truth_indices):
        retrieved_k = retrieved[:k]
        gt_set = set(gt)
        if not gt_set:
            precisions.append(0.0)
            continue
        hits = sum(1 for doc in retrieved_k if doc in gt_set)
        precisions.append(hits / len(retrieved_k) if k > 0 else 0.0)
    return np.mean(precisions)


def calculate_hit_at_k(retrieved_indices, ground_truth_indices, k=10):
    """
    Hit@K: 1 if any of top-k results is relevant, else 0.
    """
    hits = []
    for retrieved, gt in zip(retrieved_indices, ground_truth_indices):
        retrieved_k = retrieved[:k]
        gt_set = set(gt)
        if not gt_set:
            hits.append(0.0)
            continue
        hit = 1 if any(doc in gt_set for doc in retrieved_k) else 0
        hits.append(float(hit))
    return np.mean(hits)


def calculate_mrr_at_k(retrieved_indices, ground_truth_indices, k=20):
    """
    MRR@K: mean reciprocal rank of the first relevant result in top-k.
    """
    mrrs = []
    for retrieved, gt in zip(retrieved_indices, ground_truth_indices):
        retrieved_k = retrieved[:k]
        gt_set = set(gt)
        if not gt_set:
            mrrs.append(0.0)
            continue
        mrr = 0.0
        for rank, doc in enumerate(retrieved_k):
            if doc in gt_set:
                mrr = 1.0 / (rank + 1)
                break
        mrrs.append(mrr)
    return np.mean(mrrs)


def calculate_recall_at_k(retrieved_indices, ground_truth_indices, k=10):
    """
    Recall@K: fraction of relevant documents found in top-k.
    """
    recalls = []
    for retrieved, gt in zip(retrieved_indices, ground_truth_indices):
        retrieved_k = retrieved[:k]
        gt_set = set(gt)
        if len(gt_set) > 0:
            intersection = set(retrieved_k).intersection(gt_set)
            recalls.append(len(intersection) / len(gt_set))
        else:
            recalls.append(0.0)
    return np.mean(recalls)


def evaluate_paper_level(retrieved_indices, ground_truth_paper_ids, corpus_paper_ids, k_values=[1, 10, 20]):
    """
    Paper-level evaluation: map retrieved theorem indices to paper IDs,
    then compute metrics based on whether the correct paper was found.
    
    Args:
        retrieved_indices: List of arrays, each containing the top-k retrieved corpus indices per query.
        ground_truth_paper_ids: List of target paper_id strings, one per query.
        corpus_paper_ids: Array/list of paper_id for each theorem in the corpus (indexed same as corpus).
        k_values: List of k values to compute metrics for.
    
    Returns:
        dict with paper-level P@k, Hit@k, MRR@k for each k.
    """
    results = {}
    
    for k in k_values:
        hits = []
        precisions = []
        mrrs = []
        
        for retrieved, gt_paper_id in zip(retrieved_indices, ground_truth_paper_ids):
            if not gt_paper_id:
                hits.append(0.0)
                precisions.append(0.0)
                mrrs.append(0.0)
                continue
            
            retrieved_k = retrieved[:k]
            # Map retrieved theorem indices to paper IDs
            retrieved_papers = []
            for idx in retrieved_k:
                if idx >= 0 and idx < len(corpus_paper_ids):
                    retrieved_papers.append(str(corpus_paper_ids[idx]))
                else:
                    retrieved_papers.append("")
            
            # Check if any retrieved paper matches the ground truth
            paper_hits = [1 if p and gt_paper_id in p else 0 for p in retrieved_papers]
            
            hits.append(1.0 if any(paper_hits) else 0.0)
            precisions.append(sum(paper_hits) / k if k > 0 else 0.0)
            
            mrr = 0.0
            for rank, h in enumerate(paper_hits):
                if h:
                    mrr = 1.0 / (rank + 1)
                    break
            mrrs.append(mrr)
        
        results[f"P@{k}"] = np.mean(precisions)
        results[f"Hit@{k}"] = np.mean(hits)
        results[f"MRR@{k}"] = np.mean(mrrs)
    
    return results


def print_results_table(theorem_metrics, paper_metrics=None, model_name="Model"):
    """
    Print results in a format matching the paper's Table 2.
    """
    print(f"\n{'='*60}")
    print(f"  Results for: {model_name}")
    print(f"{'='*60}")
    
    header = f"{'Metric':<12}"
    if paper_metrics:
        header += f"{'Theorem-Level':>16} {'Paper-Level':>16}"
    else:
        header += f"{'Value':>16}"
    print(header)
    print("-" * 60)
    
    for metric_name, value in theorem_metrics.items():
        row = f"{metric_name:<12}"
        row += f"{value:>16.3f}"
        if paper_metrics and metric_name in paper_metrics:
            row += f"{paper_metrics[metric_name]:>16.3f}"
        print(row)
    
    print(f"{'='*60}")
