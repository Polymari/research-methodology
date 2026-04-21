"""
Semantic Theorem Search — Benchmark Pipeline

Recreates the evaluation from "Theorem Search" (arXiv:2602.05216).
Handles partial dataset (1.34M/9.2M theorems) by properly linking
ground truth and reporting both evaluable-only and full metrics.
"""

import argparse
import time
import sys
import os

import pandas as pd
import numpy as np

# Allow imports from src/ when running as `python src/main.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    load_corpus,
    load_theorems,
    load_slogans,
    load_test_queries_with_ground_truth,
)
from embedder import TheoremEmbedder
from retriever import VectorRetriever, HNSWRetriever
from evaluate import (
    calculate_precision_at_k,
    calculate_hit_at_k,
    calculate_mrr_at_k,
    evaluate_paper_level,
    print_results_table,
)


def run_benchmark(
    sample_size: int = 0,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    use_slogans: bool = True,
    use_hnsw: bool = False,
    batch_size: int = 32,
):
    """
    Run the full benchmark pipeline.

    Args:
        sample_size: Number of theorems in corpus (0 = all available ~1.34M).
        model_name: Embedding model to use.
        use_slogans: Embed slogans (True) or raw LaTeX bodies (False, for ablation).
        use_hnsw: Use HNSW index (True) or flat brute-force (False).
        batch_size: Batch size for embedding.
    """
    print("=" * 70)
    print("  Semantic Theorem Search — Benchmark Pipeline")
    print("=" * 70)
    print(f"  Model:       {model_name}")
    print(f"  Use slogans: {use_slogans}")
    print(f"  Index:       {'HNSW + rerank' if use_hnsw else 'Flat (exact)'}")
    print(f"  Sample size: {sample_size if sample_size > 0 else 'ALL (~1.34M)'}")
    print("=" * 70)

    # =========================================================================
    # 1. Load Data & Link Ground Truth
    # =========================================================================
    print("\n[1/5] Loading data...")

    df_theorems = load_theorems()

    # Link test queries to ground-truth theorem IDs
    test_records = load_test_queries_with_ground_truth(df_theorems)
    n_total = len(test_records)
    n_evaluable = sum(1 for r in test_records if r["is_evaluable"])
    n_exact = sum(1 for r in test_records if r["is_exact"])
    print(f"  Total queries: {n_total}")
    print(f"  Evaluable (paper in corpus): {n_evaluable}")
    print(f"  Exact theorem match: {n_exact}")

    # Load slogans if needed
    if use_slogans:
        df_slogans = load_slogans()
        df_corpus = df_theorems.merge(
            df_slogans, on="theorem_id", how="left", suffixes=("", "_slogan")
        )
        n_with_slogan = df_corpus["slogan"].notna().sum()
        print(f"  Slogans available: {n_with_slogan}/{len(df_corpus)}")
    else:
        df_corpus = df_theorems

    # =========================================================================
    # 2. Build Corpus (sample if requested, always include ground-truth)
    # =========================================================================
    print("\n[2/5] Building corpus...")

    # Collect ground-truth theorem IDs that must be in the corpus
    gt_theorem_ids = set()
    gt_paper_ids_set = set()
    for r in test_records:
        if r["gt_theorem_id"] is not None:
            gt_theorem_ids.add(r["gt_theorem_id"])
        if r["gt_paper_id"]:
            gt_paper_ids_set.add(r["gt_paper_id"])

    # For paper-level eval, include ALL theorems from ground-truth papers
    gt_paper_mask = df_corpus["paper_id"].astype(str).isin(gt_paper_ids_set)
    gt_rows = df_corpus[gt_paper_mask].drop_duplicates(subset=["theorem_id"])

    if sample_size > 0 and sample_size < len(df_corpus):
        remaining = max(0, sample_size - len(gt_rows))
        other = df_corpus[~df_corpus["theorem_id"].isin(gt_rows["theorem_id"])]
        random_sample = other.sample(n=min(remaining, len(other)), random_state=42)
        df_sample = pd.concat([gt_rows, random_sample]).reset_index(drop=True)
    else:
        df_sample = df_corpus.reset_index(drop=True)

    print(f"  Corpus size: {len(df_sample)} theorems")
    print(f"  Ground-truth theorems included: {len(gt_rows)}")

    # Build lookup: theorem_id -> index in df_sample
    tid_to_idx = {}
    for idx, row in df_sample.iterrows():
        tid_to_idx[row["theorem_id"]] = idx

    # =========================================================================
    # 3. Build Ground-Truth Index Lists
    # =========================================================================
    # For each query, identify which indices in df_sample are correct
    queries = []
    # Theorem-level: exact theorem_id match
    theorem_gt_indices_evaluable = []
    # Paper-level: any theorem from the correct paper
    paper_gt_ids_evaluable = []

    # Full lists (including non-evaluable queries as automatic misses)
    theorem_gt_indices_full = []
    paper_gt_ids_full = []

    evaluable_mask = []

    for r in test_records:
        queries.append(r["query"])

        if r["is_evaluable"]:
            evaluable_mask.append(True)

            # Theorem-level ground truth
            if r["gt_theorem_id"] is not None and r["gt_theorem_id"] in tid_to_idx:
                theorem_gt_indices_evaluable.append([tid_to_idx[r["gt_theorem_id"]]])
                theorem_gt_indices_full.append([tid_to_idx[r["gt_theorem_id"]]])
            else:
                # Paper exists but no exact theorem match — use all theorems from paper
                paper_thms = df_sample[
                    df_sample["paper_id"].astype(str) == r["gt_paper_id"]
                ]
                idxs = list(paper_thms.index)
                theorem_gt_indices_evaluable.append(idxs)
                theorem_gt_indices_full.append(idxs)

            paper_gt_ids_evaluable.append(r["arxiv_id"])
            paper_gt_ids_full.append(r["arxiv_id"])
        else:
            evaluable_mask.append(False)
            theorem_gt_indices_full.append([])  # automatic miss
            paper_gt_ids_full.append("")

    # =========================================================================
    # 4. Embed & Index
    # =========================================================================
    print("\n[3/5] Embedding corpus...")

    embedder = TheoremEmbedder(model_name=model_name)

    # Prepare passage text
    passages = []
    for _, row in df_sample.iterrows():
        text = ""
        if use_slogans and pd.notna(row.get("slogan", None)):
            text = str(row["slogan"])
        elif pd.notna(row.get("body", None)):
            text = str(row["body"])
        passages.append(text if text else "")

    t0 = time.time()
    passage_embeddings = embedder.embed_passages(passages, batch_size=batch_size)
    embed_time = time.time() - t0
    rate = len(passages) / embed_time if embed_time > 0 else 0
    print(f"  Embedded {len(passages)} passages in {embed_time:.1f}s ({rate:.0f}/s)")

    print("\n[4/5] Building index and searching...")

    embedding_dim = passage_embeddings.shape[1]
    if use_hnsw:
        retriever = HNSWRetriever(embedding_dim=embedding_dim)
    else:
        retriever = VectorRetriever(embedding_dim=embedding_dim)

    retriever.add_embeddings(passage_embeddings)

    t0 = time.time()
    query_embeddings = embedder.embed_queries(queries, batch_size=batch_size)
    t1 = time.time()
    distances, retrieved_indices = retriever.search(query_embeddings, top_k=20)
    t2 = time.time()

    print(f"  Query embed: {t1-t0:.2f}s | Search: {t2-t1:.2f}s")

    # =========================================================================
    # 5. Evaluate
    # =========================================================================
    print("\n[5/5] Computing metrics...")

    corpus_paper_ids = df_sample["paper_id"].astype(str).tolist()

    # --- Evaluable-only metrics (comparable to paper's ablation studies) ---
    eval_indices = [i for i, e in enumerate(evaluable_mask) if e]
    retrieved_eval = [retrieved_indices[i] for i in eval_indices]

    # Theorem-level
    t_p1 = calculate_precision_at_k(retrieved_eval, theorem_gt_indices_evaluable, k=1)
    t_h10 = calculate_hit_at_k(retrieved_eval, theorem_gt_indices_evaluable, k=10)
    t_h20 = calculate_hit_at_k(retrieved_eval, theorem_gt_indices_evaluable, k=20)
    t_mrr = calculate_mrr_at_k(retrieved_eval, theorem_gt_indices_evaluable, k=20)

    # Paper-level
    p_metrics_eval = evaluate_paper_level(
        retrieved_eval, paper_gt_ids_evaluable, corpus_paper_ids, k_values=[1, 10, 20]
    )

    theorem_metrics_eval = {"P@1": t_p1, "Hit@10": t_h10, "Hit@20": t_h20, "MRR@20": t_mrr}

    # --- Full metrics (all 110 queries, missing ground truth = miss) ---
    t_p1_f = calculate_precision_at_k(retrieved_indices, theorem_gt_indices_full, k=1)
    t_h10_f = calculate_hit_at_k(retrieved_indices, theorem_gt_indices_full, k=10)
    t_h20_f = calculate_hit_at_k(retrieved_indices, theorem_gt_indices_full, k=20)
    t_mrr_f = calculate_mrr_at_k(retrieved_indices, theorem_gt_indices_full, k=20)

    p_metrics_full = evaluate_paper_level(
        retrieved_indices, paper_gt_ids_full, corpus_paper_ids, k_values=[1, 10, 20]
    )

    theorem_metrics_full = {"P@1": t_p1_f, "Hit@10": t_h10_f, "Hit@20": t_h20_f, "MRR@20": t_mrr_f}

    # --- Print results ---
    mode = "slogans" if use_slogans else "LaTeX bodies"
    model_short = model_name.split("/")[-1]

    print(f"\n{'='*72}")
    print(f"  EVALUABLE-ONLY ({n_evaluable} queries with ground truth in corpus)")
    print(f"{'='*72}")
    print_results_table(
        theorem_metrics_eval, p_metrics_eval,
        model_name=f"{model_short} ({mode})"
    )

    print(f"\n{'='*72}")
    print(f"  FULL EVALUATION ({n_total} queries, missing ground truth = miss)")
    print(f"{'='*72}")
    print_results_table(
        theorem_metrics_full, p_metrics_full,
        model_name=f"{model_short} ({mode})"
    )

    return {
        "evaluable": {"theorem": theorem_metrics_eval, "paper": p_metrics_eval},
        "full": {"theorem": theorem_metrics_full, "paper": p_metrics_full},
        "n_evaluable": n_evaluable,
        "n_total": n_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Semantic Theorem Search Benchmark")
    parser.add_argument(
        "--sample-size", type=int, default=0,
        help="Corpus size (0=all ~1.34M, >0 = sample N theorems)"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model name"
    )
    parser.add_argument(
        "--no-slogans", action="store_true",
        help="Embed raw LaTeX bodies instead of slogans (ablation)"
    )
    parser.add_argument(
        "--hnsw", action="store_true",
        help="Use HNSW approximate index (faster for large corpora)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Embedding batch size"
    )

    args = parser.parse_args()

    run_benchmark(
        sample_size=args.sample_size,
        model_name=args.model,
        use_slogans=not args.no_slogans,
        use_hnsw=args.hnsw,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
