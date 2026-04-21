import pandas as pd
from datasets import load_dataset


def load_theorems():
    """Loads the main theorem dataset from HuggingFace."""
    print("Loading theorem dataset...")
    ds = load_dataset("uw-math-ai/theorem-search-dataset", "theorem")
    return ds["train"].to_pandas()


def load_slogans():
    """Loads the natural-language slogans (descriptions) for the theorems."""
    print("Loading theorem slogans dataset...")
    ds = load_dataset("uw-math-ai/theorem-search-dataset", "theorem_slogan")
    return ds["train"].to_pandas()


def load_papers():
    """Loads the metadata for the papers."""
    print("Loading papers dataset...")
    ds = load_dataset("uw-math-ai/theorem-search-dataset", "paper")
    return ds["train"].to_pandas()


def load_test_queries():
    """Loads the 110 evaluation queries written by professional mathematicians."""
    print("Loading test queries...")
    ds = load_dataset("uw-math-ai/theorem-search-dataset", "theorem-test")
    split = "test" if "test" in ds else "train"
    df = ds[split].to_pandas()
    return df


def parse_arxiv_id(url: str) -> str:
    """Extract the arXiv ID from a URL like 'https://arxiv.org/abs/2310.15076'."""
    if not url or "arxiv.org" not in str(url):
        return ""
    parts = str(url).split("arxiv.org/")
    if len(parts) > 1:
        return parts[1].split("/")[-1].strip()
    return ""


def load_test_queries_with_ground_truth(df_theorems=None):
    """
    Loads test queries and links each to its ground-truth theorem_id in the corpus.
    
    Uses the 'theorem number' field from the test set and matches against 
    the 'name' field of theorems from the same paper.
    
    Returns:
        list of dicts, each with:
            - query: str
            - arxiv_id: str
            - paper_title: str
            - theorem_number: str
            - gt_theorem_id: int or None (exact theorem match)
            - gt_paper_id: str (paper_id in corpus, or "")
            - is_evaluable: bool (True if ground truth exists in corpus)
            - is_exact: bool (True if exact theorem-level match found)
    """
    df_test = load_test_queries()
    
    if df_theorems is None:
        df_theorems = load_theorems()
    
    results = []
    for _, row in df_test.iterrows():
        arxiv_id = parse_arxiv_id(row.get("link to paper on arxiv", ""))
        thm_number = str(row.get("theorem number", "")).strip()
        
        record = {
            "query": row["query"],
            "arxiv_id": arxiv_id,
            "paper_title": row.get("paper title", ""),
            "theorem_number": thm_number,
            "gt_theorem_id": None,
            "gt_paper_id": "",
            "is_evaluable": False,
            "is_exact": False,
        }
        
        if not arxiv_id:
            results.append(record)
            continue
        
        # Find theorems from this paper in the corpus
        paper_mask = df_theorems["paper_id"].astype(str).str.contains(
            arxiv_id, na=False, regex=False
        )
        paper_thms = df_theorems[paper_mask]
        
        if paper_thms.empty:
            results.append(record)
            continue
        
        # Paper exists in corpus
        record["gt_paper_id"] = str(paper_thms["paper_id"].iloc[0])
        record["is_evaluable"] = True
        
        # Try exact theorem number match
        if thm_number:
            # Match "Theorem 3.1" against name field like "Theorem 3.1" or "Theorem 3.1 (Some note)"
            exact = paper_thms[
                paper_thms["name"].str.contains(thm_number, na=False, regex=False)
            ]
            
            if len(exact) == 1:
                record["gt_theorem_id"] = int(exact["theorem_id"].iloc[0])
                record["is_exact"] = True
            elif len(exact) > 1:
                # Multiple matches (e.g., "Theorem 1.1" also matches "Theorem 1.12")
                # Try stricter match: name starts with or equals the theorem number
                strict = exact[
                    exact["name"].str.strip().str.startswith(thm_number)
                ]
                # Among strict matches, prefer exact name match
                for _, t in strict.iterrows():
                    name = str(t["name"]).strip()
                    # Exact: "Theorem 3.1" or "Theorem 3.1 (note)" but not "Theorem 3.12"
                    name_base = name.split("(")[0].strip().rstrip(".")
                    target_base = thm_number.rstrip(".")
                    if name_base == target_base:
                        record["gt_theorem_id"] = int(t["theorem_id"])
                        record["is_exact"] = True
                        break
                
                # If still ambiguous, take the first strict match
                if record["gt_theorem_id"] is None and len(strict) > 0:
                    record["gt_theorem_id"] = int(strict["theorem_id"].iloc[0])
                    record["is_exact"] = True
        
        results.append(record)
    
    n_evaluable = sum(1 for r in results if r["is_evaluable"])
    n_exact = sum(1 for r in results if r["is_exact"])
    print(f"Ground-truth linking: {n_evaluable}/{len(results)} evaluable, {n_exact} exact theorem matches")
    
    return results


def load_corpus(with_slogans=True):
    """
    Loads the theorem corpus, optionally joined with slogans.
    """
    df_theorems = load_theorems()
    
    if with_slogans:
        df_slogans = load_slogans()
        df = df_theorems.merge(df_slogans, on="theorem_id", how="left", suffixes=("", "_slogan"))
        print(f"Corpus: {len(df)} theorems, {df['slogan'].notna().sum()} with slogans")
    else:
        df = df_theorems
        print(f"Corpus: {len(df)} theorems (no slogans)")
    
    return df


def load_full_corpus():
    """
    Loads the full corpus with slogans AND paper metadata joined.
    """
    df = load_corpus(with_slogans=True)
    df_papers = load_papers()
    df = df.merge(df_papers, on="paper_id", how="left", suffixes=("", "_paper"))
    print(f"Full corpus: {len(df)} theorems with paper metadata")
    return df


if __name__ == "__main__":
    # Quick validation of ground-truth linking
    df_thm = load_theorems()
    records = load_test_queries_with_ground_truth(df_thm)
    
    print("\n--- Evaluable Queries ---")
    for r in records:
        if r["is_evaluable"]:
            status = "EXACT" if r["is_exact"] else "PAPER"
            print(f"[{status}] {r['query'][:70]}  ->  thm_id={r['gt_theorem_id']}")
