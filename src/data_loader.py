import pandas as pd

def load_theorems():
    """Loads the main theorem dataset."""
    print("Loading theorem dataset...")
    return pd.read_parquet("hf://datasets/uw-math-ai/theorem-search-dataset/theorem.parquet")

def load_slogans():
    """Loads the natural-language slogans (descriptions) for the theorems."""
    print("Loading theorem slogans dataset...")
    return pd.read_parquet("hf://datasets/uw-math-ai/theorem-search-dataset/theorem_slogan.parquet")

def load_papers():
    """Loads the metadata for the papers."""
    print("Loading papers dataset...")
    return pd.read_parquet("hf://datasets/uw-math-ai/theorem-search-dataset/paper.parquet")

if __name__ == "__main__":
    df_theorems = load_theorems()
    print(f"Loaded {len(df_theorems)} theorems.")
    print(df_theorems.head(2))
