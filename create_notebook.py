import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# 1. Setup and imports
cell_imports = """\
import pandas as pd
import numpy as np
import time
import torch
from tqdm.auto import tqdm

# Import our custom modules
import sys
sys.path.append('src')

from data_loader import load_theorems, load_test_queries_with_ground_truth
from embedder import TheoremEmbedder
from retriever import HNSWRetriever, VectorRetriever
from evaluate import calculate_precision_at_k, calculate_hit_at_k, calculate_mrr_at_k, evaluate_paper_level, print_results_table

# Check Hardware
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""

# 2. Config cell
cell_config = """\
# ==========================================
# CONFIGURATION
# ==========================================

# How many theorems to include in the corpus (0 = all ~1.34M)
# A sample of ~50,000 takes around 15 minutes on an RTX 4060
SAMPLE_SIZE = 50000 

# Embedding Model
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B" # Base model
# EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2" # Fast, CPU-friendly
# EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B" # Needs ~24GB+ VRAM depending on batch size

# Batch Size for Embedding (Lower this if you run into CUDA Out of Memory errors)
# e.g., RTX 4060 (8GB VRAM) needs batch size <= 4 for the Qwen3 0.6B model with max sequence length.
BATCH_SIZE = 4

# Retrieval method
USE_HNSW = False # Set to True for large corpus (approximate metric search)

# Slogan Generation Configuration
# The HF dataset comes with DeepSeek-v3.1 slogans. We can use those, 
# or generate our own to experiment with different prompt strategies/models.
USE_GENERATED_SLOGANS = False # True = Use our custom generated slogans, False = Use original or raw body
"""

# 3. Slogan Generation Module
cell_slogan_gen = """\
# ==========================================
# SLOGAN GENERATION FRAMEWORK
# ==========================================
# Plug in any LLM API (OpenAI, Anthropic, Gemini, local vLLM/RunPod)
# to generate new slogans and immediately test their retrieval impact.

class BaseSloganGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        
    def generate_slogan(self, theorem_body, context=""):
        raise NotImplementedError("Subclasses must implement this.")
        
    def batch_generate(self, theorems_df):
        slogans = []
        for idx, row in tqdm(theorems_df.iterrows(), total=len(theorems_df), desc=f"Generating slogans with {self.model_name}"):
            # Pass abstract/context here if you load the paper dataset
            slogan = self.generate_slogan(row['body'], context="")
            slogans.append(slogan)
        return slogans

# Example: Stub for a new Generator (replace with API/Runpod API requests)
class DummySloganGenerator(BaseSloganGenerator):
    def generate_slogan(self, theorem_body, context=""):
        # e.g., runpod_client.run(...) or openai.chat.completions.create(...)
        return f"This generated text replaces the body during retrieval."

def generate_and_attach_slogans(df_corpus, generator):
    new_slogans = generator.batch_generate(df_corpus)
    df_corpus['custom_slogan'] = new_slogans
    return df_corpus
"""

# 4. Data Loading
cell_data = """\
# ==========================================
# DATA LOADING & EVALUATION SETUP
# ==========================================

print("Loading original theorems...")
df_theorems = load_theorems()
test_records = load_test_queries_with_ground_truth(df_theorems)

n_evaluable = sum(1 for r in test_records if r["is_evaluable"])
print(f"Loaded {len(test_records)} queries ({n_evaluable} evaluable)")

# Collect ground-truth theorems that MUST be in the corpus
gt_paper_ids = {r["gt_paper_id"] for r in test_records if r["gt_paper_id"]}
gt_mask = df_theorems["paper_id"].astype(str).isin(gt_paper_ids)
gt_rows = df_theorems[gt_mask].drop_duplicates(subset=["theorem_id"])

if SAMPLE_SIZE > 0 and SAMPLE_SIZE < len(df_theorems):
    remaining = max(0, SAMPLE_SIZE - len(gt_rows))
    other = df_theorems[~df_theorems["theorem_id"].isin(gt_rows["theorem_id"])]
    random_sample = other.sample(n=min(remaining, len(other)), random_state=42)
    df_sample = pd.concat([gt_rows, random_sample]).reset_index(drop=True)
else:
    df_sample = df_theorems.reset_index(drop=True)

print(f"Final corpus size: {len(df_sample)}")

if USE_GENERATED_SLOGANS:
    print("\\nGenerating custom slogans...")
    my_generator = DummySloganGenerator(model_name="RunPod-LLM")
    df_sample = generate_and_attach_slogans(df_sample, my_generator)
else:
    from data_loader import load_slogans
    print("\\nLoading default slogans from dataset...")
    df_slogans = load_slogans()
    df_sample = df_sample.merge(df_slogans, on="theorem_id", how="left")
"""

# 5. Embedding
cell_embed = """\
# ==========================================
# EMBEDDING
# ==========================================

passages = []
for _, row in df_sample.iterrows():
    if USE_GENERATED_SLOGANS and 'custom_slogan' in row:
        text = str(row['custom_slogan'])
    elif 'slogan' in row and pd.notna(row['slogan']):
        text = str(row['slogan'])
    else:
        text = str(row['body'])
    passages.append(text)

print(f"Initializing embedder: {EMBEDDING_MODEL_NAME}")
embedder = TheoremEmbedder(model_name=EMBEDDING_MODEL_NAME)

print(f"Embedding {len(passages)} passages with batch_size={BATCH_SIZE}...")
t0 = time.time()
passage_embeddings = embedder.embed_passages(passages, batch_size=BATCH_SIZE)
print(f"Done in {time.time()-t0:.1f}s")
"""

# 6. Indexing & Evaluation
cell_eval = """\
# ==========================================
# RETRIEVAL & EVALUATION
# ==========================================

embedding_dim = passage_embeddings.shape[1]
if USE_HNSW:
    retriever = HNSWRetriever(embedding_dim=embedding_dim)
else:
    retriever = VectorRetriever(embedding_dim=embedding_dim)

print("Building index...")
retriever.add_embeddings(passage_embeddings)

queries = [r["query"] for r in test_records]
print(f"Embedding {len(queries)} queries...")
query_embeddings = embedder.embed_queries(queries, batch_size=BATCH_SIZE)

print("Searching...")
distances, retrieved_indices = retriever.search(query_embeddings, top_k=20)

# Build Ground Truth mappings
tid_to_idx = {row["theorem_id"]: idx for idx, row in df_sample.iterrows()}
theorem_gt_indices = []
paper_gt_ids = []
evaluable_mask = []

for r in test_records:
    if r["is_evaluable"]:
        evaluable_mask.append(True)
        if r["gt_theorem_id"] is not None and r["gt_theorem_id"] in tid_to_idx:
            theorem_gt_indices.append([tid_to_idx[r["gt_theorem_id"]]])
        else:
            paper_thms = df_sample[df_sample["paper_id"].astype(str) == r["gt_paper_id"]]
            theorem_gt_indices.append(list(paper_thms.index))
        paper_gt_ids.append(r["arxiv_id"])
    else:
        evaluable_mask.append(False)
        theorem_gt_indices.append([])
        paper_gt_ids.append("")

# CALCULATE EVALUABLE-ONLY METRICS
eval_indices = [i for i, e in enumerate(evaluable_mask) if e]
ret_eval = [retrieved_indices[i] for i in eval_indices]
gt_eval = [theorem_gt_indices[i] for i in eval_indices]

t_metrics = {
    "P@1": calculate_precision_at_k(ret_eval, gt_eval, k=1),
    "Hit@10": calculate_hit_at_k(ret_eval, gt_eval, k=10),
    "Hit@20": calculate_hit_at_k(ret_eval, gt_eval, k=20),
    "MRR@20": calculate_mrr_at_k(ret_eval, gt_eval, k=20)
}

paper_metrics = evaluate_paper_level(
    ret_eval, [paper_gt_ids[i] for i in eval_indices], 
    df_sample["paper_id"].astype(str).tolist(), k_values=[1, 10, 20]
)

print("\\n" + "="*50)
print("EVALUABLE QUERIES (15) RESULTS")
print("="*50)
print_results_table(t_metrics, paper_metrics, model_name=EMBEDDING_MODEL_NAME)

# CALCULATE FULL DATASET METRICS
t_metrics_full = {
    "P@1": calculate_precision_at_k(retrieved_indices, theorem_gt_indices, k=1),
    "Hit@10": calculate_hit_at_k(retrieved_indices, theorem_gt_indices, k=10),
    "Hit@20": calculate_hit_at_k(retrieved_indices, theorem_gt_indices, k=20),
    "MRR@20": calculate_mrr_at_k(retrieved_indices, theorem_gt_indices, k=20)
}

paper_metrics_full = evaluate_paper_level(
    retrieved_indices, paper_gt_ids, df_sample["paper_id"].astype(str).tolist(), k_values=[1, 10, 20]
)

print("\\n" + "="*50)
print("FULL DATASET (110) QUERIES RESULTS")
print("="*50)
print_results_table(t_metrics_full, paper_metrics_full, model_name=EMBEDDING_MODEL_NAME)
"""

nb.cells = [
    nbf.v4.new_markdown_cell("# Theorem Slogan Generation & Benchmarking\\nReproducible, hardware-scalable notebook for dataset evaluation and slogan generation.\\n**Updated with `BATCH_SIZE` configuration to smoothly run on hardware-constrained nodes like RTX 4060 instances.**"),
    nbf.v4.new_code_cell(cell_imports),
    nbf.v4.new_code_cell(cell_config),
    nbf.v4.new_code_cell(cell_slogan_gen),
    nbf.v4.new_code_cell(cell_data),
    nbf.v4.new_code_cell(cell_embed),
    nbf.v4.new_code_cell(cell_eval)
]

with open('benchmark_experiments.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
