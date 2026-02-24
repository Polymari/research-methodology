import pandas as pd
import numpy as np
import time
from data_loader import load_theorems
from embedder import TheoremEmbedder
from retriever import VectorRetriever

def run_pipeline():
    print("--- Semantic Search Pipeline ---")
    
    # 1. Load Data
    print("\n1. Loading Data...")
    df_theorems = load_theorems()
    
    # Let's take a small subset for a quick test so it doesn't take hours
    subset_size = 1000
    df_sample =  df_theorems.head(subset_size).copy()
    print(f"Using a sample of {subset_size} theorems for testing.")
    
    # 2. Embed Documents
    print("\n2. Initializing Embedder & Embedding Documents...")
    embedder = TheoremEmbedder()
    
    # The 'text' column likely holds the theorem statement
    # We will embed the 'body' column which holds the theorem
    passages = df_sample['body'].fillna("").tolist() 
    
    start_time = time.time()
    passage_embeddings = embedder.embed_passages(passages)
    print(f"Embedding {subset_size} passages took {time.time() - start_time:.2f} seconds.")
    
    # 3. Build Vector Index
    print("\n3. Building Vector Index...")
    embedding_dim = passage_embeddings.shape[1]
    retriever = VectorRetriever(embedding_dim=embedding_dim)
    retriever.add_embeddings(passage_embeddings)
    
    # 4. Search and Retrieve
    print("\n4. Running Search Queries...")
    sample_queries = [
        "What is the Pythagorean theorem?",
        "A theorem about the properties of continuous functions on compact spaces",
        "Linear algebra eigenvalue properties"
    ]
    
    query_embeddings = embedder.embed_queries(sample_queries)
    distances, indices = retriever.search(query_embeddings, top_k=3)
    
    print("\n--- Search Results ---")
    for i, query in enumerate(sample_queries):
        print(f"\nQuery: '{query}'")
        for rank, (dist, idx) in enumerate(zip(distances[i], indices[i])):
            theorem_id = df_sample.iloc[idx]['theorem_id'] if 'theorem_id' in df_sample.columns else idx
            # print a snippet of the text
            text_snippet = df_sample.iloc[idx]['body'][:200].replace('\n', ' ') + "..."
            print(f"  Rank {rank+1} (Score: {dist:.4f}) [ID: {theorem_id}]: {text_snippet}")

if __name__ == "__main__":
    run_pipeline()
