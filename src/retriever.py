import faiss
import numpy as np
import os

class VectorRetriever:
    def __init__(self, embedding_dim):
        """
        Initializes FAISS with Inner Product search.
        Since embeddings from sentence-transformers are normalized (L2 norm = 1),
        Inner Product is equivalent to Cosine Similarity.
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        
    def add_embeddings(self, embeddings: np.ndarray):
        """Add vectors to the FAISS index."""
        assert embeddings.dtype == np.float32, "FAISS requires float32 embeddings."
        assert embeddings.shape[1] == self.embedding_dim, f"Expected dimension {self.embedding_dim}, got {embeddings.shape[1]}"
        self.index.add(embeddings)
        print(f"Added {embeddings.shape[0]} vectors. Total in index: {self.index.ntotal}")

    def search(self, query_embeddings: np.ndarray, top_k=10):
        """Search the FAISS index for the closest top_k vectors."""
        assert query_embeddings.dtype == np.float32, "FAISS requires float32 query embeddings."
        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices

    def save_index(self, filepath: str):
        """Save the FAISS index to disk."""
        faiss.write_index(self.index, filepath)
        print(f"Index saved to {filepath}")

    def load_index(self, filepath: str):
        """Load the FAISS index from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No index found at {filepath}")
        self.index = faiss.read_index(filepath)
        print(f"Index loaded from {filepath}. Total vectors: {self.index.ntotal}")
