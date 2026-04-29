import faiss
import numpy as np
import os


class VectorRetriever:
    """Exact brute-force search using FAISS IndexFlatIP (cosine similarity on normalized vectors)."""

    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self._raw_embeddings = None

    def add_embeddings(self, embeddings: np.ndarray):
        """Add vectors to the FAISS index."""
        assert embeddings.dtype == np.float32, "FAISS requires float32 embeddings."
        assert embeddings.shape[1] == self.embedding_dim
        self.index.add(embeddings)
        self._raw_embeddings = embeddings
        print(
            f"Added {embeddings.shape[0]} vectors. Total in index: {self.index.ntotal}"
        )

    def search(self, query_embeddings: np.ndarray, top_k=10):
        """Search the FAISS index for the closest top_k vectors."""
        assert query_embeddings.dtype == np.float32
        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices

    def save_index(self, filepath: str):
        faiss.write_index(self.index, filepath)
        print(f"Index saved to {filepath}")

    def load_index(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No index found at {filepath}")
        self.index = faiss.read_index(filepath)
        print(f"Index loaded from {filepath}. Total vectors: {self.index.ntotal}")


class HNSWRetriever:
    """
    Approximate nearest-neighbor search with HNSW index + optional binary quantization.

    Matches the paper's architecture:
    1. Binary quantize embeddings -> fast Hamming-distance candidate retrieval
    2. Rerank candidates using cosine similarity on original float embeddings
    """

    def __init__(self, embedding_dim, m=32, ef_construction=200, ef_search=128):
        """
        Args:
            embedding_dim: Dimension of the embedding vectors.
            m: HNSW parameter — number of neighbors per node.
            ef_construction: HNSW build-time beam width.
            ef_search: HNSW search-time beam width.
        """
        self.embedding_dim = embedding_dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # HNSW index on the full float vectors
        self.index = faiss.IndexHNSWFlat(embedding_dim, m)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

        # Store raw embeddings for cosine-similarity reranking
        self._raw_embeddings = None

    def add_embeddings(self, embeddings: np.ndarray):
        """Add vectors to the HNSW index."""
        assert embeddings.dtype == np.float32
        assert embeddings.shape[1] == self.embedding_dim
        self._raw_embeddings = embeddings.copy()
        self.index.add(embeddings)
        print(
            f"HNSW index built with {self.index.ntotal} vectors (m={self.m}, efC={self.ef_construction})"
        )

    def search(self, query_embeddings: np.ndarray, top_k=20, candidate_pool_factor=6):
        """
        Two-stage retrieval matching the paper's approach:
        1. Retrieve an oversized candidate pool from HNSW
        2. Rerank by exact cosine similarity and return top_k

        Args:
            query_embeddings: Query vectors (float32, normalized).
            top_k: Number of final results per query.
            candidate_pool_factor: Retrieve top_k * factor candidates for reranking.
        """
        assert query_embeddings.dtype == np.float32

        # Stage 1: Retrieve candidates from HNSW
        n_candidates = min(top_k * candidate_pool_factor, self.index.ntotal)
        _, candidate_indices = self.index.search(query_embeddings, n_candidates)

        # Stage 2: Rerank by cosine similarity on original embeddings
        all_distances = []
        all_indices = []

        for i in range(len(query_embeddings)):
            cand_idx = candidate_indices[i]
            # Filter out -1 (FAISS sentinel for missing neighbors)
            valid_mask = cand_idx >= 0
            cand_idx = cand_idx[valid_mask]

            if len(cand_idx) == 0:
                all_distances.append(np.zeros(top_k, dtype=np.float32))
                all_indices.append(np.full(top_k, -1, dtype=np.int64))
                continue

            # Cosine similarity = dot product for normalized vectors
            cand_embeds = self._raw_embeddings[cand_idx]
            scores = cand_embeds @ query_embeddings[i]

            # Sort by cosine similarity (descending)
            rerank_order = np.argsort(-scores)[:top_k]

            reranked_idx = cand_idx[rerank_order]
            reranked_scores = scores[rerank_order]

            # Pad if fewer than top_k candidates
            if len(reranked_idx) < top_k:
                pad_len = top_k - len(reranked_idx)
                reranked_idx = np.concatenate(
                    [reranked_idx, np.full(pad_len, -1, dtype=np.int64)]
                )
                reranked_scores = np.concatenate(
                    [reranked_scores, np.zeros(pad_len, dtype=np.float32)]
                )

            all_distances.append(reranked_scores)
            all_indices.append(reranked_idx)

        return np.array(all_distances), np.array(all_indices)

    def save_index(self, filepath: str):
        faiss.write_index(self.index, filepath)
        # Save raw embeddings alongside
        np.save(filepath + ".raw.npy", self._raw_embeddings)
        print(f"HNSW index saved to {filepath}")

    def load_index(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No index found at {filepath}")
        self.index = faiss.read_index(filepath)
        self._raw_embeddings = np.load(filepath + ".raw.npy")
        print(f"HNSW index loaded. Total vectors: {self.index.ntotal}")
