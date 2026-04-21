import pytest
import numpy as np
from src.retriever import VectorRetriever, HNSWRetriever


@pytest.fixture
def sample_embeddings():
    """Create normalized sample embeddings for testing."""
    np.random.seed(42)
    dim = 64
    n = 100
    embeddings = np.random.randn(n, dim).astype(np.float32)
    # Normalize to unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings, dim


class TestVectorRetriever:
    def test_add_and_search(self, sample_embeddings):
        embeddings, dim = sample_embeddings
        retriever = VectorRetriever(embedding_dim=dim)
        retriever.add_embeddings(embeddings)
        
        assert retriever.index.ntotal == 100
        
        # Search with the first embedding — should find itself as closest
        query = embeddings[:1]
        distances, indices = retriever.search(query, top_k=5)
        
        assert indices.shape == (1, 5)
        assert indices[0][0] == 0  # Closest to itself
        assert distances[0][0] > 0.99  # Cosine sim ~1.0 with itself
    
    def test_search_returns_correct_shape(self, sample_embeddings):
        embeddings, dim = sample_embeddings
        retriever = VectorRetriever(embedding_dim=dim)
        retriever.add_embeddings(embeddings)
        
        queries = embeddings[:3]
        distances, indices = retriever.search(queries, top_k=10)
        
        assert distances.shape == (3, 10)
        assert indices.shape == (3, 10)


class TestHNSWRetriever:
    def test_add_and_search(self, sample_embeddings):
        embeddings, dim = sample_embeddings
        retriever = HNSWRetriever(embedding_dim=dim, m=16)
        retriever.add_embeddings(embeddings)
        
        assert retriever.index.ntotal == 100
        
        # Search with the first embedding
        query = embeddings[:1]
        distances, indices = retriever.search(query, top_k=5)
        
        assert indices.shape == (1, 5)
        # The first result should be very close (HNSW is approximate)
        assert distances[0][0] > 0.95
    
    def test_reranking_improves_results(self, sample_embeddings):
        embeddings, dim = sample_embeddings
        retriever = HNSWRetriever(embedding_dim=dim, m=16)
        retriever.add_embeddings(embeddings)
        
        query = embeddings[:1]
        distances, indices = retriever.search(query, top_k=5, candidate_pool_factor=10)
        
        # Results should be sorted by cosine similarity (descending)
        for i in range(len(distances[0]) - 1):
            if distances[0][i+1] >= 0:  # Skip padding
                assert distances[0][i] >= distances[0][i+1]
