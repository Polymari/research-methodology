from sentence_transformers import SentenceTransformer
import numpy as np

class TheoremEmbedder:
    def __init__(self, model_name="intfloat/e5-small-v2"):
        """
        Initializes the sentence transformer model.
        e5 models are highly efficient for English and general text embedding.
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def process_text_for_e5(self, texts, prefix="query: "):
        """e5 models require prefixes like 'query: ' or 'passage: ' to perform optimally."""
        if isinstance(texts, str):
            texts = [texts]
        return [f"{prefix}{text}" for text in texts]
    
    def embed_queries(self, queries, batch_size=32):
        """Embed search queries (e.g. natural language search)."""
        processed = self.process_text_for_e5(queries, prefix="query: ")
        embeddings = self.model.encode(processed, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        return np.array(embeddings).astype("float32")
        
    def embed_passages(self, passages, batch_size=32):
        """Embed theorem statements or descriptions for the vector database index."""
        processed = self.process_text_for_e5(passages, prefix="passage: ")
        embeddings = self.model.encode(processed, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        return np.array(embeddings).astype("float32")

if __name__ == "__main__":
    embedder = TheoremEmbedder()
    res = embedder.embed_queries(["Find a theorem about compact manifolds."])
    print(f"Embedding shape: {res.shape}")
