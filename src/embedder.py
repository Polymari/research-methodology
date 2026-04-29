from sentence_transformers import SentenceTransformer
import numpy as np

# Math-specific task instructions for Qwen3-Embedding models
# Based on the paper's Table 10 and LeanSearch-style prompts
QUERY_INSTRUCTION = (
    "Instruct: Given a mathematical search query, retrieve the theorem statement "
    "that is semantically closest to the query.\nQuery: "
)
PASSAGE_INSTRUCTION = (
    "Instruct: Represent this mathematical theorem description for retrieval.\nQuery: "
)


class TheoremEmbedder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        """
        Initializes the embedding model.

        Default: Qwen3-Embedding-0.6B (paper uses 8B for best results,
        0.6B is a good tradeoff for development).

        Args:
            model_name: HuggingFace model identifier.
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._is_qwen = "qwen" in model_name.lower()

    def embed_queries(self, queries, batch_size=32):
        """
        Embed search queries (natural language math questions).

        For Qwen3 models, uses the math-specific task instruction.
        """
        if isinstance(queries, str):
            queries = [queries]

        if self._is_qwen:
            processed = [f"{QUERY_INSTRUCTION}{q}" for q in queries]
        else:
            processed = [f"query: {q}" for q in queries]

        embeddings = self.model.encode(
            processed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.array(embeddings).astype("float32")

    def embed_passages(self, passages, batch_size=32):
        """
        Embed theorem slogans (or raw LaTeX for ablation) for indexing.

        For Qwen3 models, uses the math-specific passage instruction.
        """
        if isinstance(passages, str):
            passages = [passages]

        if self._is_qwen:
            processed = [f"{PASSAGE_INSTRUCTION}{p}" for p in passages]
        else:
            processed = [f"passage: {p}" for p in passages]

        embeddings = self.model.encode(
            processed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.array(embeddings).astype("float32")

    def embed_unprompted(self, texts, batch_size=32):
        """
        Embed texts without any task instruction prefix (for ablation studies).
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.array(embeddings).astype("float32")


if __name__ == "__main__":
    embedder = TheoremEmbedder()
    res = embedder.embed_queries(["Find a theorem about compact manifolds."])
    print(f"Embedding shape: {res.shape}")
    print(f"Embedding dtype: {res.dtype}")
    print(f"L2 norm: {np.linalg.norm(res[0]):.4f}")
