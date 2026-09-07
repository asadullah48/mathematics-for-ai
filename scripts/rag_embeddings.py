"""Practical application: a tiny RAG-style retrieval demo.

Shows how the linear algebra already in this repo (cosine similarity) is
the actual mechanism behind "semantic search" in a Retrieval-Augmented
Generation pipeline: embed a corpus once, embed a query, rank by
similarity, return the top matches to feed to an LLM as context.

No external embedding API/model is used here on purpose - this is a math
demo, not a production RAG system, and it should run offline with only
this repo's own dependencies (numpy). `embed()` uses a deterministic
bag-of-words + hashing vectorizer so the same text always maps to the same
vector; swap it for a real embedding model (OpenAI, Sentence-Transformers,
Claude, ...) in a production system - the retrieval math below (cosine
similarity ranking) is unchanged either way.

Run directly:
    python scripts/rag_embeddings.py
"""
from __future__ import annotations  # list[str]/dict[...] annotations need this on Python 3.8

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from math_utils.linear_algebra import LinearAlgebra  # noqa: E402

_TOKEN_RE = re.compile(r"[a-z0-9]+")
EMBEDDING_DIM = 256


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization - deliberately simple."""
    return _TOKEN_RE.findall(text.lower())


def embed(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Deterministic hashing-vectorizer embedding: each token votes +1/-1
    into a pseudo-random dimension of a fixed-size vector (the classic
    "feature hashing" trick), then the vector is L2-normalized. Same text
    in -> same vector out, every time, with no model download.

    This is a stand-in for a real embedding model - the point of this
    script is the retrieval math (cosine similarity ranking), not the
    embedding quality.
    """
    vector = np.zeros(dim)
    for token in tokenize(text):
        h = hash(token)
        index = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def build_index(documents: list[str]) -> np.ndarray:
    """Embed every document once - the "index" a real RAG system persists
    in a vector store (FAISS, pgvector, Pinecone, ...)."""
    return np.stack([embed(doc) for doc in documents])


def retrieve(query: str, documents: list[str], index: np.ndarray, top_k: int = 3):
    """Rank `documents` against `query` by cosine similarity - the actual
    retrieval step in "Retrieval-Augmented Generation". Returns
    (document, score) pairs, most similar first.
    """
    query_vector = embed(query)
    scores = [LinearAlgebra.cosine_similarity(query_vector, doc_vector) for doc_vector in index]
    ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


_DEMO_CORPUS = [
    "Eigenvectors are directions a linear map scales without rotating.",
    "Gradient descent minimizes a loss function by stepping against its gradient.",
    "The determinant of a matrix tells you how it scales area or volume.",
    "Cosine similarity measures the angle between two vectors, not their magnitude.",
    "Newton's method uses the Hessian for faster, curvature-aware convergence.",
    "A neural network's forward pass is a sequence of matrix multiplications and nonlinearities.",
]


def main() -> None:
    index = build_index(_DEMO_CORPUS)
    query = "How do neural networks use matrices?"

    results = retrieve(query, _DEMO_CORPUS, index, top_k=3)

    print(f"Query: {query!r}\n")
    print("Top matches (retrieved context for an LLM prompt):")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"  {rank}. (cosine={score:.3f}) {doc}")


if __name__ == "__main__":
    main()
