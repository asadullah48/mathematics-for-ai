"""Tests for scripts/rag_embeddings.py - the RAG-style retrieval demo."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rag_embeddings import build_index, embed, retrieve, tokenize  # noqa: E402


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Gradient Descent, and Newton's Method!") == [
        "gradient", "descent", "and", "newton", "s", "method",
    ]


def test_embed_is_deterministic():
    v1 = embed("eigenvectors and eigenvalues")
    v2 = embed("eigenvectors and eigenvalues")
    assert np.array_equal(v1, v2)


def test_embed_is_unit_normalized():
    vector = embed("cosine similarity measures the angle between vectors")
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_embed_empty_string_returns_zero_vector_without_dividing_by_zero():
    vector = embed("")
    assert np.array_equal(vector, np.zeros_like(vector))


def test_identical_texts_have_cosine_similarity_one():
    a = embed("matrix multiplication")
    b = embed("matrix multiplication")
    similarity = float(np.dot(a, b))
    assert np.isclose(similarity, 1.0)


def test_retrieve_ranks_most_similar_document_first():
    documents = [
        "Eigenvectors are directions a linear map scales without rotating.",
        "The weather today is sunny with a light breeze.",
    ]
    index = build_index(documents)

    results = retrieve("What are eigenvectors and eigenvalues?", documents, index, top_k=2)

    assert len(results) == 2
    top_doc, top_score = results[0]
    assert top_doc == documents[0]
    assert top_score >= results[1][1]


def test_retrieve_respects_top_k():
    documents = [f"document number {i} about linear algebra" for i in range(5)]
    index = build_index(documents)

    results = retrieve("linear algebra", documents, index, top_k=2)

    assert len(results) == 2
