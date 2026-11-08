"""Tests for chunking."""
import pytest

from src.chunker import build_chunks, split_into_chunks


def test_split_returns_non_empty_chunks():
    text = "Первое предложение. " * 80
    chunks = split_into_chunks(text, chunk_size=200, overlap=30)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_split_does_not_break_words():
    text = "слово " * 100
    chunks = split_into_chunks(text, chunk_size=120, overlap=0)

    for chunk in chunks:
        assert set(chunk.split()) == {"слово"}


def test_split_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        split_into_chunks("текст", chunk_size=50, overlap=10)
    with pytest.raises(ValueError):
        split_into_chunks("текст", chunk_size=200, overlap=200)


def test_build_chunks_links_to_documents(documents):
    records = build_chunks(documents, chunk_size=230, overlap=30)

    assert len(records) >= len(documents)
    assert {record["doc_id"] for record in records} == {
        doc["id"] for doc in documents
    }
    assert all(record["chunk_id"].startswith(record["doc_id"]) for record in records)
