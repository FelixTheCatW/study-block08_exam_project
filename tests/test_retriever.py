"""Tests for the TF-IDF retriever."""
import pytest

from src.assistant import RagAssistant
from src.retriever import TfidfRetriever

OVERFITTING_QUESTION = "Что такое переобучение и как его распознать?"
KMEANS_QUESTION = "Почему для KMeans обязательно масштабирование признаков?"


@pytest.fixture(scope="module")
def assistant(documents):
    return RagAssistant(documents)


def test_invalid_analyzer():
    with pytest.raises(ValueError):
        TfidfRetriever("unknown")


def test_overfitting_question_finds_s01(assistant):
    found = assistant.search(OVERFITTING_QUESTION, top_k=3)

    assert found
    assert found[0]["doc_id"] == "S01"
    assert 0.0 <= float(found[0]["score"]) <= 1.0


def test_kmeans_question_finds_s02(assistant):
    found = assistant.search(KMEANS_QUESTION, top_k=3)

    assert found
    assert found[0]["doc_id"] == "S02"


def test_search_returns_unique_documents(assistant):
    found = assistant.search(OVERFITTING_QUESTION, top_k=5)
    doc_ids = [item["doc_id"] for item in found]

    assert len(doc_ids) == len(set(doc_ids))
    assert len(found) <= 5


def test_empty_question_returns_nothing(assistant):
    assert assistant.search("   ") == []
