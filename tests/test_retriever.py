"""Tests for the TF-IDF retriever."""
import pytest

from src.assistant import RagAssistant
from src.config import CUSTOM_QUESTION, DEMO_QUESTION
from src.retriever import TfidfRetriever


@pytest.fixture(scope="module")
def assistant(documents):
    return RagAssistant(documents)


def test_invalid_analyzer():
    with pytest.raises(ValueError):
        TfidfRetriever("unknown")


def test_demo_question_finds_d04(assistant):
    found = assistant.search(DEMO_QUESTION, top_k=3)

    assert found
    assert found[0]["doc_id"] == "D04"
    assert 0.0 <= float(found[0]["score"]) <= 1.0


def test_custom_question_finds_student_document(assistant):
    found = assistant.search(CUSTOM_QUESTION, top_k=3)

    assert found
    assert found[0]["doc_id"] == "STUDENT"


def test_search_returns_unique_documents(assistant):
    found = assistant.search(DEMO_QUESTION, top_k=5)
    doc_ids = [item["doc_id"] for item in found]

    assert len(doc_ids) == len(set(doc_ids))
    assert len(found) <= 5


def test_empty_question_returns_nothing(assistant):
    assert assistant.search("   ") == []
