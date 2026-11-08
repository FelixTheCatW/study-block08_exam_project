"""Tests for evidence answers."""
import pytest

from src.answerer import NO_ANSWER, EvidenceAnswerer
from src.assistant import RagAssistant
from src.config import UNKNOWN_MIN_SIMILARITY, UNKNOWN_QUESTION

KMEANS_QUESTION = "Почему для KMeans обязательно масштабирование признаков?"


@pytest.fixture(scope="module")
def assistant(documents):
    return RagAssistant(documents)


def test_kmeans_answer_contains_source_and_scaler(assistant):
    result = assistant.ask(KMEANS_QUESTION)

    assert result["answer"] != NO_ANSWER
    assert "S02" in str(result["answer"])
    assert "StandardScaler" in str(result["answer"])


def test_unknown_question_has_no_confirmed_answer(assistant):
    result = assistant.ask(
        UNKNOWN_QUESTION, min_similarity=UNKNOWN_MIN_SIMILARITY
    )

    assert result["answer"] == NO_ANSWER


def test_low_similarity_returns_no_answer(assistant):
    answerer = EvidenceAnswerer(min_similarity=0.99)
    found = assistant.search(KMEANS_QUESTION)
    answer = answerer.answer(KMEANS_QUESTION, found, assistant.retriever)

    assert answer == NO_ANSWER


def test_empty_found_returns_no_answer(assistant):
    answerer = EvidenceAnswerer()
    answer = answerer.answer(KMEANS_QUESTION, [], assistant.retriever)

    assert answer == NO_ANSWER
