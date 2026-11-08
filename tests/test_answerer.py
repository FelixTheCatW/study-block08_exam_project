"""Tests for evidence answers."""
import pytest

from src.answerer import NO_ANSWER, EvidenceAnswerer
from src.assistant import RagAssistant
from src.config import (
    CUSTOM_QUESTION,
    UNKNOWN_MIN_SIMILARITY,
    UNKNOWN_QUESTION,
)


@pytest.fixture(scope="module")
def assistant(documents):
    return RagAssistant(documents)


def test_custom_answer_contains_source_and_flag(assistant):
    result = assistant.ask(CUSTOM_QUESTION)

    assert result["answer"] != NO_ANSWER
    assert "STUDENT" in str(result["answer"])
    assert "--ask" in str(result["answer"])


def test_unknown_question_has_no_confirmed_answer(assistant):
    result = assistant.ask(
        UNKNOWN_QUESTION, min_similarity=UNKNOWN_MIN_SIMILARITY
    )

    assert result["answer"] == NO_ANSWER


def test_low_similarity_returns_no_answer(assistant):
    answerer = EvidenceAnswerer(min_similarity=0.99)
    found = assistant.search(CUSTOM_QUESTION)
    answer = answerer.answer(CUSTOM_QUESTION, found, assistant.retriever)

    assert answer == NO_ANSWER


def test_empty_found_returns_no_answer(assistant):
    answerer = EvidenceAnswerer()
    answer = answerer.answer(CUSTOM_QUESTION, [], assistant.retriever)

    assert answer == NO_ANSWER
