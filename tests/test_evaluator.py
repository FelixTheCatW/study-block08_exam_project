"""Tests for retrieval evaluation."""
from src.assistant import RagAssistant
from src.evaluator import SearchEvaluator


def test_check_returns_expected_columns(documents, check_questions):
    assistant = RagAssistant(documents)
    detail_df, summary = SearchEvaluator.check(
        assistant.retriever, check_questions, top_k=3
    )

    assert len(detail_df) == len(check_questions)
    assert list(detail_df.columns) == [
        "вопрос",
        "верный документ",
        "первый найденный",
        "top1",
        "hit_at_k",
    ]
    assert 0.0 <= summary["top1_accuracy"] <= 1.0
    assert 0.0 <= summary["hit_at_k"] <= 1.0
    assert summary["top1_accuracy"] <= summary["hit_at_k"] + 1e-9


def test_custom_question_is_hit(documents, check_questions):
    assistant = RagAssistant(documents)
    detail_df, _ = SearchEvaluator.check(
        assistant.retriever, check_questions, top_k=3
    )
    row = detail_df[detail_df["верный документ"] == "STUDENT"].iloc[0]

    assert row["top1"] == 1
    assert row["hit_at_k"] == 1
