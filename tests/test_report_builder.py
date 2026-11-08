"""Tests for the markdown report builder."""
import pandas as pd

from src.answerer import NO_ANSWER
from src.report_builder import ReportBuilder


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"опыт": f"{code}: тест", "чанк": 200, "overlap": 30,
             "поиск": "char_wb", "top_k": 3, "чанков": 40,
             "top1": 0.9, "hit_at_k": 1.0}
            for code in ["A", "B", "C", "D", "E"]
        ]
    )


def test_report_contains_key_sections(tmp_path):
    documents = [
        {"id": "D01", "title": "Тестовый документ", "text": "Текст " * 40},
        {"id": "STUDENT", "title": "Свой документ", "text": "Текст " * 40},
    ]
    settings = {
        "chunk_size": 230,
        "overlap": 30,
        "search_method": "char_wb",
        "top_k": 3,
        "unknown_min_similarity": 0.2,
    }
    custom_result = {
        "question": "Свой вопрос?",
        "found": [{"doc_id": "STUDENT", "title": "Свой документ", "score": 0.5}],
        "answer": "• Ответ. [Источник: STUDENT]",
    }
    unknown_result = {
        "question": "Посторонний вопрос?",
        "found": [{"doc_id": "D01", "title": "Тестовый документ", "score": 0.1}],
        "answer": NO_ANSWER,
    }

    builder = ReportBuilder(tmp_path / "report.md")
    text = builder.build_report(
        documents=documents,
        settings=settings,
        results_df=_results(),
        baseline_summary={"top1_accuracy": 1.0, "hit_at_k": 1.0},
        custom_result=custom_result,
        unknown_result=unknown_result,
        limitations=["Ограничение 1"],
    )
    builder.save(text)

    assert "## 4. Пять экспериментов" in text
    assert "STUDENT" in text
    assert NO_ANSWER in text
    assert "TF-IDF" in text
    assert (tmp_path / "report.md").is_file()
