"""Tests for ANSI console styling."""
from src import console_style as cs
from src.assistant import RagAssistant


def test_colors_disabled_with_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)

    assert cs.question("вопрос") == "вопрос"
    assert cs.source("источник") == "источник"


def test_colors_forced(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")

    assert "\033[" in cs.question("вопрос")
    assert "\033[" in cs.source("источник")


def test_format_result_colored(documents, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assistant = RagAssistant(documents)
    result = assistant.ask("Что такое переобучение и как его распознать?")

    text = assistant.format_result(result, color=True)

    assert "\033[" in text
    assert "Вопрос:" in text
    assert "Найденные источники:" in text
