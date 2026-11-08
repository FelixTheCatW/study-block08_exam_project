"""Tests for the DeepSeek generator and .env loader."""
import os

import pytest

from src.deepseek_generator import DeepSeekGenerator
from src.env_loader import load_env_file


def test_generate_uses_api_response(monkeypatch):
    generator = DeepSeekGenerator(api_key="test-key")
    monkeypatch.setattr(
        generator,
        "_post",
        lambda payload: {
            "choices": [{"message": {"content": "Ответ [S01]"}}]
        },
    )

    answer = generator.generate(
        "Что такое переобучение?",
        [{"doc_id": "S01", "title": "Тест", "text": "Текст"}],
    )

    assert answer == "Ответ [S01]"


def test_generate_without_fragments_raises():
    generator = DeepSeekGenerator(api_key="test-key")

    with pytest.raises(RuntimeError):
        generator.generate("Вопрос?", [])


def test_generate_without_key_raises():
    generator = DeepSeekGenerator(api_key=" ")

    with pytest.raises(RuntimeError):
        generator.generate(
            "Вопрос?",
            [{"doc_id": "S01", "title": "Тест", "text": "Текст"}],
        )


def test_load_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCODE_TEST_ENV", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("OPENCODE_TEST_ENV=hello\n", encoding="utf-8")

    load_env_file(env_path)

    assert os.environ["OPENCODE_TEST_ENV"] == "hello"
    monkeypatch.delenv("OPENCODE_TEST_ENV", raising=False)
