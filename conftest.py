"""Pytest configuration for the Block 8 exam project.

- Перенаправляет временную папку pytest в проект.
- Даёт синтетический мини-корпус для быстрых unit-тестов.
- Даёт реальный корпус Handbook для интеграционной проверки
  (пропускается, если корпус ещё не собран).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

TMP_ROOT = BASE_DIR / ".pytest_tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(TMP_ROOT)

from src.config import (  # noqa: E402
    HANDBOOK_CHECK_QUESTIONS_PATH,
    HANDBOOK_CORPUS_PATH,
)
from src.data_loader import DataLoader  # noqa: E402


SYNTHETIC_DOCUMENTS = [
    {
        "id": "S01",
        "title": "Переобучение и кросс-валидация",
        "text": (
            "Переобучение возникает, когда модель запоминает обучающие "
            "данные и плохо работает на новых. Если качество на train "
            "заметно выше, чем на test, модель переобучилась. "
            "Кросс-валидация помогает устойчивее оценить качество. "
            "random_state фиксирует случайность и обеспечивает "
            "воспроизводимость."
        ),
    },
    {
        "id": "S02",
        "title": "KMeans и масштабирование",
        "text": (
            "KMeans распределяет объекты по кластерам по расстоянию. "
            "Поэтому числовые признаки обязательно масштабируют через "
            "StandardScaler: иначе признак с большими числами доминирует. "
            "Число кластеров выбирают методом локтя и по Silhouette Score."
        ),
    },
    {
        "id": "S03",
        "title": "SQL: JOIN, GROUP BY и транзакции",
        "text": (
            "INNER JOIN возвращает только совпавшие строки, а LEFT JOIN — "
            "все строки левой таблицы и совпадения. GROUP BY группирует "
            "строки, WHERE фильтрует до группировки, а HAVING — после. "
            "Транзакция обладает свойствами ACID. Для защиты от "
            "SQL-инъекции используют параметры запроса."
        ),
    },
    {
        "id": "S04",
        "title": "RAG и TF-IDF",
        "text": (
            "RAG — это поиск релевантных фрагментов и передача их "
            "языковой модели. TF-IDF — статистический поиск по словам "
            "или группам символов. Нейросетевой поиск использует "
            "эмбеддинги и cosine similarity. LLM может галлюцинировать, "
            "поэтому ответ проверяют по источникам."
        ),
    },
    {
        "id": "S05",
        "title": "pandas: DataFrame и пропуски",
        "text": (
            "DataFrame — таблица pandas из строк и столбцов. Пропуски "
            "находят через isna().sum() и заполняют медианой или "
            "константой. groupby агрегирует данные по группам, а "
            "pivot_table строит сводную таблицу."
        ),
    },
]


SYNTHETIC_QUESTIONS = [
    ("Что такое переобучение и как его распознать?", ["S01"]),
    ("Почему для KMeans обязательно масштабирование признаков?", ["S02"]),
    ("Чем INNER JOIN отличается от LEFT JOIN?", ["S03"]),
    ("Что такое RAG и TF-IDF?", ["S04"]),
    ("Что такое DataFrame и как найти пропуски?", ["S05"]),
    ("Зачем нужен random_state?", ["S01"]),
    ("Что такое транзакция и SQL-инъекция?", ["S03"]),
]


@pytest.fixture(scope="session")
def documents() -> list[dict[str, str]]:
    """Синтетический мини-корпус для быстрых unit-тестов."""
    return SYNTHETIC_DOCUMENTS


@pytest.fixture(scope="session")
def check_questions() -> list[tuple[str, str | list[str]]]:
    """Синтетические контрольные вопросы для unit-тестов."""
    return SYNTHETIC_QUESTIONS


@pytest.fixture(scope="session")
def handbook_documents() -> list[dict[str, str]]:
    """Реальный корпус Handbook (если он собран локально)."""
    if not HANDBOOK_CORPUS_PATH.exists():
        pytest.skip(
            "Корпус Handbook не собран. Запустите: python main.py --build-corpus"
        )
    return DataLoader(HANDBOOK_CORPUS_PATH).load_documents()


@pytest.fixture(scope="session")
def handbook_check_questions() -> list[tuple[str, str | list[str]]]:
    """Реальные контрольные вопросы по Handbook."""
    return DataLoader.load_check_questions(HANDBOOK_CHECK_QUESTIONS_PATH)
