"""Pytest configuration for the Block 8 exam project.

- Перенаправляет временную папку pytest в проект (на Windows системный
  temp может быть защищён).
- Даёт сессионные фикстуры: документы и контрольные вопросы.
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

from src.config import CHECK_QUESTIONS_PATH, DOCUMENTS_PATH  # noqa: E402
from src.data_loader import DataLoader  # noqa: E402


@pytest.fixture(scope="session")
def documents() -> list[dict[str, str]]:
    """Учебная база знаний (16 документов + свой STUDENT)."""
    return DataLoader(DOCUMENTS_PATH).load_documents()


@pytest.fixture(scope="session")
def check_questions() -> list[tuple[str, str]]:
    """Контрольные вопросы с известными документами-ответами."""
    return DataLoader.load_check_questions(CHECK_QUESTIONS_PATH)
