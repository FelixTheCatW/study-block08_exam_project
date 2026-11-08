"""Data loading module (блок 8).

Загружает учебную базу знаний и контрольные вопросы, а также сохраняет
табличные и текстовые артефакты отчёта. Загрузка данных не смешивается
с поиском и оценкой.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class DataLoader:
    """Класс для загрузки документов и сохранения результатов."""

    def __init__(self, documents_path: Path):
        self.documents_path = Path(documents_path)

    def load_documents(self) -> list[dict[str, str]]:
        """Загрузить список документов базы знаний."""
        if not self.documents_path.exists():
            raise FileNotFoundError(f"База знаний не найдена: {self.documents_path}")
        documents = json.loads(self.documents_path.read_text(encoding="utf-8"))
        self._validate_documents(documents)
        return documents

    @staticmethod
    def load_check_questions(path: Path) -> list[tuple[str, str | list[str]]]:
        """Загрузить контрольные вопросы и ожидаемый документ-ответ.

        Поле `expected` может быть строкой или списком допустимых
        источников: в большом корпусе ответ часто есть в нескольких
        документах (конспект, первоисточник, глоссарий).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Контрольные вопросы не найдены: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [(item["question"], item["expected"]) for item in raw]

    @staticmethod
    def _validate_documents(documents: list[dict[str, str]]) -> None:
        if not isinstance(documents, list) or not documents:
            raise ValueError("База знаний должна быть непустым списком документов")
        ids = [doc["id"] for doc in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Идентификаторы документов должны быть уникальны")
        for doc in documents:
            if not {"id", "title", "text"} <= set(doc):
                raise ValueError(f"У документа не хватает полей: {doc}")

    @staticmethod
    def save_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
        """Сохранить DataFrame в CSV (utf-8-sig для Excel)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False, encoding="utf-8-sig")

    @staticmethod
    def save_json(data: dict, path: Path) -> None:
        """Сохранить словарь в JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def save_text(text: str, path: Path) -> None:
        """Сохранить текст в файл."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
