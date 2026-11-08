"""Optional neural retrieval (блок 8, факультатив).

Поиск по смыслу через эмбеддинги предобученной нейросети
`sentence-transformers`. Требует интернет для первой загрузки модели
и пакет `sentence-transformers`. Обязательная часть проекта работает
без этого модуля.
"""
from __future__ import annotations

import numpy as np

from src.config import NEURAL_MODEL_ID


class NeuralRetriever:
    """Класс семантического поиска по эмбеддингам."""

    def __init__(self, model_id: str = NEURAL_MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.records: list[dict[str, str]] = []
        self.vectors = None

    @staticmethod
    def is_available() -> bool:
        """Проверить, установлен ли sentence-transformers."""
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def build(self, records: list[dict[str, str]]) -> "NeuralRetriever":
        """Загрузить модель и посчитать эмбеддинги чанков."""
        from sentence_transformers import SentenceTransformer

        self.records = records
        self.model = SentenceTransformer(self.model_id)
        self.vectors = self.model.encode(
            [record["text"] for record in records],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self

    def search(self, question: str, top_k: int = 3) -> list[dict[str, object]]:
        """Найти до top_k разных документов по смысловому сходству."""
        if self.model is None or self.vectors is None:
            raise RuntimeError("Сначала вызовите build()")

        question_vector = self.model.encode(
            [question], normalize_embeddings=True
        )[0]
        scores = self.vectors @ question_vector

        used_docs: set[str] = set()
        results: list[dict[str, object]] = []
        for position in np.argsort(-scores):
            record = self.records[int(position)]
            if record["doc_id"] in used_docs:
                continue
            used_docs.add(record["doc_id"])
            results.append({**record, "score": float(scores[int(position)])})
            if len(results) >= top_k:
                break
        return results
