"""TF-IDF retrieval module (блок 8).

Обычный статистический поиск, без обучения глубокой нейросети.
`analyzer='word'` сравнивает слова и биграммы, `analyzer='char_wb'`
сравнивает группы символов и лучше переносит разные окончания.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    """Класс TF-IDF индекса и поиска ближайших чанков."""

    def __init__(self, analyzer: str = "char_wb"):
        if analyzer not in {"word", "char_wb"}:
            raise ValueError("analyzer должен быть 'word' или 'char_wb'")
        self.analyzer = analyzer
        self.records: list[dict[str, str]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    def build(self, records: list[dict[str, str]]) -> "TfidfRetriever":
        """Построить индекс по готовым чанкам."""
        if not records:
            raise ValueError("Нельзя построить индекс без чанков")

        if self.analyzer == "word":
            vectorizer = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), lowercase=True
            )
        else:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), lowercase=True
            )

        self.records = records
        self.vectorizer = vectorizer
        self.matrix = vectorizer.fit_transform([rec["text"] for rec in records])
        return self

    def transform(self, texts: list[str]):
        """Преобразовать тексты в TF-IDF-векторы."""
        if self.vectorizer is None:
            raise RuntimeError("Сначала вызовите build()")
        return self.vectorizer.transform(texts)

    def search(self, question: str, top_k: int = 3) -> list[dict[str, object]]:
        """Найти до top_k разных документов с наиболее близкими фрагментами."""
        if not question.strip() or self.matrix is None:
            return []

        query_vector = self.transform([question])
        scores = cosine_similarity(query_vector, self.matrix).ravel()

        used_docs: set[str] = set()
        results: list[dict[str, object]] = []
        for index in np.argsort(-scores):
            record = self.records[int(index)]
            if record["doc_id"] in used_docs:
                continue
            used_docs.add(record["doc_id"])
            results.append({**record, "score": float(scores[int(index)])})
            if len(results) >= top_k:
                break
        return results

    def score_sentences(self, question: str, sentences: list[str]):
        """Посчитать сходство вопроса с отдельными предложениями."""
        query_vector = self.transform([question])
        sentence_matrix = self.transform(sentences)
        return cosine_similarity(query_vector, sentence_matrix).ravel()
