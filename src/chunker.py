"""Chunking module (блок 8).

Документы режутся на небольшие фрагменты (чанки) по словам, чтобы не
разрывать слова пополам. Overlap задаёт приблизительное повторение
символов на границе соседних чанков.
"""
from __future__ import annotations

from typing import Iterable


def split_into_chunks(text: str, chunk_size: int = 400, overlap: int = 65) -> list[str]:
    """Разбить текст по словам с приблизительным перекрытием."""
    if not (chunk_size >= 100 and 0 <= overlap < chunk_size):
        raise ValueError("Нужно chunk_size >= 100 и 0 <= overlap < chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start
        current_length = 0
        while end < len(words):
            length_after = current_length + len(words[end]) + (1 if end > start else 0)
            if end > start and length_after > chunk_size:
                break
            current_length = length_after
            end += 1

        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break

        # Отступаем назад: немного повторяем предыдущие слова.
        new_start = end
        repeated = 0
        while new_start > start + 1 and repeated < overlap:
            new_start -= 1
            repeated += len(words[new_start]) + 1
        start = new_start

    return chunks


def build_chunks(
    documents: Iterable[dict[str, str]],
    chunk_size: int = 400,
    overlap: int = 65,
) -> list[dict[str, str]]:
    """Собрать единый список чанков со ссылками на документы."""
    records: list[dict[str, str]] = []
    for doc in documents:
        parts = split_into_chunks(doc["text"], chunk_size, overlap)
        for number, part in enumerate(parts, 1):
            records.append(
                {
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "chunk_id": f"{doc['id']}-{number:02d}",
                    "text": part,
                }
            )
    return records
