"""Tests for chart saving."""
import pandas as pd

from src.visualizer import Visualizer


def test_save_charts(tmp_path):
    results = pd.DataFrame(
        [
            {
                "опыт": "A: слова / 1 источник",
                "чанк": 380,
                "overlap": 45,
                "поиск": "word",
                "top_k": 1,
                "чанков": 20,
                "top1": 0.9,
                "hit_at_k": 0.9,
            },
            {
                "опыт": "E: слова / без перекрытия / 5 источников",
                "чанк": 300,
                "overlap": 0,
                "поиск": "word",
                "top_k": 5,
                "чанков": 18,
                "top1": 0.8,
                "hit_at_k": 1.0,
            },
        ]
    )
    documents = [
        {
            "id": "README",
            "title": "README",
            "text": "Текст " * 100,
            "category": "handbook",
            "kind": "md",
            "words": 100,
        },
        {
            "id": "konspekty/ml",
            "title": "Конспект ML",
            "text": "Текст " * 200,
            "category": "konspekt",
            "kind": "md",
            "words": 200,
        },
        {
            "id": "istochniki/guide",
            "title": "Guide",
            "text": "Текст " * 300,
            "category": "source",
            "kind": "docx",
            "words": 300,
        },
    ]
    visualizer = Visualizer(tmp_path)

    paths = [
        visualizer.save_experiment_metrics(results, "metrics.png", "Тест"),
        visualizer.save_chunks_per_experiment(results, "chunks.png", "Тест"),
        visualizer.save_corpus_composition(
            documents, "composition.png", "Тест"
        ),
        visualizer.save_document_sizes(documents, "sizes.png", "Тест"),
        visualizer.save_topk_gain(results, "gain.png", "Тест"),
    ]

    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0
