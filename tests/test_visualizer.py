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
    visualizer = Visualizer(tmp_path)

    metrics_path = visualizer.save_experiment_metrics(
        results, "metrics.png", "Тест"
    )
    chunks_path = visualizer.save_chunks_per_experiment(
        results, "chunks.png", "Тест"
    )

    assert metrics_path.is_file()
    assert metrics_path.stat().st_size > 0
    assert chunks_path.is_file()
    assert chunks_path.stat().st_size > 0
