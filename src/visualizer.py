"""Visualization module (блок 8).

Сохраняет графики экспериментов: Top-1 / Hit@K по вариантам и число
чанков, получившихся при разных размерах.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:
    """Класс сохранения графиков отчёта."""

    def __init__(self, charts_dir: Path):
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    def save_experiment_metrics(
        self,
        results: pd.DataFrame,
        filename: str,
        title: str,
    ) -> Path:
        """Две панели: Top-1 и Hit@K по экспериментам."""
        output_path = self.charts_dir / filename
        labels = [self._short_name(name) for name in results["опыт"]]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].bar(labels, results["top1"], color="#4C72B0")
        axes[0].set_title("Правильный документ на первом месте (Top-1)")
        axes[1].bar(labels, results["hit_at_k"], color="#55A868")
        axes[1].set_title("Правильный документ среди TOP-K (Hit@K)")

        for ax in axes:
            ax.set_ylabel("Доля правильных ответов (0–1)")
            ax.set_ylim(0.0, 1.02)
            ax.tick_params(axis="x", labelrotation=25)
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path

    def save_chunks_per_experiment(
        self,
        results: pd.DataFrame,
        filename: str,
        title: str,
    ) -> Path:
        """Столбчатая диаграмма числа чанков по экспериментам."""
        output_path = self.charts_dir / filename
        labels = [self._short_name(name) for name in results["опыт"]]

        plt.figure(figsize=(11, 5))
        plt.bar(labels, results["чанков"], color="#C44E52")
        plt.title(title)
        plt.xlabel("Эксперимент")
        plt.ylabel("Количество чанков")
        plt.xticks(rotation=25, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close()
        return output_path

    def save_corpus_composition(
        self,
        documents: list[dict[str, object]],
        filename: str,
        title: str,
    ) -> Path:
        """Две панели: документы по категориям и по типу файла."""
        output_path = self.charts_dir / filename
        category_names = {
            "handbook": "тематические",
            "konspekt": "конспекты",
            "source": "первоисточники",
        }
        kind_names = {"md": "Markdown", "docx": "Word"}

        categories = Counter(
            str(doc.get("category", "unknown")) for doc in documents
        )
        kinds = Counter(str(doc.get("kind", "unknown")) for doc in documents)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        cat_labels = [category_names.get(key, key) for key in categories]
        axes[0].bar(cat_labels, list(categories.values()), color="#4C72B0")
        axes[0].set_title("Документы по категориям")
        axes[0].set_ylabel("Количество документов")
        axes[0].tick_params(axis="x", labelrotation=15)
        for index, value in enumerate(categories.values()):
            axes[0].text(index, value, str(value), ha="center", va="bottom")

        kind_labels = [kind_names.get(key, key) for key in kinds]
        axes[1].bar(kind_labels, list(kinds.values()), color="#55A868")
        axes[1].set_title("Документы по типу файла")
        axes[1].set_ylabel("Количество документов")
        for index, value in enumerate(kinds.values()):
            axes[1].text(index, value, str(value), ha="center", va="bottom")

        for ax in axes:
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path

    def save_document_sizes(
        self,
        documents: list[dict[str, object]],
        filename: str,
        title: str,
        top_n: int = 15,
    ) -> Path:
        """Горизонтальная диаграмма самых крупных документов по словам."""
        output_path = self.charts_dir / filename
        top = sorted(
            documents,
            key=lambda doc: int(doc.get("words", 0)),
        )[-top_n:]
        labels = [
            str(doc["id"])[:55] + ("…" if len(str(doc["id"])) > 55 else "")
            for doc in top
        ]
        values = [int(doc.get("words", 0)) for doc in top]

        plt.figure(figsize=(11, 7))
        plt.barh(labels, values, color="#8172B2")
        plt.title(title)
        plt.xlabel("Количество слов")
        plt.ylabel("Документ")
        for index, value in enumerate(values):
            plt.text(value, index, f" {value}", va="center")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close()
        return output_path

    def save_topk_gain(
        self,
        results: pd.DataFrame,
        filename: str,
        title: str,
    ) -> Path:
        """Столбчатая диаграмма прироста Hit@K относительно Top-1."""
        output_path = self.charts_dir / filename
        labels = [self._short_name(name) for name in results["опыт"]]
        gain = results["hit_at_k"] - results["top1"]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(labels, gain, color="#C44E52")
        plt.title(title)
        plt.xlabel("Эксперимент")
        plt.ylabel("Hit@K − Top-1")
        plt.ylim(0, max(0.35, float(gain.max()) + 0.05))
        plt.grid(True, axis="y", alpha=0.3)
        for bar, value in zip(bars, gain):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close()
        return output_path

    @staticmethod
    def _short_name(name: str) -> str:
        """Оставить в подписи только код эксперимента (A, B, C, D, E)."""
        return str(name).split(":", 1)[0].strip()
