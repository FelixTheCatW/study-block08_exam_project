"""Visualization module (блок 8).

Сохраняет графики экспериментов: Top-1 / Hit@K по вариантам и число
чанков, получившихся при разных размерах.
"""
from __future__ import annotations

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

    @staticmethod
    def _short_name(name: str) -> str:
        """Оставить в подписи только код эксперимента (A, B, C, D, E)."""
        return str(name).split(":", 1)[0].strip()
