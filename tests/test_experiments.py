"""Tests for the experiment runner."""
from src.experiments import ExperimentRunner


def test_run_five_experiments(documents, check_questions):
    results_df, details = ExperimentRunner(documents, check_questions).run()

    assert results_df.shape[0] == 5
    assert list(results_df["опыт"].str[0]) == ["A", "B", "C", "D", "E"]
    assert {"top1", "hit_at_k", "чанков"} <= set(results_df.columns)
    assert results_df["top1"].between(0, 1).all()
    assert results_df["hit_at_k"].between(0, 1).all()
    assert (results_df["hit_at_k"] >= results_df["top1"]).all()
    assert len(details) == 5


def test_experiment_e_changes_several_parameters(documents, check_questions):
    results_df, _ = ExperimentRunner(documents, check_questions).run()
    row_e = results_df[results_df["опыт"].str.startswith("E:")].iloc[0]

    assert row_e["поиск"] == "word"
    assert row_e["overlap"] == 0
    assert row_e["top_k"] == 5
