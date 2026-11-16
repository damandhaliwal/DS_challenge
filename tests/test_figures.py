"""Tests for reporting/figures.py, on small made-up result tables.

The figures are checked by eye when they change; these tests check that every figure
builds from tables of the saved shape, and that the numbers behind the risk figures
are right.
"""

import numpy as np
import pandas as pd
import pytest

import config
from cleaning.clean_incidents import clean_incidents
from cleaning.load import read_raw_incidents
from reporting import figures
from reporting.figures import major_rate_by, wilson_interval

MODELS = ["base_rate", "symptom_line_table", "logistic", "gradient_boosting"]


@pytest.fixture
def saved_results(tmp_path, monkeypatch):
    """Made-up tables with the columns the real outputs/backtest/ files have."""
    budgets = np.round(np.arange(0.01, 0.31, 0.01), 2)
    tables = {
        "decision_curve": pd.DataFrame({
            "budget": budgets, "major_caught": np.sqrt(budgets),
            "safety_caught": np.sqrt(budgets) + 0.01, "minutes_caught": 1.2 * budgets,
        }),
        "main_metrics": pd.DataFrame({
            "fold": np.repeat([2021, 2022, 2023], 4), "model": MODELS * 3,
            "pr_auc": np.tile([0.05, 0.13, 0.21, 0.23], 3),
        }),
        "holdout_metrics": pd.DataFrame({
            "period": "2024-2025", "model": MODELS, "pr_auc": [0.05, 0.2, 0.29, 0.31],
            "pr_auc_low": [0.04, 0.17, 0.25, 0.27], "pr_auc_high": [0.06, 0.24, 0.34, 0.36],
        }),
        "holdout_reliability": pd.DataFrame({
            "group": range(1, 11), "mean_predicted": np.linspace(0.01, 0.2, 10),
            "observed_rate": np.linspace(0.01, 0.21, 10),
        }),
        "decision_cost_grid": pd.DataFrame({
            "avoidable_share": np.repeat([0.1, 0.25, 0.5], 3),
            "false_alarm_cost_minutes": [0.5, 1.0, 2.0] * 3,
            "net_minutes_saved_per_year": [78, 49, 0, 321, 233, 138, 694, 641, 465],
            "share_escalated": [0.02, 0.005, 0, 0.05, 0.03, 0.006, 0.14, 0.05, 0.03],
        }),
        "reliability_rates_by_line_year": pd.DataFrame({
            "line": np.repeat(["green", "orange", "blue", "yellow"], 3),
            "year": [2022, 2023, 2024] * 4,
            "km_is_estimated": [False, False, True] * 4,
            "months": [12, 9, 12] * 4,
            "delay_5plus_per_million_km": np.tile([10.0, 12.0, 11.0], 4),
            "delay_5plus_per_million_km_low": np.tile([9.0, 10.5, 9.5], 4),
            "delay_5plus_per_million_km_high": np.tile([11.0, 13.5, 12.5], 4),
        }),
    }
    for name, table in tables.items():
        table.to_csv(tmp_path / f"{name}.csv", index=False)
    monkeypatch.setattr(config, "BACKTEST_DIR", tmp_path)
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    return tmp_path


def test_every_figure_is_written(saved_results):
    incidents = clean_incidents(read_raw_incidents())
    paths = figures.make_figures(incidents)
    assert [path.name for path in paths] == [
        "capture_curve.png", "model_ladder.png", "calibration.png",
        "cost_grid.png", "risk_by_symptom.png", "risk_by_location.png",
        "reliability_rates.png",
    ]
    assert all(path.stat().st_size > 10_000 for path in paths)


def test_wilson_interval_contains_the_rate_and_stays_in_0_1():
    low, high = wilson_interval(pd.Series([0, 5, 50]), pd.Series([50, 50, 50]))
    assert (low >= 0).all() and (high <= 1).all()
    assert low[1] < 0.1 < high[1]


def test_major_rate_by_group_by_hand():
    labelled = pd.DataFrame({
        "y_major": [True] * 10 + [False] * 90 + [True] * 1 + [False] * 9,
        "group": ["a"] * 100 + ["b"] * 10,
    })
    rates = major_rate_by(labelled, labelled["group"], {"a": "Group A", "b": "Group B"})
    # Group B has fewer than MIN_GROUP_SIZE incidents, so it is left out.
    assert list(rates.index) == ["Group A"]
    assert rates.loc["Group A", "rate"] == pytest.approx(0.10)
    assert rates.attrs["overall_rate"] == pytest.approx(11 / 110)
