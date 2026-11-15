"""Tests for modeling/decision.py, on small hand-made examples."""

import numpy as np
import pandas as pd
import pytest

from modeling.decision import budget_table, cost_grid, score_holdout


@pytest.fixture
def scored():
    # 10 incidents over one year (366 days, 2024); the two riskiest are major, and one
    # of them is a safety response.
    dates = pd.date_range("2024-01-01", "2024-12-31", periods=10).normalize()
    return pd.DataFrame({
        "service_date": dates,
        "probability": [0.9, 0.8, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.02, 0.01],
        "y_major": [True, True] + [False] * 8,
        "y_safety": [True] + [False] * 9,
        "delay_minutes": [30.0, 10.0, 5.0, 5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })


def test_budget_table_by_hand(scored):
    row = budget_table(scored, [0.2]).iloc[0]
    assert row["major_caught"] == 1.0                 # both majors are in the top 2
    assert row["safety_caught"] == 1.0
    assert row["minutes_caught"] == pytest.approx(40 / 53)
    assert row["escalations_per_major_found"] == 1.0
    assert row["escalations_per_year"] == pytest.approx(2 * 365.25 / 366)


def test_cost_grid_cut_off_and_net_minutes(scored):
    grid = cost_grid(scored, minutes_per_major=20.0)
    cell = grid[(grid["avoidable_share"] == 0.25) & (grid["false_alarm_cost_minutes"] == 1.0)].iloc[0]
    # cut-off = 1 / (0.25 x 20) = 0.2 -> escalate the four incidents with p >= 0.2
    assert cell["probability_cut_off"] == pytest.approx(0.2)
    assert cell["share_escalated"] == pytest.approx(0.4)
    # saved = 0.25 x (30 + 10); false alarms = 2 x 1 minute
    assert cell["net_minutes_saved_per_year"] == pytest.approx((10 - 2) * 365.25 / 366)


def test_a_cut_off_above_one_escalates_nothing(scored):
    grid = cost_grid(scored, minutes_per_major=5.0)
    expensive = grid[grid["probability_cut_off"] > 1]
    assert len(expensive) > 0
    assert (expensive["share_escalated"] == 0).all()
    assert (expensive["net_minutes_saved_per_year"] == 0).all()


def test_wider_budgets_catch_more(scored):
    table = budget_table(scored, np.round(np.arange(0.1, 1.01, 0.1), 1))
    assert table["major_caught"].is_monotonic_increasing
    assert table["probability_cut_off"].is_monotonic_decreasing


def test_unmatched_predictions_fail_loudly():
    labelled = pd.DataFrame({"incident_id": ["T1"], "delay_floor_minutes": [5]})
    predictions = pd.DataFrame({"incident_id": ["T1", "T2"], "probability": [0.1, 0.2]})
    with pytest.raises(ValueError, match="do not match"):
        score_holdout(labelled, predictions)
