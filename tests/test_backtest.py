"""Tests for modeling/backtest.py.

The full backtest takes about 40 seconds, so these tests run a reduced version: one
fold, the two fastest models and a small bootstrap.
"""

import numpy as np
import pandas as pd
import pytest

from cleaning.clean_incidents import clean_incidents
from cleaning.load import read_raw_incidents
from modeling import backtest
from modeling.backtest import capture, compare_models, run_backtest
from modeling.target import add_targets, model_population

FAST_MODELS = ["base_rate", "symptom_line_table"]


# ---------------------------------------------------------------------------
# capture, checked by hand
# ---------------------------------------------------------------------------
def test_capture_without_ties():
    # Top half = the two riskiest incidents; they hold 1 of the 2 positives.
    assert capture(pd.Series([3, 2, 1, 0]), pd.Series([1, 0, 1, 0]), 0.5) == 0.5


def test_capture_splits_a_tie_at_the_cut_off():
    # Top half: the 2-risk incident (value 0) plus one of the two tied 1-risk
    # incidents, i.e. half of their combined value 1.
    assert capture(pd.Series([2, 1, 1, 0]), pd.Series([0, 1, 0, 0]), 0.5) == 0.5


def test_capture_of_a_constant_score_is_the_budget():
    values = pd.Series(np.random.default_rng(0).integers(0, 2, size=1_000))
    assert capture(pd.Series(np.zeros(1_000)), values, 0.1) == pytest.approx(0.1)


def test_capture_works_with_delay_minutes():
    assert capture(pd.Series([2, 1, 0]), pd.Series([30.0, 10.0, 0.0]), 1 / 3) == 0.75


# ---------------------------------------------------------------------------
# A reduced backtest
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def labelled():
    return add_targets(model_population(clean_incidents(read_raw_incidents())))


@pytest.fixture(scope="module")
def reduced(labelled):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backtest, "FOLDS", {2021: (2019, 2020)})
        patch.setattr(backtest, "BOOTSTRAP_SAMPLES", 50)
        training_years = []
        real_fit = backtest.fit_model

        def recording_fit(model, X, y, weight):
            training_years.append(set(labelled.loc[X.index, "service_date"].dt.year))
            return real_fit(model, X, y, weight)

        patch.setattr(backtest, "fit_model", recording_fit)
        results = run_backtest(labelled, model_names=FAST_MODELS)
        comparison = compare_models(results, labelled, "symptom_line_table", "symptom_line_table")
    return results, training_years, comparison


def test_no_model_ever_trains_on_the_test_year(reduced):
    _, training_years, _ = reduced
    assert training_years                                   # tuning and final fits ran
    assert all(2021 not in years for years in training_years)


def test_predictions_cover_exactly_the_test_year(reduced, labelled):
    results, _, _ = reduced
    test_ids = set(labelled.loc[labelled["service_date"].dt.year == 2021, "incident_id"])
    for _, group in results["predictions"].groupby("model"):
        assert set(group["incident_id"]) == test_ids


def test_one_metrics_row_per_fold_and_model(reduced):
    results, _, _ = reduced
    assert len(results["metrics"]) == len(FAST_MODELS)
    assert len(results["settings"]) == len(FAST_MODELS)


def test_base_rate_catches_exactly_its_budget(reduced):
    results, _, _ = reduced
    base = results["metrics"].set_index("model").loc["base_rate"]
    assert base["y_major_caught_10%"] == pytest.approx(0.10)
    assert base["pr_auc"] == pytest.approx(base["base_rate"])


def test_bootstrap_interval_contains_the_estimate(reduced):
    results, _, _ = reduced
    table = results["metrics"].set_index("model").loc["symptom_line_table"]
    assert table["pr_auc_low"] <= table["pr_auc"] <= table["pr_auc_high"]


def test_a_model_compared_with_itself_differs_by_zero(reduced):
    _, _, comparison = reduced
    assert (comparison["pr_auc_difference"] == 0).all()
    assert (comparison[["low", "high"]] == 0).all().all()
