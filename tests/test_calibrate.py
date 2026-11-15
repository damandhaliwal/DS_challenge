"""Tests for modeling/calibrate.py."""

import numpy as np
import pytest

from cleaning.clean_incidents import clean_incidents
from cleaning.load import read_raw_incidents
from modeling import backtest, calibrate
from modeling.calibrate import (
    CALIBRATION_METHOD,
    CALIBRATION_METHODS,
    IsotonicCalibrator,
    NoCalibration,
    PlattCalibrator,
    reliability_table,
)
from modeling.target import add_targets, model_population


@pytest.fixture
def inflated():
    # True probabilities between 1% and 30%; the scores double them, as a 2x weight on
    # positives roughly does.
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.01, 0.30, size=20_000)
    outcome = (rng.uniform(size=truth.size) < truth).astype(int)
    return np.clip(2 * truth, 0, 0.99), outcome


@pytest.mark.parametrize("calibrator", [PlattCalibrator, IsotonicCalibrator])
def test_calibration_removes_the_inflation(inflated, calibrator):
    scores, outcome = inflated
    probability = calibrator().fit(scores, outcome).transform(scores)
    assert scores.mean() > 1.8 * outcome.mean()
    assert probability.mean() == pytest.approx(outcome.mean(), abs=0.005)


@pytest.mark.parametrize("calibrator", [PlattCalibrator, IsotonicCalibrator])
def test_calibration_keeps_the_ranking(inflated, calibrator):
    scores, outcome = inflated
    probability = calibrator().fit(scores, outcome).transform(np.sort(scores))
    assert (np.diff(probability) >= 0).all()


def test_no_calibration_returns_the_score(inflated):
    scores, outcome = inflated
    assert np.array_equal(NoCalibration().fit(scores, outcome).transform(scores), scores)


def test_reliability_table_uses_ten_equal_groups(inflated):
    scores, outcome = inflated
    table = reliability_table(scores, outcome)
    assert list(table["group"]) == list(range(1, 11))
    assert (table["incidents"] == len(scores) // 10).all()
    assert table["mean_predicted"].is_monotonic_increasing


def test_the_chosen_method_is_one_of_the_compared_ones():
    assert CALIBRATION_METHOD in CALIBRATION_METHODS


def test_reduced_calibration_backtest_never_sees_the_test_year():
    labelled = add_targets(model_population(clean_incidents(read_raw_incidents())))
    training_years = []
    real_fit = calibrate.fit_model

    def recording_fit(model, X, y, weight):
        training_years.append(set(labelled.loc[X.index, "service_date"].dt.year))
        return real_fit(model, X, y, weight)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(calibrate, "FOLDS", {2021: (2019, 2020)})
        patch.setattr(calibrate, "fit_model", recording_fit)
        patch.setattr(backtest, "fit_model", recording_fit)
        results = calibrate.run_calibration_backtest(labelled, model_name="symptom_line_table")

    assert all(2021 not in years for years in training_years)
    # The calibrator's model is trained before the validation year 2020.
    assert {2019} in training_years
    assert list(results["metrics"]["method"]) == CALIBRATION_METHODS
    platt = results["metrics"].set_index("method").loc["platt"]
    assert platt["brier"] < platt["brier_reference"]
