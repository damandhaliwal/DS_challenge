"""Tests for modeling/holdout.py.

The real holdout (2024-2025) is run once by main.py. These tests run the same code on
a stand-in split (train 2019-2020, test 2021-2022) with the two fastest models, so a
bug can be found without looking at the holdout years.
"""

import pytest

from cleaning.clean_incidents import clean_incidents
from cleaning.load import read_raw_incidents
from modeling import backtest, holdout
from modeling.target import add_targets, model_population


@pytest.fixture(scope="module")
def stand_in():
    labelled = add_targets(model_population(clean_incidents(read_raw_incidents())))
    training_years = []
    real_fit = holdout.fit_model

    def recording_fit(model, X, y, weight):
        training_years.append(set(labelled.loc[X.index, "service_date"].dt.year))
        return real_fit(model, X, y, weight)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(holdout, "TRAIN_YEARS", (2019, 2020))
        patch.setattr(holdout, "TEST_YEARS", (2021, 2022))
        patch.setattr(holdout, "HOLDOUT_LABEL", "2021-2022")
        patch.setattr(holdout, "MODEL_NAMES", ["base_rate", "symptom_line_table"])
        patch.setattr(holdout, "CHOSEN_MODEL", "symptom_line_table")
        patch.setattr(holdout, "fit_model", recording_fit)
        patch.setattr(backtest, "fit_model", recording_fit)
        patch.setattr(backtest, "BOOTSTRAP_SAMPLES", 50)
        results = holdout.run_holdout(labelled)
    return labelled, results, training_years


def test_no_model_trains_on_a_test_year(stand_in):
    _, _, training_years = stand_in
    assert training_years
    assert all(not years & {2021, 2022} for years in training_years)


def test_calibration_model_is_trained_before_the_last_training_year(stand_in):
    _, _, training_years = stand_in
    assert {2019} in training_years


def test_results_for_the_whole_holdout_and_each_year(stand_in):
    _, results, _ = stand_in
    assert set(results["metrics"]["period"]) == {"2021-2022", "2021", "2022"}
    assert list(results["calibration"]["period"]) == ["2021-2022", "2021", "2022"]


def test_predictions_cover_every_test_incident_once_per_model(stand_in):
    labelled, results, _ = stand_in
    test_ids = set(labelled.loc[labelled["service_date"].dt.year.between(2021, 2022), "incident_id"])
    for _, group in results["predictions"].groupby("model"):
        assert set(group["incident_id"]) == test_ids
        assert group["incident_id"].is_unique


def test_only_the_chosen_model_gets_a_probability(stand_in):
    _, results, _ = stand_in
    predictions = results["predictions"]
    chosen = predictions["model"] == "symptom_line_table"
    assert predictions.loc[chosen, "probability"].between(0, 1).all()
    assert predictions.loc[~chosen, "probability"].isna().all()


def test_each_model_is_compared_with_the_one_below_it(stand_in):
    _, results, _ = stand_in
    assert list(results["comparisons"]["comparison"]) == ["symptom_line_table - base_rate"]
