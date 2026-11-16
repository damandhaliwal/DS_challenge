"""Tests for modeling/rescoring.py, on a reduced setup (one fold, small bootstrap).

The holdout test (Decision 11) runs here on a stand-in split (train 2019-2020, test
2021-2022), as in test_holdout.py, so a bug can be found without looking at the
holdout years.
"""

import pytest
from sklearn.metrics import average_precision_score

from cleaning.clean_incidents import clean_incidents
from cleaning.load import read_raw_incidents
from modeling import backtest, holdout, rescoring
from modeling.backtest import run_backtest
from modeling.target import add_targets, model_population


@pytest.fixture(scope="module")
def reduced():
    labelled = add_targets(model_population(clean_incidents(read_raw_incidents())))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backtest, "FOLDS", {2021: (2019, 2020)})
        patch.setattr(rescoring, "FOLDS", {2021: (2019, 2020)})
        patch.setattr(backtest, "BOOTSTRAP_SAMPLES", 50)
        patch.setattr(rescoring, "RETRAINED_MODELS", ["base_rate", "symptom_line_table"])
        patch.setattr(rescoring, "CHOSEN_MODEL", "symptom_line_table")
        first = run_backtest(labelled, model_names=["symptom_line_table"])
        return labelled, rescoring.run_rescoring(labelled, first["predictions"])


def test_each_pool_holds_every_30plus_incident_and_shrinks_with_time(reduced):
    _, results = reduced
    pools = results["pools"].set_index("stage")
    assert (pools["share_of_30plus"] == 1).all()
    assert pools.loc["after_10_minutes", "incidents"] < pools.loc["after_5_minutes", "incidents"]
    assert pools["share_of_incidents"].between(0, 0.5).all()


def test_pool_size_matches_the_delay_bucket(reduced):
    labelled, results = reduced
    in_2021 = labelled[labelled["service_date"].dt.year == 2021]
    pools = results["pools"].set_index("stage")
    assert pools.loc["after_5_minutes", "incidents"] == in_2021["y_5plus"].sum()
    assert pools.loc["after_10_minutes", "incidents"] == in_2021["y_10plus"].sum()


def test_the_first_score_is_judged_on_the_same_pool_as_the_retrained_models(reduced):
    _, results = reduced
    metrics = results["metrics"]
    assert set(metrics["model"]) == {
        "base_rate", "symptom_line_table", rescoring.FIRST_SCORE, rescoring.FIRST_SCORE_30PLUS
    }
    for _, stage in metrics.groupby("stage"):
        assert stage["incidents"].nunique() == 1
        assert stage["base_rate"].nunique() == 1


def test_the_base_rate_finds_30plus_incidents_at_the_pool_rate(reduced):
    _, results = reduced
    base = results["metrics"][results["metrics"]["model"] == "base_rate"]
    assert base["precision_10%"].to_numpy() == pytest.approx(base["base_rate"].to_numpy())


def test_every_retrained_model_is_compared_with_both_first_scores(reduced):
    _, results = reduced
    assert set(results["comparisons"]["comparison"]) == {
        "symptom_line_table - first_score",
        "symptom_line_table - first_score_30plus",
        "first_score_30plus - first_score",
    }
    assert len(results["comparisons"]) == 3 * len(rescoring.STAGES)
    summary = results["summary"].set_index(["stage", "model"])
    for reference in [rescoring.FIRST_SCORE, rescoring.FIRST_SCORE_30PLUS]:
        compared = summary.loc[(slice(None), "symptom_line_table"), f"pr_auc_difference_vs_{reference}"]
        assert compared.notna().all()



STAND_IN = {"TRAIN_YEARS": (2019, 2020), "TEST_YEARS": (2021, 2022), "HOLDOUT_LABEL": "2021-2022",
            "CHOSEN_MODEL": "symptom_line_table"}
FAST_MODELS = ["base_rate", "symptom_line_table"]


@pytest.fixture(scope="module")
def stand_in():
    labelled = add_targets(model_population(clean_incidents(read_raw_incidents())))
    with pytest.MonkeyPatch.context() as patch:
        for module in (holdout, rescoring):
            for name, value in STAND_IN.items():
                patch.setattr(module, name, value)
        patch.setattr(holdout, "MODEL_NAMES", FAST_MODELS)
        patch.setattr(rescoring, "RETRAINED_MODELS", FAST_MODELS)
        patch.setattr(rescoring, "DECIDING_MODEL", "symptom_line_table")
        patch.setattr(backtest, "BOOTSTRAP_SAMPLES", 50)
        first = holdout.run_holdout(labelled)["predictions"]
        results = rescoring.run_rescoring_holdout(labelled, first)

        # The same ranking with every test-year outcome flipped: if no model trains or
        # tunes on a test year, its scores cannot change.
        flipped = labelled.copy()
        in_test = flipped["service_date"].dt.year.between(2021, 2022)
        flipped.loc[in_test, rescoring.TARGET] = ~flipped.loc[in_test, rescoring.TARGET]
        risks = [rescoring._holdout_risk(frame, "symptom_line_table")[0] for frame in (labelled, flipped)]
    return labelled, first, results, risks


def test_holdout_scores_do_not_depend_on_test_year_outcomes(stand_in):
    labelled, _, _, (risk, risk_flipped) = stand_in
    test_ids = labelled.loc[labelled["service_date"].dt.year.between(2021, 2022), "incident_id"]
    assert set(risk.index) == set(test_ids)
    assert risk.to_numpy() == pytest.approx(risk_flipped.to_numpy())


def test_holdout_results_for_the_whole_period_and_each_year(stand_in):
    _, _, results, _ = stand_in
    for table in ["pools", "metrics", "comparisons"]:
        assert set(results[table]["period"]) == {"2021-2022", "2021", "2022"}


def test_holdout_pools_hold_every_30plus_incident(stand_in):
    _, _, results, _ = stand_in
    pools = results["pools"].set_index(["stage", "period"])
    assert (pools["share_of_30plus"] == 1).all()
    assert (pools.loc[rescoring.AT_REPORT, "share_of_incidents"] == 1).all()


def test_the_first_score_is_reused_from_the_holdout_as_it_is(stand_in):
    labelled, first, results, _ = stand_in
    chosen = first[first["model"] == "symptom_line_table"].merge(labelled, on="incident_id")
    metrics = results["metrics"].set_index(["stage", "period", "model"])
    reused = metrics.loc[(rescoring.AT_REPORT, "2021-2022", rescoring.FIRST_SCORE), "pr_auc"]
    assert reused == pytest.approx(average_precision_score(chosen[rescoring.TARGET], chosen["risk"]))


def test_the_second_look_escalates_fewer_incidents_than_at_report(stand_in):
    _, _, results, _ = stand_in
    escalations = results["metrics"].groupby(["stage", "period"])["escalations_a_year"].first()
    for period in ["2021-2022", "2021", "2022"]:
        assert escalations[("after_5_minutes", period)] < escalations[(rescoring.AT_REPORT, period)]


def test_holdout_comparisons_and_the_one_that_decides(stand_in):
    _, _, results, _ = stand_in
    assert len(results["comparisons"]) == 3 * len(rescoring.STAGES) * 3
    decision = results["decision"]
    assert len(decision) == 1
    row = decision.iloc[0]
    assert (row["stage"], row["period"], row["model"], row["reference"]) == (
        "after_5_minutes", "2021-2022", "symptom_line_table", rescoring.FIRST_SCORE_30PLUS
    )
    assert row["reading"] == rescoring.reading(row["low"], row["high"])


@pytest.mark.parametrize("low, high, expected", [
    (0.01, 0.2, "confirmed"), (-0.05, 0.1, "not confirmed"), (-0.2, -0.01, "rejected"),
])
def test_reading_of_the_deciding_interval(low, high, expected):
    assert rescoring.reading(low, high) == expected
