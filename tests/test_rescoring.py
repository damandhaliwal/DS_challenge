"""Tests for modeling/rescoring.py, on a reduced setup (one fold, small bootstrap)."""

import pytest

from cleaning.clean_incidents import clean_incidents
from cleaning.load import read_raw_incidents
from modeling import backtest, rescoring
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
