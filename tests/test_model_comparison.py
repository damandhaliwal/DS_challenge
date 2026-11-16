"""Tests for modeling/model_comparison.py, on a reduced setup (one fold, small bootstrap)."""

import pytest

from cleaning.clean_incidents import clean_incidents
from cleaning.load import read_raw_incidents
from modeling import backtest, model_comparison
from modeling.target import add_targets, model_population


@pytest.fixture(scope="module")
def reduced():
    labelled = add_targets(model_population(clean_incidents(read_raw_incidents())))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backtest, "FOLDS", {2021: (2019, 2020)})
        patch.setattr(model_comparison, "FOLDS", {2021: (2019, 2020)})
        patch.setattr(backtest, "BOOTSTRAP_SAMPLES", 50)
        return model_comparison.run_model_comparison(labelled)


def test_one_summary_row_per_compared_model(reduced):
    assert list(reduced["summary"]["model"]) == model_comparison.COMPARED_MODELS


def test_every_pair_is_compared(reduced):
    expected = [f"{better} - {simpler}" for better, simpler in model_comparison.PAIRS]
    assert list(reduced["comparisons"]["comparison"]) == expected


def test_the_chosen_model_is_the_reference(reduced):
    chosen = reduced["summary"].set_index("model").loc[model_comparison.CHOSEN_MODEL]
    assert chosen["pr_auc_difference_vs_chosen"] == 0


def test_calibration_error_is_reported_for_every_model(reduced):
    assert reduced["summary"]["calibration_error"].between(0, 1).all()
