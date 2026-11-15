"""Tests for modeling/sensitivity.py, on a reduced setup (one fold, the lookup table)."""

import pytest

from cleaning.clean_incidents import clean_incidents
from cleaning.load import read_raw_incidents
from modeling import backtest, sensitivity
from modeling.target import add_targets, model_population


@pytest.fixture(scope="module")
def reduced():
    labelled = add_targets(model_population(clean_incidents(read_raw_incidents())))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backtest, "FOLDS", {2021: (2019, 2020)})
        patch.setattr(backtest, "BOOTSTRAP_SAMPLES", 50)
        patch.setattr(sensitivity, "MODEL", "symptom_line_table")
        patch.setattr(sensitivity, "VARIANTS", {
            "main": {},
            "safety_weight_4": {"safety_weight": 4.0},
            "target_5plus": {"target": "y_5plus", "safety_weight": 1.0},
        })
        return sensitivity.run_sensitivity(labelled)


def test_one_summary_row_per_variant(reduced):
    assert list(reduced["summary"]["variant"]) == ["main", "safety_weight_4", "target_5plus"]


def test_main_is_the_reference(reduced):
    main = reduced["summary"].set_index("variant").loc["main"]
    assert main["pr_auc_difference_vs_main"] == 0
    assert "main" not in set(reduced["comparisons"]["variant"])


def test_every_variant_is_judged_on_major_incidents(reduced):
    # Even the 5+ model reports how many major incidents it catches.
    assert reduced["metrics"]["y_major_caught_10%"].notna().all()
    assert reduced["summary"]["pr_auc_y_major"].between(0, 1).all()


def test_the_5plus_model_is_worse_at_finding_major_incidents(reduced):
    target_5plus = reduced["summary"].set_index("variant").loc["target_5plus"]
    assert target_5plus["pr_auc_difference_vs_main"] < 0
