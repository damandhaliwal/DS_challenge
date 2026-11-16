"""Tests for modeling/escalation_mix.py, on small hand-made examples."""

import numpy as np
import pandas as pd
import pytest

from cleaning.clean_incidents import clean_incidents
from cleaning.load import read_raw_incidents
from modeling.backtest import capture
from modeling.escalation_mix import OUTCOME_KINDS, escalated_share, mix_by, outcome_kind, score
from modeling.holdout import CHOSEN_MODEL
from modeling.target import add_targets, model_population


# ---------------------------------------------------------------------------
# escalated_share, checked by hand and against capture
# ---------------------------------------------------------------------------
def test_escalated_share_without_ties():
    assert list(escalated_share(pd.Series([3, 2, 1, 0]), 0.5)) == [1, 1, 0, 0]


def test_escalated_share_splits_a_tie_at_the_cut_off():
    # Two slots: the 2-risk incident takes one, the two tied incidents share the other.
    assert list(escalated_share(pd.Series([2, 1, 1, 0]), 0.5)) == [1, 0.5, 0.5, 0]


def test_escalated_share_agrees_with_capture():
    rng = np.random.default_rng(0)
    risk = pd.Series(rng.integers(0, 20, size=1_000) / 20)     # many tied scores
    major = pd.Series(rng.random(1_000) < 0.1)
    share = escalated_share(risk, 0.1)
    assert share.sum() == pytest.approx(100)
    assert (share * major).sum() / major.sum() == pytest.approx(capture(risk, major, 0.1))


# ---------------------------------------------------------------------------
# mix_by, by hand
# ---------------------------------------------------------------------------
@pytest.fixture
def scored():
    # 10 incidents, 2 escalated (10% budget would be 1; the shares are given directly).
    return pd.DataFrame({
        "period": "test",
        "incident_id": range(10),
        "symptom": ["fire"] * 2 + ["customer"] * 8,
        "y_major": [True, False] + [True, True] + [False] * 6,
        "y_5plus": [True, True] + [True] + [False] * 7,
        "y_30plus": [True] + [False] * 9,
        "delay_minutes": [40.0, 6.0, 5.0] + [0.0] * 7,
        "escalated": [1.0, 1.0] + [0.0] * 8,
    })


def test_mix_by_group_by_hand(scored):
    mix = mix_by(scored, "symptom").set_index("symptom")
    # The row for all incidents comes first; then customer, which has the most major
    # incidents (2 of 3), none of them escalated.
    assert list(mix.index) == ["all", "customer", "fire"]
    assert mix.loc["all", "incidents"] == 10
    assert mix.loc["all", "share_escalated"] == pytest.approx(0.2)
    assert mix.loc["all", "major_caught"] == pytest.approx(1 / 3)
    assert mix.loc["customer", "share_of_major"] == pytest.approx(2 / 3)
    assert mix.loc["customer", "major_caught"] == 0
    assert mix.loc["customer", "share_escalated"] == 0
    assert pd.isna(mix.loc["customer", "delay_30plus_caught"])     # no 30+ incident there
    assert mix.loc["fire", "major_caught"] == 1
    assert mix.loc["fire", "delay_30plus_caught"] == 1
    assert mix.loc["fire", "share_of_escalations"] == 1
    assert mix.loc["fire", "mean_delay_minutes"] == pytest.approx(23.0)
    assert mix.loc["fire", "share_30plus"] == 0.5
    assert mix.loc["customer", "share_under_5"] == pytest.approx(7 / 8)


def test_ordered_groups_keep_their_order(scored):
    kinds = pd.Categorical(["minor"] * 2 + ["serious"] * 8, categories=["serious", "minor"])
    mix = mix_by(scored.assign(outcome=kinds), "outcome")
    assert list(mix["outcome"]) == ["all", "serious", "minor"]


# ---------------------------------------------------------------------------
# On the real incidents
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def labelled():
    return add_targets(model_population(clean_incidents(read_raw_incidents())))


def test_every_incident_has_one_outcome_kind(labelled):
    outcome = outcome_kind(labelled)
    assert outcome.notna().all()
    assert list(outcome.cat.categories) == OUTCOME_KINDS
    # The safety kinds together are the safety tier; the rest follow the delay tiers.
    assert outcome.isin(OUTCOME_KINDS[:5]).equals(labelled["y_safety"])
    assert (outcome == "very_long_30plus").sum() == 102


def test_each_fold_escalates_its_own_riskiest_incidents(labelled):
    rng = np.random.default_rng(0)
    test = labelled[labelled["service_date"].dt.year.isin([2021, 2022])]
    predictions = pd.DataFrame({
        "fold": test["service_date"].dt.year, "model": CHOSEN_MODEL,
        "incident_id": test["incident_id"], "risk": rng.random(len(test)),
    })
    scored = score(labelled, predictions, "backtest")
    by_fold = scored.groupby("fold")
    assert (by_fold["escalated"].sum() / by_fold.size()).to_numpy() == pytest.approx([0.1, 0.1])


def test_unmatched_predictions_fail_loudly(labelled):
    predictions = pd.DataFrame({"fold": 2021, "model": CHOSEN_MODEL,
                                "incident_id": ["not an incident"], "risk": [0.5]})
    with pytest.raises(ValueError, match="do not match"):
        score(labelled, predictions, "backtest")
