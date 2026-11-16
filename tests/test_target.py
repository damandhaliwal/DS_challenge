"""Tests for modeling/target.py."""

import pytest

from cleaning.clean_incidents import clean_incidents
from cleaning.load import read_raw_incidents
from modeling.target import (
    LONG,
    MINOR,
    MODERATE,
    NO_SAFETY_RESPONSE,
    SAFETY_RESPONSE,
    SAFETY_RESPONSE_KINDS,
    SEVERITY_TIERS,
    VERY_LONG,
    add_targets,
    model_population,
)


@pytest.fixture(scope="module")
def incidents():
    return clean_incidents(read_raw_incidents())


@pytest.fixture(scope="module")
def labelled(incidents):
    return add_targets(model_population(incidents))


def test_population_is_train_incidents_only(incidents, labelled):
    assert (labelled["incident_type"] == "T").all()
    assert len(labelled) == (incidents["incident_type"] == "T").sum() == 29_167


def test_tier_counts_match_notebook_04(labelled):
    counts = labelled["severity_tier"].value_counts()
    assert counts[SAFETY_RESPONSE] == 1_515
    assert counts[VERY_LONG] == 102
    assert counts[LONG] == 1_041
    assert counts[MODERATE] == 4_074
    assert counts[MINOR] == 22_435


def test_every_incident_has_exactly_one_tier(labelled):
    assert labelled["severity_tier"].notna().all()
    assert labelled["severity_tier"].cat.ordered
    assert list(labelled["severity_tier"].cat.categories) == SEVERITY_TIERS


def test_tiers_follow_their_definitions(labelled):
    tier, delay = labelled["severity_tier"], labelled["delay_floor_minutes"]
    assert labelled["y_safety"].equals(tier == SAFETY_RESPONSE)
    assert (delay[tier == VERY_LONG] >= 30).all()
    assert delay[tier == LONG].between(10, 25).all()
    assert delay[tier == MODERATE].eq(5).all()
    assert (delay[tier == MINOR] < 5).all()


def test_tiers_compare_by_severity(labelled):
    # "long or worse" = everything except minor and moderate.
    long_or_worse = labelled["severity_tier"] >= LONG
    assert long_or_worse.sum() == 1_515 + 102 + 1_041


def test_most_very_long_delays_had_a_safety_response(labelled):
    # About 80% of the 524 incidents of 30+ min are in the safety tier (notebook 04).
    severe = labelled[labelled["y_30plus"]]
    assert len(severe) == 524
    assert severe["y_safety"].mean() == pytest.approx(0.805, abs=0.005)


def test_delay_labels_are_nested(labelled):
    for shorter, longer in [(5, 10), (10, 15), (15, 30)]:
        assert not (labelled[f"y_{longer}plus"] & ~labelled[f"y_{shorter}plus"]).any()
    assert labelled["y_5plus"].sum() == 6_413


def test_safety_response_kind_splits_the_safety_tier(labelled):
    kind = labelled["safety_response_kind"]
    assert kind.notna().all()
    assert list(kind.cat.categories) == SAFETY_RESPONSE_KINDS
    assert (kind != NO_SAFETY_RESPONSE).equals(labelled["y_safety"])
    counts = kind.value_counts()
    assert counts["train_evacuated"] == 968
    assert counts["train_and_station_evacuated"] == 293
    assert counts["station_evacuated"] == counts["car_evacuated"] == 55
    # The emergency team with nothing evacuated.
    assert counts["emergency_team_only"] == 144
    assert labelled.loc[kind == "emergency_team_only", "emergency_team"].all()


def test_a_wider_evacuation_is_a_longer_incident(labelled):
    long = labelled.groupby("safety_response_kind", observed=True)["y_30plus"].mean()
    assert long["train_and_station_evacuated"] > 0.7
    assert long["train_evacuated"] < 0.25
    assert long["car_evacuated"] == 0


def test_major_is_safety_or_30_plus(labelled):
    assert labelled["y_major"].equals(labelled["y_safety"] | labelled["y_30plus"])
    assert labelled["y_major"].equals(labelled["severity_tier"] >= VERY_LONG)
    assert labelled["y_major"].sum() == 1_515 + 102
