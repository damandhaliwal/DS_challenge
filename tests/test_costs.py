"""Tests for modeling/costs.py."""

import pytest

from cleaning.clean_incidents import clean_incidents
from cleaning.load import read_raw_incidents
from modeling.costs import (
    ALERT_BUDGETS,
    SAFETY_WEIGHT,
    SAFETY_WEIGHT_GRID,
    delay_minutes,
    sample_weights,
)
from modeling.target import add_targets, model_population


@pytest.fixture(scope="module")
def labelled():
    return add_targets(model_population(clean_incidents(read_raw_incidents())))


def test_the_main_settings_are_part_of_their_grids():
    assert SAFETY_WEIGHT in SAFETY_WEIGHT_GRID
    assert all(0 < budget < 1 for budget in ALERT_BUDGETS)


def test_safety_incidents_get_the_safety_weight(labelled):
    weights = sample_weights(labelled)
    assert (weights[labelled["y_safety"]] == SAFETY_WEIGHT).all()
    assert (weights[~labelled["y_safety"]] == 1.0).all()


def test_a_weight_of_one_treats_every_incident_the_same(labelled):
    assert (sample_weights(labelled, safety_weight=1.0) == 1.0).all()


def test_delay_minutes_are_the_cleaned_delay(labelled):
    minutes = delay_minutes(labelled)
    assert minutes.equals(labelled["delay_minutes"].astype(float))
    # 30+ incidents now count their recorded delay, not a flat 30.
    assert minutes[labelled["y_30plus"]].mean() > 30
