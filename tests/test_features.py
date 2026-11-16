"""Tests for modeling/features.py."""

import pandas as pd
import pytest

import config
from cleaning.clean_incidents import clean_incidents
from cleaning.load import read_raw_incidents
from modeling.features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    LEAK_SUSPECT_FEATURES,
    MAIN_FEATURES,
    NO_SIGNAL_FEATURES,
    TIMING_ASSUMED_FEATURES,
    UNSPECIFIED_LOCATION,
    build_features,
)
from modeling.target import add_targets, model_population

# Columns that are only known after the incident; none may ever be an input.
OUTCOME_COLUMNS = {
    "primary_cause", "secondary_cause", "rolling_stock", "end_at", "recorded_minutes", "delay_minutes",
    "delay_bucket", "delay_floor_minutes", "possible_damage", "emergency_team",
    "traction_power_cut", "evacuation", "severity_tier", "safety_response_kind",
}


@pytest.fixture(scope="module")
def incidents():
    return add_targets(model_population(clean_incidents(read_raw_incidents())))


@pytest.fixture(scope="module")
def features(incidents):
    return build_features(incidents)


def test_one_row_per_incident_and_no_missing_values(incidents, features):
    assert features.index.equals(incidents.index)
    assert features.notna().all().all()


def test_columns_and_types(features):
    assert list(features.columns) == MAIN_FEATURES + NO_SIGNAL_FEATURES + LEAK_SUSPECT_FEATURES
    for column in CATEGORICAL_FEATURES:
        assert isinstance(features[column].dtype, pd.CategoricalDtype), column
    for column in BOOLEAN_FEATURES + NO_SIGNAL_FEATURES + LEAK_SUSPECT_FEATURES:
        assert pd.api.types.is_bool_dtype(features[column]), column


def test_no_outcome_and_no_label_is_an_input(features):
    names = set(features.columns)
    assert not names & OUTCOME_COLUMNS
    assert not any(name.startswith("y_") for name in names)


def test_leak_suspects_and_no_signal_inputs_are_not_main_inputs():
    assert not set(LEAK_SUSPECT_FEATURES) & set(MAIN_FEATURES)
    assert not set(NO_SIGNAL_FEATURES) & set(MAIN_FEATURES)


def test_inputs_with_an_assumed_recording_time_are_main_inputs():
    # The sensitivity run "without_flags" drops exactly these from the main model.
    assert set(TIMING_ASSUMED_FEATURES) <= set(MAIN_FEATURES)


def test_whole_line_location_is_hidden_from_the_main_location_kind(incidents, features):
    whole_line = incidents["location_kind"] == "whole_line_or_network"
    assert features["whole_line_location"].equals(whole_line)
    assert (features.loc[whole_line, "location_kind"] == UNSPECIFIED_LOCATION).all()
    assert "whole_line_or_network" not in features["location_kind"].cat.categories


def test_line_group(incidents, features):
    assert (features.loc[incidents["n_lines"] > 1, "line_group"] == "multi_line").all()
    assert (features.loc[incidents["line"].isna(), "line_group"] == "unknown").all()
    assert (features.loc[incidents["line"] == "green", "line_group"] == "green").all()


def test_network_stations_are_spelled_as_in_the_data(incidents):
    stations = set(incidents["station_from"].dropna())
    assert config.TERMINAL_STATIONS <= stations
    assert config.TRANSFER_STATIONS <= stations


def test_a_segment_next_to_a_terminal_counts_as_at_a_terminal(incidents, features):
    row = incidents.index[incidents["location"] == "Angrignon/Monk"][0]
    assert features.loc[row, "at_terminal"]
    assert not features.loc[row, "at_transfer_station"]


def test_time_of_day_uses_the_service_day_clock(incidents, features):
    # 00:30 the next calendar day is 24:30 on the service clock: "late", not "early".
    after_midnight = incidents["start_at"].dt.normalize() > incidents["service_date"]
    assert (features.loc[after_midnight, "time_of_day"] == "late").all()


def test_unseen_symptom_fails_loudly(incidents):
    tampered = incidents.head(3).copy()
    tampered.loc[tampered.index[0], "symptom"] = "Nouveau symptôme"
    with pytest.raises(ValueError, match="Unexpected values"):
        build_features(tampered)
