"""Tests for cleaning/clean_incidents.py, one group per cleaning step."""

import pandas as pd
import pytest

import config
from cleaning.clean_incidents import (
    BETWEEN_STATIONS,
    DELAY_BUCKET_FLOORS,
    FACILITY,
    FLAG_COLUMNS,
    NO_EVACUATION,
    PLACEHOLDERS,
    RAW_DATE_TIME_COLUMNS,
    STATION,
    TRANSFER_PLATFORM,
    build_timestamps,
    clean_incidents,
    classify_locations,
    convert_flags,
    normalise_lines,
    order_delay_buckets,
    replace_placeholders,
)
from cleaning.load import read_raw_incidents


@pytest.fixture(scope="module")
def raw():
    return read_raw_incidents()


# ---------------------------------------------------------------------------
# Step 1: placeholders
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def step1(raw):
    return replace_placeholders(raw)


def test_no_placeholder_text_is_left(step1):
    for column, placeholder in PLACEHOLDERS.items():
        assert not (step1[column] == placeholder).any(), column
    assert not (step1["evacuation"] == "#").any()


def test_every_placeholder_becomes_exactly_one_missing_value(raw, step1):
    # Missing values appear where the placeholders were, and nowhere else.
    for column, placeholder in PLACEHOLDERS.items():
        assert step1[column].isna().equals(raw[column] == placeholder), column


def test_known_placeholder_counts(step1):
    # Counts from notebook 01; a change here means the raw file or the rule changed.
    assert step1["primary_cause"].isna().sum() == 482
    assert step1["symptom"].isna().sum() == 377
    assert step1["line"].isna().sum() == 103


def test_train_fields_are_always_missing_on_station_incidents(step1):
    station = step1["incident_type"] == "S"
    for column in ["run_number", "car_number", "door_numbers", "rolling_stock"]:
        assert step1.loc[station, column].isna().all(), column


def test_no_evacuation_is_a_category_not_a_missing_value(step1):
    assert step1["evacuation"].notna().all()
    assert (step1["evacuation"] == NO_EVACUATION).sum() == 44_190


def test_other_columns_are_untouched(raw, step1):
    changed = set(PLACEHOLDERS) | {"evacuation"}
    unchanged = [column for column in raw.columns if column not in changed]
    pd.testing.assert_frame_equal(raw[unchanged], step1[unchanged])


def test_input_is_not_modified(raw):
    before = raw.copy()
    replace_placeholders(raw)
    pd.testing.assert_frame_equal(raw, before)


# ---------------------------------------------------------------------------
# Step 2: dates and times
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def step2(raw):
    return build_timestamps(raw)


def test_raw_date_and_time_columns_are_replaced(step2):
    assert not set(RAW_DATE_TIME_COLUMNS) & set(step2.columns)
    for column in ["service_date", "start_at", "end_at"]:
        assert pd.api.types.is_datetime64_dtype(step2[column]), column
        assert step2[column].notna().all(), column


def test_service_date_matches_the_incident_id(step2):
    id_date = pd.to_datetime(step2["incident_id"].str[1:7], format="%d%m%y")
    assert (step2["service_date"] == id_date).all()


def test_start_falls_on_the_real_calendar_date(raw, step2):
    calendar_date = pd.to_datetime(raw["calendar_date"])
    assert (step2["start_at"].dt.normalize() == calendar_date).all()


def test_after_midnight_start_moves_to_the_next_day(raw, step2):
    row = raw.index[raw["start_time"] == "24:30:00"][0]
    service_date = step2.loc[row, "service_date"]
    assert step2.loc[row, "start_at"] == service_date + pd.Timedelta(hours=24, minutes=30)


def test_real_clock_night_rows_read_as_26_00(raw, step2):
    # The 60 rows written "02:00" on the next calendar date (notebook 01).
    next_calendar_day = pd.to_datetime(raw["calendar_date"]) > step2["service_date"]
    night = (raw["start_time"] == "02:00") & next_calendar_day
    assert night.sum() == 60
    expected = step2.loc[night, "service_date"] + pd.Timedelta(hours=26)
    assert (step2.loc[night, "start_at"] == expected).all()


def test_end_is_never_before_start(step2):
    assert (step2["end_at"] >= step2["start_at"]).all()
    assert (step2["recorded_minutes"] >= 0).all()


def test_broken_assumption_fails_loudly(raw):
    tampered = raw.copy()
    tampered.loc[0, "incident_id"] = "S31129901"   # ID date 1999-12-31
    with pytest.raises(ValueError, match="ID date differs"):
        build_timestamps(tampered)


# ---------------------------------------------------------------------------
# Step 3: lines
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def step3(raw):
    return raw.pipe(replace_placeholders).pipe(normalise_lines)


def test_single_lines_use_their_colour(raw, step3):
    assert (step3.loc[raw["line"] == "Ligne verte", "line"] == "green").all()
    assert (step3.loc[raw["line"] == "Ligne bleue", "line"] == "blue").all()


def test_reordered_spellings_become_one_value(raw, step3):
    # "Ligne 1, 2" and "Ligne 2, 1" are the same two lines.
    same_pair = raw["line"].isin(["Ligne 1, 2", "Ligne 2, 1"])
    assert (step3.loc[same_pair, "line"] == "green+orange").all()
    assert same_pair.sum() == 164


def test_excel_date_artefact_is_lines_2_and_4(raw, step3):
    assert step3.loc[raw["line"] == "02-avr", "line"].tolist() == ["orange+yellow"]


def test_line_count(step3):
    assert step3["n_lines"].isna().equals(step3["line"].isna())
    assert (step3["n_lines"] == step3["line"].str.count(r"\+") + 1).all()
    # 378 raw "Ligne a, b" values plus the "02-avr" row.
    assert (step3["n_lines"] > 1).sum() == 379


def test_unknown_line_stays_missing(step3):
    assert step3["line"].isna().sum() == 103


def test_every_value_is_a_known_line_set(step3):
    names = set(config.LINE_NAMES.values())
    parts = step3["line"].dropna().str.split("+").explode()
    assert set(parts) <= names


def test_new_line_spelling_fails_loudly():
    with pytest.raises(ValueError, match="Unknown line value"):
        normalise_lines(pd.DataFrame({"line": ["Ligne 3"]}))


# ---------------------------------------------------------------------------
# Step 4: locations
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def step4(raw):
    return classify_locations(raw)


def classified(step4, raw, location):
    row = step4.loc[raw["location"] == location].iloc[0]
    return row["location_kind"], row["station_from"], row["station_to"]


def test_one_example_of_each_kind(raw, step4):
    assert classified(step4, raw, "Atwater") == (STATION, "Atwater", "Atwater")
    assert classified(step4, raw, "Berri-UQAM - Ligne 2, 1") == (
        TRANSFER_PLATFORM, "Berri-UQAM", "Berri-UQAM"
    )
    assert classified(step4, raw, "Atwater/Guy-Concordia") == (
        BETWEEN_STATIONS, "Atwater", "Guy-Concordia"
    )
    assert classified(step4, raw, "Cartier - Henri-Bourassa") == (
        BETWEEN_STATIONS, "Cartier", "Henri-Bourassa"
    )
    assert classified(step4, raw, "Ligne 2")[0] == "whole_line_or_network"
    assert classified(step4, raw, "Réseau")[0] == "whole_line_or_network"
    assert classified(step4, raw, "Angrignon PR")[0] == "ancillary_structure"
    assert classified(step4, raw, "Garage Angrignon")[0] == FACILITY


def test_capital_letter_codes_with_a_slash_are_facilities_not_segments(raw, step4):
    assert classified(step4, raw, "INTER.ST-MICHEL/PIE-IX PL5")[0] == FACILITY


def test_aliases_use_the_station_spelling(raw, step4):
    assert classified(step4, raw, "Berri L-2")[1:] == ("Berri-UQAM", "Berri-UQAM")
    assert classified(step4, raw, "l'Assomption/Cadillac")[1] == "L'Assomption"
    assert classified(step4, raw, "Vendôme/St-Henri")[2] == "Place St-Henri"


def test_the_network_has_68_stations(step4):
    # The STM metro has 68 stations; transfer stations are counted once.
    assert step4.loc[step4["location_kind"] == STATION, "station_from"].nunique() == 68


def test_every_named_station_is_a_real_station(step4):
    stations = set(step4.loc[step4["location_kind"] == STATION, "station_from"])
    named = set(step4["station_from"].dropna()) | set(step4["station_to"].dropna())
    assert named == stations


def test_both_ends_are_the_same_station_except_on_segments(step4):
    same_end = step4["station_from"] == step4["station_to"]
    assert same_end.equals(step4["location_kind"].isin([STATION, TRANSFER_PLATFORM]))


def test_stations_are_missing_exactly_where_there_is_no_station(step4):
    at_a_station = step4["location_kind"].isin([STATION, TRANSFER_PLATFORM, BETWEEN_STATIONS])
    for column in ["station_from", "station_to"]:
        assert step4[column].notna().equals(at_a_station), column


def test_every_location_gets_a_kind_except_the_unexplained_code(raw, step4):
    assert raw.loc[step4["location_kind"].isna(), "location"].tolist() == ["BR42"]


# ---------------------------------------------------------------------------
# Step 5: flags
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def step5(raw):
    return raw.pipe(replace_placeholders).pipe(convert_flags)


def test_flags_are_booleans(step5):
    for column in FLAG_COLUMNS:
        assert pd.api.types.is_bool_dtype(step5[column]), column


def test_flags_keep_their_raw_meaning(raw, step5):
    for column in [c for c in FLAG_COLUMNS if c != "door_involved"]:
        assert step5[column].equals(raw[column] == "1"), column


def test_door_involved_when_either_source_says_so(raw, step5):
    either = (raw["door_involved"] == "1") | (raw["door_numbers"] != "#")
    assert step5["door_involved"].equals(either)
    # 1,576 both + 1,335 flag only + 1,219 door numbers only (notebook 01).
    assert step5["door_involved"].sum() == 4_130


def test_station_incidents_never_involve_a_door(step5):
    assert not step5.loc[step5["incident_type"] == "S", "door_involved"].any()


def test_unexpected_flag_value_fails_loudly(raw):
    tampered = replace_placeholders(raw)
    tampered.loc[0, "emergency_brake"] = "2"
    with pytest.raises(ValueError, match="emergency_brake"):
        convert_flags(tampered)


# ---------------------------------------------------------------------------
# Step 6: delay bucket
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def step6(raw):
    return order_delay_buckets(raw)


def test_buckets_sort_by_length_not_alphabetically(step6):
    bucket = step6["delay_bucket"]
    assert bucket.cat.ordered
    assert list(bucket.cat.categories) == list(DELAY_BUCKET_FLOORS)
    assert (bucket >= "05 à 09 min").sum() == (step6["delay_floor_minutes"] >= 5).sum()


def test_bucket_text_is_unchanged(raw, step6):
    assert (step6["delay_bucket"].astype(str) == raw["delay_bucket"]).all()


def test_known_threshold_counts(step6):
    # Counts from notebook 03 (all incidents, train and station).
    floor = step6["delay_floor_minutes"]
    assert (floor >= 5).sum() == 6_413
    assert (floor >= 30).sum() == 524


def test_station_incidents_are_all_in_the_shortest_bucket(step6):
    station = step6["incident_type"] == "S"
    assert (step6.loc[station, "delay_floor_minutes"] == 0).all()


def test_unknown_bucket_fails_loudly(raw):
    tampered = raw.copy()
    tampered.loc[0, "delay_bucket"] = "45 min"
    with pytest.raises(ValueError, match="Unknown delay buckets"):
        order_delay_buckets(tampered)


# ---------------------------------------------------------------------------
# Whole pipeline
# ---------------------------------------------------------------------------
def test_full_cleaning_keeps_every_incident_once(raw):
    cleaned = clean_incidents(raw)
    assert len(cleaned) == len(raw)
    assert cleaned["incident_id"].is_unique
