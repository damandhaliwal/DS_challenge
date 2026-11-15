"""Tests for cleaning/clean_km.py."""

import pandas as pd
import pytest

import config
from cleaning.clean_km import DAY_TYPES, clean_km, collapse_to_line_days
from cleaning.load import read_raw_km


@pytest.fixture(scope="module")
def raw():
    return read_raw_km()


@pytest.fixture(scope="module")
def cleaned(raw):
    return clean_km(raw)


def test_one_row_per_line_and_day(cleaned):
    # 1,734 days (2019-01-01 to 2023-09-30) x 4 lines.
    assert len(cleaned) == 1_734 * 4
    assert not cleaned.duplicated(["line", "service_date"]).any()
    assert set(cleaned["line"]) == set(config.LINE_NAMES.values())


def test_no_km_is_lost_or_double_counted(raw, cleaned):
    raw_total = raw["planned_km"].str.replace(",", ".").astype(float).sum()
    assert cleaned["planned_km"].sum() == pytest.approx(raw_total)


def test_every_line_day_has_planned_km(cleaned):
    assert (cleaned["planned_km"] > 0).all()


def test_decimal_comma_is_read_as_a_decimal(raw, cleaned):
    # Blue line, 2019-01-05: raw "16315,92" on the regular row.
    row = cleaned[(cleaned["line"] == "blue") & (cleaned["service_date"] == "2019-01-05")]
    assert row["planned_km"].item() == pytest.approx(16_315.92)


def test_km_on_the_placeholder_row_is_kept(raw, cleaned):
    # For period 19U the km sits on the placeholder row, the regular row has 0.
    regular = raw[raw["service_period"] == "19U"].iloc[0]
    assert regular["planned_km"] == "0"
    row = cleaned[
        (cleaned["line"] == config.LINE_NAMES[config.FRENCH_LINE_NUMBERS[regular["line"]]])
        & (cleaned["service_date"] == regular["calendar_date"])
    ]
    assert row["service_period"].item() == "19U"
    assert row["planned_km"].item() > 0


def test_no_placeholder_is_left(cleaned):
    assert (cleaned["service_period"] != "#").all()
    assert cleaned["day_type"].isin(list(DAY_TYPES.values())).all()


def test_holiday_flag(cleaned):
    assert cleaned["is_holiday"].equals(cleaned["day_type"].str.startswith("holiday"))


def test_day_type_agrees_with_the_calendar(cleaned):
    # Saturday and Sunday service only ever run on those days (holidays aside).
    weekday = cleaned["service_date"].dt.dayofweek    # Monday = 0
    assert (weekday[cleaned["day_type"] == "saturday"] == 5).all()
    assert (weekday[cleaned["day_type"] == "sunday"] == 6).all()
    assert (weekday[cleaned["day_type"] == "weekday"] < 5).all()


def test_two_rows_with_km_on_one_day_fails_loudly(raw):
    tampered = raw.copy()
    placeholder = tampered.index[(tampered["service_period"] == "#") & (tampered["planned_km"] == "0")][0]
    tampered.loc[placeholder, "planned_km"] = "100"
    with pytest.raises(ValueError, match="exactly one row with km"):
        collapse_to_line_days(tampered)
