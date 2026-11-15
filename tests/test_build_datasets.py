"""Tests for cleaning/build_datasets.py (line-month table)."""

import pandas as pd
import pytest

import config
from cleaning.build_datasets import DELAY_THRESHOLDS, build_line_month
from cleaning.clean_incidents import clean_incidents
from cleaning.clean_km import clean_km
from cleaning.load import read_raw_incidents, read_raw_km


@pytest.fixture(scope="module")
def incidents():
    return clean_incidents(read_raw_incidents())


@pytest.fixture(scope="module")
def km():
    return clean_km(read_raw_km())


@pytest.fixture(scope="module")
def line_month(incidents, km):
    return build_line_month(incidents, km)


@pytest.fixture(scope="module")
def train_with_line(incidents):
    return incidents[(incidents["incident_type"] == "T") & incidents["line"].notna()]


def test_every_line_in_every_month(line_month):
    # 4 lines x 81 months (2019-01 to 2025-09), no gaps and no duplicates.
    assert len(line_month) == 4 * 81
    assert not line_month.duplicated(["line", "month"]).any()
    assert set(line_month["line"]) == set(config.LINE_NAMES.values())


def test_multi_line_incidents_count_on_each_line(line_month, train_with_line):
    # Each incident adds one per line it involves, and nothing else is counted.
    assert line_month["n_incidents"].sum() == train_with_line["n_lines"].sum()


def test_single_line_counts_match_the_incidents(line_month, train_with_line):
    green = train_with_line[train_with_line["line"].str.contains("green")]
    january = green[green["service_date"].dt.to_period("M") == "2019-01"]
    row = line_month[(line_month["line"] == "green") & (line_month["month"] == "2019-01-01")]
    assert row["n_incidents"].item() == len(january)
    assert row["n_delay_30plus"].item() == (january["delay_floor_minutes"] >= 30).sum()


def test_counts_shrink_as_the_threshold_rises(line_month):
    columns = ["n_incidents"] + [f"n_delay_{t}plus" for t in DELAY_THRESHOLDS]
    assert (line_month[columns].diff(axis=1).iloc[:, 1:] <= 0).all().all()


def test_real_km_matches_the_daily_km(line_month, km):
    with_km = line_month["planned_km"].notna()
    assert line_month.loc[with_km, "planned_km"].sum() == pytest.approx(km["planned_km"].sum())


def test_real_km_stops_where_the_km_file_stops(line_month):
    last_real_month = pd.Timestamp("2023-09-01")
    assert line_month["planned_km"].notna().equals(line_month["month"] <= last_real_month)
    assert line_month["km_is_estimated"].equals(line_month["month"] > last_real_month)


def test_filled_km_repeats_the_last_real_year(line_month):
    filled = line_month.set_index(["line", "month"])["planned_km_filled"]
    assert filled[("orange", pd.Timestamp("2024-03-01"))] == filled[("orange", pd.Timestamp("2023-03-01"))]
    assert filled[("blue", pd.Timestamp("2025-09-01"))] == filled[("blue", pd.Timestamp("2023-09-01"))]
    assert filled[("green", pd.Timestamp("2023-11-01"))] == filled[("green", pd.Timestamp("2022-11-01"))]
    assert line_month["planned_km_filled"].notna().all()


def test_km_ending_mid_month_fails_loudly(incidents, km):
    with pytest.raises(ValueError, match="ends mid-month"):
        build_line_month(incidents, km[km["service_date"] < "2023-09-15"])
