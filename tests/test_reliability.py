"""Tests for reporting/reliability.py."""

import pandas as pd
import pytest

from cleaning.build_datasets import build_line_month
from cleaning.clean_incidents import clean_incidents
from cleaning.clean_km import clean_km
from cleaning.load import read_raw_incidents, read_raw_km
from modeling.target import add_targets, model_population
from reporting.reliability import poisson_interval, run_reliability


@pytest.fixture(scope="module")
def incidents():
    return clean_incidents(read_raw_incidents())


@pytest.fixture(scope="module")
def labelled(incidents):
    return add_targets(model_population(incidents))


@pytest.fixture(scope="module")
def line_month(incidents):
    return build_line_month(incidents, clean_km(read_raw_km()))


@pytest.fixture(scope="module")
def tables(labelled, line_month):
    return run_reliability(labelled, line_month)


def test_poisson_interval_by_hand():
    # Standard exact intervals: a count of 0 gives [0, 3.69], a count of 10 [4.80, 18.39].
    low, high = poisson_interval(pd.Series([0, 10]))
    assert low == pytest.approx([0, 4.795], abs=0.001)
    assert high == pytest.approx([3.689, 18.390], abs=0.001)


# ---------------------------------------------------------------------------
# Rates per line and year
# ---------------------------------------------------------------------------
def test_line_year_counts_match_the_line_month_table(tables, line_month):
    rates = tables["rates_by_line_year"]
    assert rates["months"].sum() == len(line_month)
    assert rates["incidents"].sum() == line_month["n_incidents"].sum()
    assert rates["delay_5plus"].sum() == line_month["n_delay_5plus"].sum()
    assert rates["planned_km_million"].sum() * 1e6 == pytest.approx(line_month["planned_km_filled"].sum())


def test_a_rate_never_mixes_real_and_estimated_km(tables):
    rates = tables["rates_by_line_year"]
    assert not rates.loc[rates["year"] <= 2022, "km_is_estimated"].any()
    assert rates.loc[rates["year"] >= 2024, "km_is_estimated"].all()
    # 2023: January to September on real km, October to December on estimated km.
    split = rates[rates["year"] == 2023].set_index(["line", "km_is_estimated"])["months"]
    assert (split.xs(False, level="km_is_estimated") == 9).all()
    assert (split.xs(True, level="km_is_estimated") == 3).all()


def test_a_rate_is_the_count_over_the_km_and_sits_inside_its_interval(tables):
    rates = tables["rates_by_line_year"]
    green_2019 = rates[(rates["line"] == "green") & (rates["year"] == 2019)].iloc[0]
    assert green_2019["delay_5plus"] == 416
    assert green_2019["delay_5plus_per_million_km"] == pytest.approx(416 / green_2019["planned_km_million"])
    for kind in ["delay_5plus", "major"]:
        rate = rates[f"{kind}_per_million_km"]
        assert (rates[f"{kind}_per_million_km_low"] <= rate).all()
        assert (rate <= rates[f"{kind}_per_million_km_high"]).all()


# ---------------------------------------------------------------------------
# Network trend by year
# ---------------------------------------------------------------------------
def test_the_network_counts_each_incident_once(tables, labelled):
    trend = tables["trend_by_year"]
    assert trend["incidents"].sum() == len(labelled)
    assert trend["major"].sum() == labelled["y_major"].sum()
    assert (trend["under_5"] + trend["delay_5plus"] == trend["incidents"]).all()


def test_rates_on_real_km_stop_when_the_km_file_stops(tables):
    trend = tables["trend_by_year"].set_index("year")
    assert trend.loc[:2023, "delay_5plus_per_million_km"].notna().all()
    assert trend.loc[2024:, "delay_5plus_per_million_km"].isna().all()
    assert trend.loc[2023, "months_with_real_km"] == 9
    # Up to 2022 no km is estimated, so the two rates are the same number.
    assert trend.loc[:2022, "delay_5plus_per_million_km"].to_numpy() == pytest.approx(
        trend.loc[:2022, "delay_5plus_per_million_km_filled"].to_numpy()
    )


def test_reporting_and_classification_shares(tables):
    trend = tables["trend_by_year"].set_index("year")
    shares = ["share_under_5", "recorded_delay_in_bucket", "symptom_confirmed",
              "rolling_stock_symptom_confirmed"]
    assert trend[shares].stack().between(0, 1).all()
    # Notebook 04: rolling-stock symptoms were confirmed 28.4% of the time in 2024,
    # against 44-54% in every other year.
    assert trend.loc[2024, "rolling_stock_symptom_confirmed"] == pytest.approx(0.284, abs=0.001)
    assert trend["rolling_stock_symptom_confirmed"].drop(2024).min() > 0.44
