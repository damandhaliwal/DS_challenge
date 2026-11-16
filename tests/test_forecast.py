"""Tests for modeling/forecast.py: the forecasters by hand, the backtest on one fold."""

import numpy as np
import pandas as pd
import pytest

from cleaning.clean_incidents import clean_incidents
from cleaning.load import read_raw_incidents
from modeling import backtest, forecast
from modeling.target import add_targets, model_population


@pytest.fixture
def history():
    """24 months of two lines: line a alternates 0/2, line b is always 3."""
    months = pd.date_range("2019-01-01", periods=24, freq="MS", name="month")
    return pd.DataFrame({"a": [0, 2] * 12, "b": [3] * 24}, index=months)


def test_simple_forecasters_by_hand(history):
    assert forecast.long_run_mean(history).to_dict() == {"a": 1.0, "b": 3.0}
    assert forecast.recent_mean(history).to_dict() == {"a": 1.0, "b": 3.0}
    # The next month is 2021-01; 12 months earlier, 2020-01, line a had 0.
    assert forecast.same_month_last_year(history).to_dict() == {"a": 0.0, "b": 3.0}


def test_ar1_follows_an_alternating_line(history):
    # Last month line a had 2, so an AR(1) fitted to 0, 2, 0, 2, ... forecasts 0.
    assert forecast.ar1(history.assign(b=np.arange(24) % 5))["a"] == pytest.approx(0.0, abs=1e-6)


def test_seasonal_poisson_learns_the_calendar_month(history):
    # Every line is 3 times higher in even calendar months than in odd ones (the model
    # has a line effect and a month effect, no interaction). The next month, January,
    # is odd.
    seasonal = history.assign(a=[1, 3] * 12, b=[2, 6] * 12)
    assert forecast.seasonal_poisson(seasonal).to_dict() == pytest.approx({"a": 1.0, "b": 2.0}, rel=0.01)


def test_counts_cover_every_month_with_zeros():
    labelled = add_targets(model_population(clean_incidents(read_raw_incidents())))
    counts = forecast.monthly_major_counts(labelled)
    assert list(counts.columns) == ["blue", "green", "orange", "yellow"]
    assert len(counts) == len(pd.date_range(counts.index.min(), counts.index.max(), freq="MS"))
    # A multi-line incident counts on each of its lines, so the table holds at least
    # every major incident with a known line.
    assert counts.to_numpy().sum() >= labelled.loc[labelled["line"].notna(), "y_major"].sum()


@pytest.fixture(scope="module")
def reduced():
    labelled = add_targets(model_population(clean_incidents(read_raw_incidents())))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(forecast, "FOLDS", {2021: (2019, 2020)})
        patch.setattr(forecast, "BOOTSTRAP_SAMPLES", 50)
        return forecast.run_forecast_backtest(labelled)


def test_every_month_line_and_model_is_forecast(reduced):
    assert len(reduced["predictions"]) == 12 * 4 * len(forecast.FORECASTERS)
    assert (reduced["predictions"]["forecast"] >= 0).all()


def test_the_reference_is_compared_with_every_other_model(reduced):
    expected = [f"{first} - {second}" for first, second in forecast.PAIRS]
    assert list(reduced["comparisons"]["comparison"]) == expected
    reference = reduced["summary"].set_index("model").loc[forecast.REFERENCE_MODEL]
    assert reference["mae_difference_vs_reference"] == 0
