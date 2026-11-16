"""Next month's major incidents on each line: does a VAR beat simple baselines?

A different question from the incident model. The incident model ranks each incident
as it is reported; this forecasts how many major incidents each line will have next
month, from the monthly counts alone. That is where a time-series model such as a VAR
belongs. Notebook 08 looks at the counts first: they are small (1 to 9 a month), close
to random (Poisson-like) counts, with no clear seasonality, some memory within a line
and none across lines.

Forecasters, each given every month before the one it forecasts:
    long_run_mean          the line's average over all past months
    recent_mean            the line's average over the last 12 months
    same_month_last_year   the line's count 12 months earlier
    seasonal_poisson       Poisson regression on line + calendar month
    ar1                    each line regressed on its own last month (AR(1))
    var1                   all four lines regressed on all four lines' last month
                           (VAR(1)). The only difference with ar1 is the cross-line
                           terms, so var1 - ar1 is the test of what a VAR adds.
One lag only: with 24 to 59 past months, a VAR(2) would fit 9 numbers per line.

Backtest: the same test years as the incident model (2021, 2022, 2023), one month
ahead, refitted every month on all earlier months. The holdout is not used.
Scores: MAE and RMSE over the (line, month) forecasts. Paired comparisons resample
whole test months, so the four lines of a month stay together.

A multi-line incident counts on every line it involves (cleaning/build_datasets.py).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR

import config
from cleaning.build_datasets import incidents_per_line
from modeling.backtest import BOOTSTRAP_SAMPLES, FOLDS

REFERENCE_MODEL = "recent_mean"
RECENT_MONTHS = 12


def monthly_major_counts(labelled: pd.DataFrame) -> pd.DataFrame:
    """Major incidents per month (rows) and line (columns), with zeros for quiet months."""
    per_line = incidents_per_line(labelled)
    counts = per_line.groupby(["month", "line"])["y_major"].sum().unstack(fill_value=0)
    months = pd.date_range(counts.index.min(), counts.index.max(), freq="MS", name="month")
    return counts.reindex(months, fill_value=0)


# ---------------------------------------------------------------------------
# Forecasters: history (months x lines, every month before the forecast) -> one
# forecast per line for the next month
# ---------------------------------------------------------------------------
def long_run_mean(history: pd.DataFrame) -> pd.Series:
    return history.mean()


def recent_mean(history: pd.DataFrame) -> pd.Series:
    return history.tail(RECENT_MONTHS).mean()


def same_month_last_year(history: pd.DataFrame) -> pd.Series:
    return history.iloc[-12].astype(float)


def seasonal_poisson(history: pd.DataFrame) -> pd.Series:
    """Poisson regression, log(expected count) = line effect + calendar-month effect."""
    long = history.rename_axis(index="month", columns="line").stack().rename("count").reset_index()
    next_month = history.index[-1] + pd.offsets.MonthBegin(1)
    future = pd.DataFrame({"month": next_month, "line": history.columns})

    def design(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([
            pd.get_dummies(pd.Categorical(frame["line"], categories=history.columns), prefix="line"),
            pd.get_dummies(pd.Categorical(frame["month"].dt.month, categories=range(1, 13)), prefix="month"),
        ], axis=1)

    model = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(design(long), long["count"])
    return pd.Series(model.predict(design(future)), index=history.columns)


def ar1(history: pd.DataFrame) -> pd.Series:
    forecasts = {
        line: AutoReg(history[line].to_numpy(dtype=float), lags=1).fit().forecast(1)[0]
        for line in history.columns
    }
    return pd.Series(forecasts).clip(lower=0)


def var1(history: pd.DataFrame) -> pd.Series:
    values = history.to_numpy(dtype=float)
    forecast = VAR(values).fit(1).forecast(values[-1:], steps=1)[0]
    return pd.Series(forecast, index=history.columns).clip(lower=0)


FORECASTERS = {
    "long_run_mean": long_run_mean,
    "recent_mean": recent_mean,
    "same_month_last_year": same_month_last_year,
    "seasonal_poisson": seasonal_poisson,
    "ar1": ar1,
    "var1": var1,
}

# (better, simpler) pairs for the paired comparisons.
PAIRS = [(name, REFERENCE_MODEL) for name in FORECASTERS if name != REFERENCE_MODEL] + [("var1", "ar1")]


def run_forecast_backtest(labelled: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Forecast every month of every test year with every forecaster.

        predictions   one row per (fold, month, line, model): forecast and actual count
        metrics       one row per (fold, model): MAE and RMSE
        comparisons   per (pair, fold): MAE difference (first minus second; negative
                      means the first is better), with a 95% interval over test months
        summary       one row per model: mean over folds of MAE and RMSE, and against
                      the reference the mean MAE difference and the number of folds
                      where it is clearly better / worse
    """
    counts = monthly_major_counts(labelled)

    predictions = []
    for test_year, (first_year, _) in FOLDS.items():
        for month in counts.index[counts.index.year == test_year]:
            history = counts[(counts.index.year >= first_year) & (counts.index < month)]
            for name, forecaster in FORECASTERS.items():
                predictions.append(pd.DataFrame({
                    "fold": test_year, "month": month, "line": counts.columns, "model": name,
                    "forecast": forecaster(history).reindex(counts.columns).to_numpy(),
                    "actual": counts.loc[month].to_numpy(),
                }))
    predictions = pd.concat(predictions, ignore_index=True)

    metrics = pd.DataFrame([
        {"fold": fold, "model": model,
         "mae": mean_absolute_error(scored["actual"], scored["forecast"]),
         "rmse": root_mean_squared_error(scored["actual"], scored["forecast"])}
        for (fold, model), scored in predictions.groupby(["fold", "model"], sort=False)
    ])
    comparisons = pd.concat(
        [compare_forecasts(predictions, better, simpler) for better, simpler in PAIRS], ignore_index=True
    )

    summary = metrics.groupby("model", sort=False)[["mae", "rmse"]].mean()
    against_reference = comparisons[comparisons["comparison"].str.endswith(f" - {REFERENCE_MODEL}")]
    by_model = against_reference.assign(
        model=against_reference["comparison"].str.split(" - ").str[0]
    ).groupby("model")
    summary["mae_difference_vs_reference"] = by_model["mae_difference"].mean()
    # Folds where the 95% interval of the difference lies entirely below / above zero.
    summary["folds_clearly_better"] = by_model["high"].apply(lambda high: (high < 0).sum())
    summary["folds_clearly_worse"] = by_model["low"].apply(lambda low: (low > 0).sum())
    summary.loc[REFERENCE_MODEL, ["mae_difference_vs_reference", "folds_clearly_better", "folds_clearly_worse"]] = 0

    return {
        "predictions": predictions,
        "metrics": metrics,
        "comparisons": comparisons,
        "summary": summary.reset_index(),
    }


def compare_forecasts(predictions: pd.DataFrame, first: str, second: str) -> pd.DataFrame:
    """Paired MAE difference (first minus second), fold by fold, resampling test months."""
    errors = (
        predictions.assign(error=(predictions["forecast"] - predictions["actual"]).abs())
        .pivot_table(index=["fold", "month", "line"], columns="model", values="error")
    )
    rows = []
    for fold, fold_errors in errors.groupby("fold"):
        # Error of each month, summed over its lines, so a resample keeps months whole.
        by_month = fold_errors.groupby("month")[[first, second]].sum()
        difference = (by_month[first] - by_month[second]).to_numpy()
        lines = fold_errors.index.get_level_values("line").nunique()

        rng = np.random.default_rng(config.RANDOM_SEED)
        samples = rng.integers(len(difference), size=(BOOTSTRAP_SAMPLES, len(difference)))
        resampled = difference[samples].mean(axis=1) / lines
        rows.append({
            "fold": fold,
            "comparison": f"{first} - {second}",
            "mae_difference": difference.mean() / lines,
            "low": np.percentile(resampled, 2.5),
            "high": np.percentile(resampled, 97.5),
        })
    return pd.DataFrame(rows)
