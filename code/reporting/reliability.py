"""Reliability indicators: interruptions per million planned km, and what moved over time.

No model here. These two tables answer the measurement questions of the brief: how
should reliability be measured, and do the trends in the data come from service
volume, reliability, reporting practice or classification?

    rates_by_line_year   one row per (line, year, km real or estimated): planned km,
                         train incidents (all, 5+ min, major) and delay minutes, each
                         as a count and as a rate per million planned km. The 5+ and
                         major rates come with a 95% interval, because a small line
                         has few incidents a year and its rate moves by chance.
    trend_by_year        one row per year, whole network, one block per explanation:
        volume           planned km
        reliability      5+ min and major incidents per million planned km
        reporting        share of incidents under 5 min, and share whose recorded
                         duration agrees with the published delay bucket
        classification   share of first symptoms confirmed by the cause found later

Why a rate, and why 5+ minutes. A line that runs more trains has more incidents, so
counts mix volume with reliability; dividing by planned km removes the volume. STM's
own indicators leave out interruptions under 5 minutes, and those short incidents are
also the ones whose recording changes (notebook 07), so the count of all incidents is
the least stable measure of the three.

Planned km ends on 2023-09-30. Rates on real km stop there; later months use the
carried-forward km of build_datasets.py and are marked as estimates.
A multi-line incident counts on every line it involves (build_datasets.py), so the
line table is not summed to get the network: the network table counts each incident
once.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2

from cleaning.build_datasets import incidents_per_line
from cleaning.clean_incidents import DELAY_BUCKET_CEILINGS

MILLION = 1_000_000

# Incident counts reported as a rate with an interval.
RATE_KINDS = ["delay_5plus", "major"]

# Symptoms that have a cause of the same name, so "was the first call confirmed by the
# investigation?" can be measured (notebook 04, section 2).
SYMPTOMS_WITH_A_MATCHING_CAUSE = ["Clientèle", "Matériel roulant", "Équipements fixes", "Exploitation trains"]
ROLLING_STOCK = "Matériel roulant"


def run_reliability(labelled: pd.DataFrame, line_month: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Both tables; `labelled` are the train incidents with their labels (target.py)."""
    return {
        "rates_by_line_year": rates_by_line_year(labelled, line_month),
        "trend_by_year": trend_by_year(labelled, line_month),
    }


def rates_by_line_year(labelled: pd.DataFrame, line_month: pd.DataFrame) -> pd.DataFrame:
    """Counts and rates per million planned km, per line and year.

    A year with both real and estimated km (2023) gives two rows, so a rate never
    mixes the two. `months` says how many months a row covers.
    """
    counts = (
        incidents_per_line(labelled)
        .groupby(["line", "month"])
        .agg(incidents=("incident_id", "size"), delay_5plus=("y_5plus", "sum"),
             major=("y_major", "sum"), delay_minutes=("delay_minutes", "sum"))
        .reset_index()
    )
    monthly = line_month[["line", "month", "planned_km_filled", "km_is_estimated"]].merge(
        counts, on=["line", "month"], how="left"
    )
    monthly[counts.columns[2:]] = monthly[counts.columns[2:]].fillna(0)

    table = (
        monthly.groupby(["line", monthly["month"].dt.year.rename("year"), "km_is_estimated"])
        .agg(months=("month", "size"), planned_km_million=("planned_km_filled", "sum"),
             incidents=("incidents", "sum"), delay_5plus=("delay_5plus", "sum"),
             major=("major", "sum"), delay_minutes=("delay_minutes", "sum"))
        .reset_index()
    )
    table["planned_km_million"] /= MILLION

    for kind in RATE_KINDS:
        low, high = poisson_interval(table[kind])
        table[f"{kind}_per_million_km"] = table[kind] / table["planned_km_million"]
        table[f"{kind}_per_million_km_low"] = low / table["planned_km_million"]
        table[f"{kind}_per_million_km_high"] = high / table["planned_km_million"]
    table["delay_minutes_per_million_km"] = table["delay_minutes"] / table["planned_km_million"]
    return table


def trend_by_year(labelled: pd.DataFrame, line_month: pd.DataFrame) -> pd.DataFrame:
    """One row per year for the whole network: volume, reliability, reporting, classification.

    Counts cover every month of the year in the data. The rates on real km use only
    the months that have real planned km (2023: January to September), so a count and
    its km always cover the same months; `months_with_real_km` says how many.
    """
    month = labelled["service_date"].dt.to_period("M").dt.to_timestamp()
    year = labelled["service_date"].dt.year.rename("year")
    km_by_month = line_month.groupby("month")[["planned_km", "planned_km_filled"]].sum(min_count=1)
    has_real_km = month.map(km_by_month["planned_km"].notna())

    km_by_year = km_by_month.groupby(km_by_month.index.year).agg(
        months=("planned_km_filled", "size"),
        months_with_real_km=("planned_km", "count"),
        planned_km_million=("planned_km", "sum"),
        planned_km_filled_million=("planned_km_filled", "sum"),
    )
    km_by_year[["planned_km_million", "planned_km_filled_million"]] /= MILLION

    under_5 = ~labelled["y_5plus"]
    counts = pd.DataFrame({
        "incidents": labelled.groupby(year).size(),
        "under_5": under_5.groupby(year).sum(),
        "delay_5plus": labelled["y_5plus"].groupby(year).sum(),
        "major": labelled["y_major"].groupby(year).sum(),
    })
    # The same counts, limited to the months that have real km.
    with_km = labelled[has_real_km].groupby(year[has_real_km])[["y_5plus", "y_major"]].sum()

    table = km_by_year.join(counts)
    real_km = table["planned_km_million"].where(table["months_with_real_km"] > 0)
    table["planned_km_million"] = real_km
    table["delay_5plus_per_million_km"] = with_km["y_5plus"] / real_km
    table["major_per_million_km"] = with_km["y_major"] / real_km
    table["delay_5plus_per_million_km_filled"] = table["delay_5plus"] / table["planned_km_filled_million"]
    table["major_per_million_km_filled"] = table["major"] / table["planned_km_filled_million"]

    # Reporting practice.
    recorded_in_bucket = labelled["recorded_minutes"].between(
        labelled["delay_floor_minutes"], labelled["delay_floor_minutes"].map(DELAY_BUCKET_CEILINGS)
    )
    table["share_under_5"] = table["under_5"] / table["incidents"]
    table["recorded_delay_in_bucket"] = recorded_in_bucket.groupby(year).mean()

    # Classification: is the first symptom confirmed by the cause found later?
    comparable = labelled[labelled["symptom"].isin(SYMPTOMS_WITH_A_MATCHING_CAUSE)]
    confirmed = comparable["symptom"] == comparable["primary_cause"]
    rolling_stock = comparable["symptom"] == ROLLING_STOCK
    table["symptom_confirmed"] = confirmed.groupby(year[comparable.index]).mean()
    table["rolling_stock_symptom_confirmed"] = (
        confirmed[rolling_stock].groupby(year[comparable.index][rolling_stock]).mean()
    )
    return table.rename_axis("year").reset_index()


def poisson_interval(count: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Exact 95% interval for the expected value behind a count of random events.

    A count of 0 has a lower bound of 0 and an upper bound of about 3.7.
    """
    low = np.nan_to_num(chi2.ppf(0.025, 2 * count) / 2)
    high = chi2.ppf(0.975, 2 * count + 2) / 2
    return low, high
