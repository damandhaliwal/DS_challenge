"""Analysis tables built from the cleaned incidents and planned km.

line-month table: one row per (line, month) from 2019-01 to 2025-09, with incident
counts at several delay thresholds and the planned km as exposure, so reliability can
be compared as a rate (incidents per million planned km) across lines and over time.

Decisions:
  * Train incidents only. Station incidents are always in the shortest delay bucket
    (notebook 01, to-do item 2), so they never count as a service interruption.
  * A multi-line incident counts on every line it involves: each of those lines was
    disrupted. Network totals must therefore come from the incident table, not from
    summing this one. The 97 train incidents with an unknown line are left out.
  * The planned km ends on 2023-09-30, two years before the incidents. Two exposure
    columns are kept side by side:
        planned_km          real km only, missing after 2023-09
        planned_km_filled   the same, with each later month filled by the same line
                            and calendar month of the last real year (2022-10 to
                            2023-09); km_is_estimated marks those months.
    The real column is the main one; the filled column is a sensitivity check. The
    fill assumes the schedule did not change after 2023-09; the km already changed by
    up to 10% a year, and by +34% on the yellow line in 2023 (notebook 02). The yellow
    increase started in 2023-01, so the filled yellow October-December months come
    from the lower 2022 level and are probably about 30% too low.
"""

import pandas as pd

# Count incidents with a delay of at least this many minutes. Each is a lower bound
# of a delay bucket, so the counts are exact (clean_incidents.py, step 6).
DELAY_THRESHOLDS = [5, 10, 15, 30]


def build_line_month(incidents: pd.DataFrame, km: pd.DataFrame) -> pd.DataFrame:
    """One row per (line, month): incident counts and planned km."""
    counts = _count_incidents(incidents)
    exposure = _monthly_km(km)
    return (
        counts.merge(exposure, on=["line", "month"], how="left")
        .pipe(_fill_km_after_the_last_real_year, last_real_month=exposure["month"].max())
        .sort_values(["line", "month"], ignore_index=True)
    )


def incidents_per_line(incidents: pd.DataFrame) -> pd.DataFrame:
    """One row per (incident, line involved), with its month; unknown lines left out.

    "green+orange" -> one green row and one orange row, each with every column of the
    incident.
    """
    known = incidents[incidents["line"].notna()]
    return known.assign(
        line=known["line"].str.split("+"),
        month=known["service_date"].dt.to_period("M").dt.to_timestamp(),
    ).explode("line")


def _count_incidents(incidents: pd.DataFrame) -> pd.DataFrame:
    """Train incident counts per (line, month), with zeros for months without any."""
    per_line = incidents_per_line(incidents[incidents["incident_type"] == "T"])

    columns = {"n_incidents": per_line["delay_floor_minutes"] >= 0}
    for threshold in DELAY_THRESHOLDS:
        columns[f"n_delay_{threshold}plus"] = per_line["delay_floor_minutes"] >= threshold
    counts = pd.DataFrame(columns).groupby([per_line["line"], per_line["month"]]).sum()

    # Every line in every month of the incident coverage, so a quiet month is a 0
    # rather than a missing row.
    all_line_months = pd.MultiIndex.from_product(
        [
            sorted(per_line["line"].unique()),
            pd.date_range(per_line["month"].min(), per_line["month"].max(), freq="MS"),
        ],
        names=["line", "month"],
    )
    return counts.reindex(all_line_months, fill_value=0).reset_index()


def _monthly_km(km: pd.DataFrame) -> pd.DataFrame:
    """Planned km per (line, month). Only complete months are kept."""
    last_day = km["service_date"].max()
    if last_day != last_day + pd.offsets.MonthEnd(0):
        raise ValueError(f"The planned km ends mid-month ({last_day:%Y-%m-%d}).")

    month = km["service_date"].dt.to_period("M").dt.to_timestamp()
    return km.groupby([km["line"], month.rename("month")])["planned_km"].sum().reset_index()


def _fill_km_after_the_last_real_year(table: pd.DataFrame, last_real_month: pd.Timestamp) -> pd.DataFrame:
    """Add planned_km_filled and km_is_estimated (see the module docstring)."""
    last_real_year = table[
        (table["month"] > last_real_month - pd.DateOffset(years=1))
        & (table["month"] <= last_real_month)
    ]
    km_by_calendar_month = last_real_year.set_index(
        [last_real_year["line"], last_real_year["month"].dt.month]
    )["planned_km"]

    is_estimated = table["month"] > last_real_month
    fill = pd.Series(
        list(zip(table["line"], table["month"].dt.month)), index=table.index
    ).map(km_by_calendar_month)
    return table.assign(
        planned_km_filled=table["planned_km"].where(~is_estimated, fill),
        km_is_estimated=is_estimated,
    )
