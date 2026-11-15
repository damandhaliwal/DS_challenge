"""Cleaning rules for the planned-kilometres file.

Input: the raw planned km as read by cleaning/load.py (English column names, all text).
Output: one row per (line, day) with the planned km as a number.

Each rule comes from a finding in exploration notebook 02.
"""

import pandas as pd

import config

# ---------------------------------------------------------------------------
# Step 1: placeholder rows
# ---------------------------------------------------------------------------
# On some days a line has a second, placeholder row: service period "#" and day type
# "Non affecté" (all of 2019, 2020-01-01..05 and 2020-06-01..14; notebook 02,
# "Placeholder rows" section). Exactly one of the two rows carries the day's km: usually
# the regular row, but the placeholder row for periods 19U, 19S, 19N and 20C.
PERIOD_PLACEHOLDER = "#"

# ---------------------------------------------------------------------------
# Step 2: day type
# ---------------------------------------------------------------------------
# French day type -> name used in the code. The four holiday levels are the STM's
# holiday service periods; "Premième" is a typo in the raw file for "Première".
DAY_TYPES = {
    "Semaine": "weekday",
    "Samedi": "saturday",
    "Dimanche": "sunday",
    "Premième pér. fête": "holiday_1",
    "Deuxième pér. fête": "holiday_2",
    "Troisième pér. fête": "holiday_3",
    "Quatrième pér. fête": "holiday_4",
}


def clean_km(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply every cleaning rule to the raw planned km and return the cleaned table.

    Steps:
        1. One row per (line, day): km summed over the regular and placeholder rows,
           service period and day type taken from the regular row
        2. Day type in English, plus a holiday flag

    Columns that only repeat the date (year, month, week, weekday name) are dropped;
    they can be derived again from service_date. The run category is always "001".
    """
    return raw.pipe(collapse_to_line_days).pipe(name_day_types)


def collapse_to_line_days(km: pd.DataFrame) -> pd.DataFrame:
    """Step 1: one row per (line, day) with the planned km as a number."""
    planned_km = km["planned_km"].str.replace(",", ".").astype(float)   # decimal comma
    is_placeholder = km["service_period"] == PERIOD_PLACEHOLDER
    line_day = [km["line"], km["calendar_date"]]

    _check_line_day_assumptions(km, planned_km, is_placeholder)

    regular = km.loc[~is_placeholder].set_index(["line", "calendar_date"])
    total_km = planned_km.groupby(line_day).sum().rename_axis(["line", "calendar_date"])

    collapsed = regular[["service_period", "day_type"]].assign(planned_km=total_km).reset_index()
    return pd.DataFrame(
        {
            "line": collapsed["line"].map(lambda name: config.LINE_NAMES[config.FRENCH_LINE_NUMBERS[name]]),
            # The km file has one date per day. It is called service_date so that it
            # joins the incidents' service day directly.
            "service_date": pd.to_datetime(collapsed["calendar_date"], format="%Y-%m-%d"),
            "service_period": collapsed["service_period"],
            "day_type": collapsed["day_type"],
            "planned_km": collapsed["planned_km"],
        }
    ).sort_values(["line", "service_date"], ignore_index=True)


def name_day_types(km: pd.DataFrame) -> pd.DataFrame:
    """Step 2: day type in English, and is_holiday for the four holiday levels."""
    unknown = ~km["day_type"].isin(list(DAY_TYPES))
    if unknown.any():
        raise ValueError(f"Unknown day types: {sorted(km.loc[unknown, 'day_type'].unique())}")

    day_type = km["day_type"].map(DAY_TYPES)
    return km.assign(day_type=day_type, is_holiday=day_type.str.startswith("holiday"))


def _check_line_day_assumptions(
    km: pd.DataFrame, planned_km: pd.Series, is_placeholder: pd.Series
) -> None:
    """Fail loudly if a new raw file breaks one of the facts that step 1 relies on."""
    line_day = [km["line"], km["calendar_date"]]
    regular_rows = (~is_placeholder).groupby(line_day).sum()
    rows_with_km = (planned_km > 0).groupby(line_day).sum()
    days = pd.to_datetime(km["calendar_date"], format="%Y-%m-%d")

    problems = {
        "a line-day does not have exactly one regular row": (regular_rows != 1).any(),
        "a line-day does not have exactly one row with km": (rows_with_km != 1).any(),
        "a placeholder row has a real day type": (
            km.loc[is_placeholder, "day_type"] != "Non affecté"
        ).any(),
        "a line is not one of the four metro lines": (
            ~km["line"].isin(list(config.FRENCH_LINE_NUMBERS))
        ).any(),
        "the run category is not always 001": (km["run_category"] != "001").any(),
        "some days are missing between the first and last day": (
            days.nunique() != len(pd.date_range(days.min(), days.max()))
        ),
    }
    broken = [problem for problem, failed in problems.items() if failed]
    if broken:
        raise ValueError("Planned km file: " + "; ".join(broken) + ".")
