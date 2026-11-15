"""Single entry point for getting the data: cleaned if it exists, built if it does not.

    load_incidents()               -> cleaned incidents (read from data/processed/ if saved)
    load_incidents(override=True)  -> rerun the full cleaning from the raw file and re-save

and the same for load_km() and load_line_month(). The rest of the project only ever
calls these functions, so every analysis runs on the same cleaned data.

Reading the raw files is the first step of the cleaning. It makes no decisions:
every value is kept as the original text (placeholders such as "#", "#N/A" and
"Non affecté" included), so every change to the data lives in clean_incidents.py /
clean_km.py, where it is documented and tested.
"""

from pathlib import Path
from typing import Callable

import pandas as pd

import config
from cleaning.build_datasets import build_line_month
from cleaning.clean_incidents import clean_incidents
from cleaning.clean_km import clean_km

# ---------------------------------------------------------------------------
# Column names: French header in the raw file -> English name used in the code.
# The order matches the raw files, and the raw readers check that it still does, so
# a change in the published format fails loudly instead of silently shifting columns.
# ---------------------------------------------------------------------------
INCIDENT_COLUMNS = {
    "Numero d'incident": "incident_id",
    "Type d'incident": "incident_type",          # T = train, S = station
    "Cause primaire": "primary_cause",           # set after investigation
    "Cause secondaire": "secondary_cause",       # set after investigation
    "Symptome": "symptom",                       # set at report time
    "Ligne": "line",
    "Numéro de tournée": "run_number",
    "Heure de l'incident": "start_time",         # service-day clock, can exceed 24:00
    "Heure de reprise": "end_time",              # service-day clock, can exceed 24:00
    "Incident en minutes": "delay_bucket",       # binned service delay
    "Véhicule": "car_number",
    "Porte de voiture": "door_numbers",
    "Type de matériel": "rolling_stock",
    "Code de lieu": "location",
    "Dommage matériel": "possible_damage",
    "KFS": "emergency_brake",
    "Porte": "door_involved",
    "Urgence métro": "emergency_team",
    "CAT": "traction_power_cut",
    "Évacuation": "evacuation",
    "Année civile": "year",                      # service day
    "Année civile/mois": "year_month_label",     # e.g. "janv-19"
    "Mois calendrier": "month",                  # service day
    "Jour du mois": "day_of_month",              # service day
    "Jour de la semaine": "weekday",             # service day, 1 = Monday
    "Jour calendaire": "calendar_date",          # real calendar date
}

KM_COLUMNS = {
    "Ligne": "line",
    "Période Exploitation": "service_period",    # e.g. "19J"; "#" on placeholder rows
    "Type de Jour": "day_type",
    "Tournée": "run_category",
    "Année civile": "year",
    "Année civile/mois": "year_month",
    "Année civile/semaine": "week_start",        # Monday of the week
    "Jour calendaire": "calendar_date",
    "Jour de la semaine": "weekday_name",        # French weekday name
    "Mois calendrier": "month",
    "KM planifié": "planned_km",                 # decimal comma, e.g. "16315,92"
}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def load_incidents(override: bool = False) -> pd.DataFrame:
    """Cleaned metro incidents, one row per incident.

    override=False: read the saved cleaned file if it exists, otherwise build it.
    override=True:  always rerun the full cleaning from the raw file and re-save.
    """
    return _load_or_build(
        path=config.INCIDENTS_CLEAN,
        build=lambda: clean_incidents(read_raw_incidents()),
        override=override,
    )


def load_km(override: bool = False) -> pd.DataFrame:
    """Cleaned planned kilometres, one row per (line, day). Same override rule as above."""
    return _load_or_build(
        path=config.KM_CLEAN,
        build=lambda: clean_km(read_raw_km()),
        override=override,
    )


def load_line_month(override: bool = False) -> pd.DataFrame:
    """Incident counts and planned km per (line, month). Same override rule as above.

    override=True also rebuilds the cleaned incidents and km it is built from, so the
    whole chain is refreshed from the raw files.
    """
    return _load_or_build(
        path=config.LINE_MONTH,
        build=lambda: build_line_month(load_incidents(override), load_km(override)),
        override=override,
    )


# ---------------------------------------------------------------------------
# Raw readers (first step of the cleaning)
# ---------------------------------------------------------------------------
def read_raw_incidents() -> pd.DataFrame:
    """Raw metro incidents exactly as published, all columns as text."""
    # ";"-separated, UTF-8 with a byte-order mark at the start of the file.
    return _read_as_text(config.INCIDENTS_RAW, INCIDENT_COLUMNS, sep=";", encoding="utf-8-sig")


def read_raw_km() -> pd.DataFrame:
    """Raw planned kilometres exactly as published, all columns as text."""
    # ","-separated; numbers with a decimal comma are quoted, so the default parser
    # keeps them in one field.
    return _read_as_text(config.KM_RAW, KM_COLUMNS, sep=",", encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_or_build(path: Path, build: Callable[[], pd.DataFrame], override: bool) -> pd.DataFrame:
    """Return the table saved at `path`, building and saving it first when needed."""
    if override or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        build().to_parquet(path, index=False)
    # Always return what was read back from disk, so a fresh build and a cached load
    # give exactly the same table (same column types included).
    return pd.read_parquet(path)


def _read_as_text(path: Path, rename: dict, **read_options) -> pd.DataFrame:
    """Read a delimited file with every column as text and no automatic NA detection.

    Turning off NA detection matters: pandas would otherwise convert some
    placeholders (like "#N/A") to missing values and leave others (like "#")
    untouched, mixing two kinds of "missing" before we have decided how to treat them.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, **read_options)

    if list(frame.columns) != list(rename):
        raise ValueError(
            f"Unexpected columns in {path.name}.\n"
            f"Expected: {list(rename)}\nFound:    {list(frame.columns)}"
        )
    return frame.rename(columns=rename)
