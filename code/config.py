"""Project-wide paths and constants.

Every other module imports its paths from here, so the project can be moved or
cloned anywhere without editing code.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# config.py lives in <repo>/code/, so the repository root is one level up.
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# ---------------------------------------------------------------------------
# Raw input files
# ---------------------------------------------------------------------------
# Both files were published by the Ville de Montréal open-data portal (extract
# dated 2026-07-07). They were shipped with a ".xls" extension, but they are
# really delimited text files, so they are stored here with a ".csv" extension.
#   incidents: "Incidents métro_20260707.xls"
#   km:        "Kilométrage métro planifé_20260707.xls"
INCIDENTS_RAW = RAW_DIR / "incidents_metro_20260707.csv"
KM_RAW = RAW_DIR / "km_planifie_20260707.csv"

# ---------------------------------------------------------------------------
# Cleaned outputs
# ---------------------------------------------------------------------------
# Written by cleaning/load.py the first time the data is loaded (or when it is
# called with override=True). Parquet keeps the column types (dates, categories)
# that a CSV would lose.
INCIDENTS_CLEAN = PROCESSED_DIR / "incidents_clean.parquet"
KM_CLEAN = PROCESSED_DIR / "km_clean.parquet"
LINE_MONTH = PROCESSED_DIR / "line_month.parquet"

# ---------------------------------------------------------------------------
# Model outputs
# ---------------------------------------------------------------------------
BACKTEST_DIR = OUTPUTS_DIR / "backtest"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Metro lines
# ---------------------------------------------------------------------------
# STM line number -> name used in the code. Both raw files name a single line by its
# French colour; the incidents file names several lines by their numbers ("Ligne 1, 2").
LINE_NAMES = {"1": "green", "2": "orange", "4": "yellow", "5": "blue"}
FRENCH_LINE_NUMBERS = {
    "Ligne verte": "1",
    "Ligne orange": "2",
    "Ligne jaune": "4",
    "Ligne bleue": "5",
}

# Stations at the end of a line, where trains turn around, and stations where two or
# more lines meet. Spelled as in the cleaned incidents (clean_incidents.py, step 4).
TERMINAL_STATIONS = {
    "Angrignon", "Honorée-Beaugrand",     # green
    "Côte-Vertu", "Montmorency",          # orange
    "Berri-UQAM", "Longueuil",            # yellow
    "Snowdon", "St-Michel",               # blue
}
TRANSFER_STATIONS = {"Berri-UQAM", "Lionel-Groulx", "Jean-Talon", "Snowdon"}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
