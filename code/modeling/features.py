"""Model inputs: only what is known when an incident is reported.

The inputs are a whitelist. Every other column of the cleaned incidents is left out,
in particular everything set after the incident (causes, rolling stock, end time,
delay, evacuation, emergency team, power cut, damage), so a new column in the data
can never slip into a model by accident.

Main inputs (each one checked against the data in notebooks 03-04 and before writing):
    symptom                first description of the problem       strongest signal
    line_group             line, or "multi_line" / "unknown"
    location_kind          kind of place ("whole line" excluded, see below)
    at_terminal            at or next to a terminal station       fewer major incidents
    door_involved          the raw door flag (Porte) is set       more 5+, fewer major
    emergency_brake        a passenger pulled the emergency brake more 5+, fewer major
    time_of_day            service-clock band (peaks, late night)

Dropped for carrying no signal (notebook 06, documentation.md, Decision 8): random
forest permutation importance, the lasso path and a drop-one backtest of gradient
boosting all agree that these change nothing. They are still built, so the analysis
can be rerun:
    at_transfer_station    at or next to a transfer station
    is_weekend             Saturday or Sunday service day

Leak suspects, kept out of the main model and tested separately (documentation.md):
    has_run_number         train incidents without a run number are rarely major (1.4%
                           vs 6.6%), but the data does not say whether the run number is
                           recorded at report time or added after a train was held.
    whole_line_location    a location recorded as a whole line or the network may only
                           be known once the incident has spread.
    has_door_numbers       door numbers are listed. They may be written in after the
                           doors are inspected (Decision 9); until then they were merged
                           into door_involved.

Assumed to be set at report time, kept in the main model (documentation.md, Decision 3):
    door_involved, emergency_brake   both describe what the operator sees or hears as
                           the incident starts, but the data has no time stamp for
                           them either. A sensitivity run drops both.

Left out on purpose: the station itself (68 values with few severe events each, so a
noisy input; the terminal flag carries its most robust signal), car numbers
and the door numbers themselves, month and holidays (no seasonality in the severe rate, notebook 04), and
station incidents (no precursor signal, documentation.md, Finding 1).
"""

import pandas as pd

import config
from cleaning.clean_incidents import WHOLE_LINE_OR_NETWORK

UNKNOWN = "unknown"

# Fixed category lists, so every training and test split has the same levels, and a
# value never seen before fails loudly instead of becoming a silent new category.
SYMPTOMS = [
    "Clientèle",
    "Divers",
    "Exploitation stations",
    "Exploitation trains",
    "Feu, fumée, odeur, produit, etc...",
    "Matériel roulant",
    "Équipements fixes",
    UNKNOWN,
]
LINE_GROUPS = [*config.LINE_NAMES.values(), "multi_line", UNKNOWN]
UNSPECIFIED_LOCATION = "unspecified"   # whole line / network, or an unexplained code
LOCATION_KINDS = [
    "station",
    "transfer_platform",
    "between_stations",
    "ancillary_structure",
    "facility",
    UNSPECIFIED_LOCATION,
]

# Bands of the service-day clock, in hours since the start of the service day (the
# clock runs past 24 after midnight). Band i covers [edge i, edge i+1).
TIME_OF_DAY_EDGES = [0, 7, 9, 16, 19, 22, 30]
TIME_OF_DAY_BANDS = ["early", "am_peak", "midday", "pm_peak", "evening", "late"]

CATEGORICAL_FEATURES = ["symptom", "line_group", "location_kind", "time_of_day"]
BOOLEAN_FEATURES = ["at_terminal", "door_involved", "emergency_brake"]
MAIN_FEATURES = CATEGORICAL_FEATURES + BOOLEAN_FEATURES
NO_SIGNAL_FEATURES = ["at_transfer_station", "is_weekend"]
LEAK_SUSPECT_FEATURES = ["has_run_number", "whole_line_location", "has_door_numbers"]
TIMING_ASSUMED_FEATURES = ["door_involved", "emergency_brake"]


def build_features(incidents: pd.DataFrame) -> pd.DataFrame:
    """Every input (main, no-signal and leak suspects), one row per incident, same index."""
    station_ends = incidents[["station_from", "station_to"]]
    hours_into_service_day = (incidents["start_at"] - incidents["service_date"]) / pd.Timedelta(hours=1)

    return pd.DataFrame(
        {
            "symptom": _category(incidents["symptom"].fillna(UNKNOWN), SYMPTOMS),
            "line_group": _category(_line_group(incidents), LINE_GROUPS),
            "location_kind": _category(
                incidents["location_kind"]
                .replace(WHOLE_LINE_OR_NETWORK, UNSPECIFIED_LOCATION)
                .fillna(UNSPECIFIED_LOCATION),
                LOCATION_KINDS,
            ),
            "time_of_day": _category(_time_of_day(hours_into_service_day), TIME_OF_DAY_BANDS),
            # A segment counts when either of its two stations qualifies.
            "at_terminal": station_ends.isin(config.TERMINAL_STATIONS).any(axis=1),
            "door_involved": incidents["door_involved"],
            "emergency_brake": incidents["emergency_brake"],
            "at_transfer_station": station_ends.isin(config.TRANSFER_STATIONS).any(axis=1),
            "is_weekend": incidents["service_date"].dt.dayofweek >= 5,
            "has_run_number": incidents["run_number"].notna(),
            "whole_line_location": incidents["location_kind"] == WHOLE_LINE_OR_NETWORK,
            "has_door_numbers": incidents["door_numbers"].notna(),
        },
        index=incidents.index,
    )


def _line_group(incidents: pd.DataFrame) -> pd.Series:
    """Single line by name; several lines -> "multi_line"; unknown -> "unknown"."""
    return (
        incidents["line"]
        .where(incidents["n_lines"] == 1, "multi_line")
        .where(incidents["line"].notna(), UNKNOWN)
    )


def _time_of_day(hours: pd.Series) -> pd.Series:
    """Hours into the service day -> band name."""
    if not hours.between(TIME_OF_DAY_EDGES[0], TIME_OF_DAY_EDGES[-1], inclusive="left").all():
        raise ValueError("A start time falls outside the service-day clock bands.")
    return pd.cut(hours, bins=TIME_OF_DAY_EDGES, right=False, labels=TIME_OF_DAY_BANDS).astype(str)


def _category(values: pd.Series, categories: list[str]) -> pd.Series:
    """Fixed-level categorical; a value outside the list fails loudly."""
    unexpected = ~values.isin(categories)
    if unexpected.any():
        raise ValueError(f"Unexpected values: {sorted(values[unexpected].unique())}")
    return pd.Series(pd.Categorical(values, categories=categories), index=values.index)
