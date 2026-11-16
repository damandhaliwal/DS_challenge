"""Cleaning rules for the metro incidents file.

Input: the raw incidents as read by cleaning/load.py (English column names, all text).
Output: one analysis-ready row per incident.

Each rule comes from a finding in the exploration notebooks (01 and 04); the notebook
section is cited next to each step.
"""

import re

import pandas as pd

import config

# ---------------------------------------------------------------------------
# Step 1: placeholders
# ---------------------------------------------------------------------------
# The text the STM writes instead of leaving a field empty, per column
# (notebook 01, "Placeholders" section). They hide two different meanings:
#   * unknown: the value should exist but was not recorded
#       causes and symptom "#N/A", line "Non affecté"
#   * not applicable: the field does not apply to this incident
#       run, car and door numbers "#" and rolling stock "Non affecté". These are
#       always placeholders on station incidents (type "S") and the rolling stock is
#       only filled in for the two rolling-stock secondary causes (MPM-10, MR-73).
# Both become a missing value. "Not applicable" stays recoverable from the incident
# type and the cause, so a second kind of missing value would add nothing.
PLACEHOLDERS = {
    "primary_cause": "#N/A",
    "secondary_cause": "#N/A",
    "symptom": "#N/A",
    "line": "Non affecté",
    "run_number": "#",
    "car_number": "#",
    "door_numbers": "#",
    "rolling_stock": "Non affecté",
}

# Evacuation is the exception: "#" is not a missing value but the answer "there was no
# evacuation" (97% of incidents). It gets an explicit category instead, so the column
# has no missing values and "no evacuation" cannot be mistaken for "not recorded".
EVACUATION_PLACEHOLDER = "#"
NO_EVACUATION = "Aucune"

# ---------------------------------------------------------------------------
# Step 2: dates and times
# ---------------------------------------------------------------------------
# The raw date and time fields that step 2 replaces with service_date, start_at,
# end_at and recorded_minutes. Weekday and month are derived again from
# service_date when they are needed as features.
RAW_DATE_TIME_COLUMNS = [
    "start_time",
    "end_time",
    "year",
    "year_month_label",
    "month",
    "day_of_month",
    "weekday",
    "calendar_date",
]

# ---------------------------------------------------------------------------
# Step 3: lines
# ---------------------------------------------------------------------------
# One raw value is an Excel artefact: "2, 4" was read as a date and saved as
# "02-avr" (2 April). It is the only value that is neither a colour nor a number list.
EXCEL_DATE_ARTEFACTS = {"02-avr": "Ligne 2, 4"}

# ---------------------------------------------------------------------------
# Step 4: locations
# ---------------------------------------------------------------------------
# Kinds of place mixed in the raw location column (notebook 01, section 8; their
# severity differs a lot, notebook 04, section 4).
STATION = "station"
TRANSFER_PLATFORM = "transfer_platform"          # "Berri-UQAM - Ligne 1, 2"
BETWEEN_STATIONS = "between_stations"            # "Atwater/Guy-Concordia", "Cartier - Henri-Bourassa"
WHOLE_LINE_OR_NETWORK = "whole_line_or_network"  # "Ligne 1", "Réseau"
ANCILLARY_STRUCTURE = "ancillary_structure"      # "... PR"
FACILITY = "facility"                            # yards, shops, tail tracks, control centres

# First words of facility names that are not written in capitals.
FACILITY_FIRST_WORDS = {"Garage", "Arrière", "Centre", "Atelier"}

# Other spellings of a station name -> the spelling used by the station itself.
STATION_ALIASES = {
    "Berri L-2": "Berri-UQAM",          # the orange-line part of Berri-UQAM
    "Place-d'Armes": "Place-D'Armes",
    "l'Assomption": "L'Assomption",
    "St-Henri": "Place St-Henri",
}

# Locations the text rules would misread, set by hand:
# raw value -> (kind, station_from, station_to).
LOCATION_OVERRIDES = {
    "G. H-Bour. (St-Charles)": (FACILITY, None, None),   # Henri-Bourassa garage (St-Charles)
    "Duvernay": (ANCILLARY_STRUCTURE, None, None),       # not a metro station; "Duvernay PR" is
    "BR42": (None, None, None),                          # unexplained code, 1 row
}

# ---------------------------------------------------------------------------
# Step 5: flags
# ---------------------------------------------------------------------------
# Columns recorded as "0" / "1".
FLAG_COLUMNS = [
    "possible_damage",
    "emergency_brake",
    "door_involved",
    "emergency_team",
    "traction_power_cut",
]

# ---------------------------------------------------------------------------
# Step 6: delay bucket
# ---------------------------------------------------------------------------
# The binned service delay, shortest first, with the lower bound of each bin in
# minutes. Every severity threshold we use (5, 10, 15, 30 min) is one of these lower
# bounds, so "delay of at least k minutes" is exact even though the delay is binned.
DELAY_BUCKET_FLOORS = {
    "02 min et moins": 0,
    "03 à 04 min": 3,
    "05 à 09 min": 5,
    "10 à 14 min": 10,
    "15 à 19 min": 15,
    "20 à 24 min": 20,
    "25 à 29 min": 25,
    "30 min et plus": 30,
}
# Lower bound -> upper bound of each bucket, in whole minutes. The 30+ bucket is
# open-ended.
DELAY_BUCKET_CEILINGS = {0: 2, 3: 4, 5: 9, 10: 14, 15: 19, 20: 24, 25: 29, 30: float("inf")}


def clean_incidents(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply every cleaning rule to the raw incidents and return the cleaned table.

    Steps:
        1. Placeholders ("#", "#N/A", "Non affecté") -> missing values   (done)
        2. Service date + start / end timestamps from the service-day clock,
           including the "02:00" rows that should read "26:00"            (done)
        3. Line: sorted line set, number of lines, "02-avr" artefact      (done)
        4. Location: location kind and the stations at each end           (done)
        5. Flags: 0/1 text -> booleans                                    (done)
        6. Delay bucket as an ordered category, plus its lower bound      (done)
        7. Delay minutes: recorded duration when it agrees with the bucket (done)
    """
    return (
        raw.pipe(replace_placeholders)
        .pipe(build_timestamps)
        .pipe(normalise_lines)
        .pipe(classify_locations)
        .pipe(convert_flags)
        .pipe(order_delay_buckets)
        .pipe(count_delay_minutes)
    )


def replace_placeholders(incidents: pd.DataFrame) -> pd.DataFrame:
    """Step 1: turn placeholder text into missing values, and "#" evacuation into "Aucune"."""
    cleaned = incidents.copy()

    for column, placeholder in PLACEHOLDERS.items():
        cleaned[column] = cleaned[column].mask(cleaned[column] == placeholder)

    cleaned["evacuation"] = cleaned["evacuation"].replace(EVACUATION_PLACEHOLDER, NO_EVACUATION)
    return cleaned


def build_timestamps(incidents: pd.DataFrame) -> pd.DataFrame:
    """Step 2: one service date and two real timestamps per incident.

    The raw file uses two clocks (notebook 01, "Calendar consistency" section):
      * the service-day clock: year / month / day are the service day, and times after
        midnight keep counting past 24 ("24:30:00" = 00:30 the next calendar day).
      * the real clock: calendar_date is the real date of the start. On 60 rows the
        start was written on the real clock instead ("02:00" with the next calendar
        date), and so was their end time.

    calendar_date + (start clock modulo 24 h) gives the real start in both cases, so
    no row needs a special case. The end is then the start plus the clock difference,
    which is valid because both times of a row are always written on the same clock.

    recorded_minutes is the end minus the start. For train incidents it is the service
    delay itself: it falls inside the published delay bucket for 97.8% of them (step 7).
    For station incidents it is not (4.5% agree). It is known only after the fact, so it
    must never be a model input.
    """
    service_date = pd.to_datetime(
        incidents["year"] + "-" + incidents["month"] + "-" + incidents["day_of_month"],
        format="%Y-%m-%d",
    )
    calendar_date = pd.to_datetime(incidents["calendar_date"], format="%Y-%m-%d")
    start_clock = _clock_to_timedelta(incidents["start_time"])
    end_clock = _clock_to_timedelta(incidents["end_time"])

    _check_date_time_assumptions(incidents, service_date, calendar_date, start_clock, end_clock)

    start_at = calendar_date + start_clock % pd.Timedelta(days=1)
    end_at = start_at + (end_clock - start_clock)

    return incidents.drop(columns=RAW_DATE_TIME_COLUMNS).assign(
        service_date=service_date,
        start_at=start_at,
        end_at=end_at,
        recorded_minutes=(end_at - start_at) // pd.Timedelta(minutes=1),
    )


def _clock_to_timedelta(clock: pd.Series) -> pd.Series:
    """'HH:MM' or 'HH:MM:SS' (hours can exceed 23) -> time since midnight.

    The seconds are always ":00" in the raw file, so only hours and minutes are read.
    """
    hours = clock.str[:2].astype(int)
    minutes = clock.str[3:5].astype(int)
    return pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")


def _check_date_time_assumptions(
    incidents: pd.DataFrame,
    service_date: pd.Series,
    calendar_date: pd.Series,
    start_clock: pd.Series,
    end_clock: pd.Series,
) -> None:
    """Fail loudly if a new raw file breaks one of the facts that step 2 relies on."""
    # The incident ID encodes the service date as DDMMYY, e.g. S01011901 -> 2019-01-01.
    id_date = pd.to_datetime(incidents["incident_id"].str[1:7], format="%d%m%y")
    days_after_service_date = (calendar_date - service_date).dt.days
    problems = {
        "the ID date differs from the service date": id_date != service_date,
        "the calendar date is not the service date or the day after": ~(
            days_after_service_date.isin([0, 1])
        ),
        # A start past 24:00 happens after midnight, so its calendar date must be the
        # next day; otherwise the modulo in build_timestamps would lose a day.
        "the start is past 24:00 but the calendar date is the service date": (
            (start_clock >= pd.Timedelta(days=1)) & (days_after_service_date == 0)
        ),
        "the end clock is before the start clock": end_clock < start_clock,
    }
    for problem, rows in problems.items():
        if rows.any():
            raise ValueError(f"{rows.sum()} incidents where {problem}.")


def normalise_lines(incidents: pd.DataFrame) -> pd.DataFrame:
    """Step 3: one spelling per set of lines, plus the number of lines involved.

    The raw file spells the same set of lines in several ways (notebook 01, "Ligne
    variants" section): "Ligne verte" for one line, and "Ligne 1, 2" or "Ligne 2, 1"
    for the same two lines. Every value becomes the line names in line-number order,
    joined by "+": "green", "green+orange", "green+orange+yellow".
    Unknown lines (missing since step 1) stay missing.
    """
    line_numbers = (
        incidents["line"].replace(EXCEL_DATE_ARTEFACTS).map(_line_numbers, na_action="ignore")
    )
    return incidents.assign(
        line=line_numbers.map(
            lambda numbers: "+".join(config.LINE_NAMES[n] for n in numbers), na_action="ignore"
        ),
        n_lines=line_numbers.map(len, na_action="ignore").astype("Int64"),
    )


def _line_numbers(raw_line: str) -> tuple[str, ...]:
    """'Ligne verte' -> ('1',);  'Ligne 2, 1' -> ('1', '2'). Unknown spellings fail loudly."""
    if raw_line in config.FRENCH_LINE_NUMBERS:
        return (config.FRENCH_LINE_NUMBERS[raw_line],)

    numbers = raw_line.removeprefix("Ligne ").split(", ")
    if not raw_line.startswith("Ligne ") or not set(numbers) <= set(config.LINE_NAMES):
        raise ValueError(f"Unknown line value: {raw_line!r}")
    return tuple(sorted(set(numbers), key=int))


def classify_locations(incidents: pd.DataFrame) -> pd.DataFrame:
    """Step 4: add the kind of place and, where there is one, the stations at each end.

    The raw location column mixes stations, transfer platforms, segments between two
    stations, whole lines, ancillary structures and facilities (notebook 01, section
    8). The model needs simpler fields:
      * location_kind: one of the six kinds above
      * station_from / station_to: the stations' single spelling. For a segment
        between two stations they are its two ends, in the order the raw value names
        them (the data does not say which way the train was going). For a station or
        a transfer platform both are that station. Missing for the other kinds, which
        are not at a station.
    The raw location is kept for reference.
    """
    # Classify each distinct value once (230 of them) rather than each row.
    lookup = {location: _classify_location(location) for location in incidents["location"].unique()}
    classified = incidents["location"].map(lookup)
    return incidents.assign(
        location_kind=classified.str[0],
        station_from=classified.str[1],
        station_to=classified.str[2],
    )


def _classify_location(location: str) -> tuple[str | None, str | None, str | None]:
    """'Atwater/Guy-Concordia' -> ('between_stations', 'Atwater', 'Guy-Concordia'), etc.

    The order of the rules matters: facility codes in capitals can contain "/"
    ("INTER.ST-MICHEL/PIE-IX PL5"), so they are recognised before segments.
    """
    if location in LOCATION_OVERRIDES:
        return LOCATION_OVERRIDES[location]
    if location.startswith("Ligne") or location == "Réseau":
        return WHOLE_LINE_OR_NETWORK, None, None
    if " - Ligne" in location:
        station = _canonical_station(location.split(" - Ligne")[0])
        return TRANSFER_PLATFORM, station, station
    if location.isupper() or location.split()[0] in FACILITY_FIRST_WORDS:
        return FACILITY, None, None
    if "/" in location or " - " in location:
        first, second = re.split(r"/| - ", location)
        return BETWEEN_STATIONS, _canonical_station(first), _canonical_station(second)
    if location.endswith(" PR"):
        return ANCILLARY_STRUCTURE, None, None
    station = _canonical_station(location)
    return STATION, station, station


def _canonical_station(name: str) -> str:
    return STATION_ALIASES.get(name, name)


def convert_flags(incidents: pd.DataFrame) -> pd.DataFrame:
    """Step 5: "0" / "1" flags -> booleans.

    The door flag and the door-number column disagree (notebook 01, "Porte" section):
    1,219 train incidents list door numbers with the flag at 0, and 1,335 have the
    flag at 1 with no door number. The flag is kept as recorded. Door numbers may be
    written in after the doors are inspected, so they stay a separate column and are
    tested as a leak suspect (modeling/features.py, documentation.md, Decision 9).
    Station incidents never have either.
    """
    unexpected = ~incidents[FLAG_COLUMNS].isin(["0", "1"])
    if unexpected.any().any():
        raise ValueError(f"Flag values other than 0/1 in: {list(unexpected.columns[unexpected.any()])}")

    converted = incidents.copy()
    converted[FLAG_COLUMNS] = incidents[FLAG_COLUMNS] == "1"
    return converted


def order_delay_buckets(incidents: pd.DataFrame) -> pd.DataFrame:
    """Step 6: the delay bucket as an ordered category, plus its lower bound in minutes.

    As text, "10 à 14 min" sorts before "03 à 04 min"; as an ordered category the
    buckets sort and compare by length. delay_floor_minutes is the same information
    as a number, which keeps targets readable: delay_floor_minutes >= 30.
    Both are outcomes, known only after the incident, so never model inputs.
    """
    unknown = ~incidents["delay_bucket"].isin(list(DELAY_BUCKET_FLOORS))
    if unknown.any():
        raise ValueError(f"Unknown delay buckets: {sorted(incidents.loc[unknown, 'delay_bucket'].unique())}")

    return incidents.assign(
        delay_bucket=pd.Categorical(
            incidents["delay_bucket"], categories=list(DELAY_BUCKET_FLOORS), ordered=True
        ),
        delay_floor_minutes=incidents["delay_bucket"].map(DELAY_BUCKET_FLOORS).astype(int),
    )


def count_delay_minutes(incidents: pd.DataFrame) -> pd.DataFrame:
    """Step 7: the service delay of each incident in minutes.

    For a train incident the recorded duration (step 2) is the delay, and it falls
    inside the published bucket for 97.8% of them. When it does, it is used as is.
    When it does not (a typo such as 1,605 min in the "2 min and less" bucket), the
    bucket is trusted and its lower bound is used. The 30+ bucket has no upper bound,
    so its recorded durations are kept (median 49 min, longest 1,170 min).
    Station incidents never delay service themselves: their recorded duration is not a
    delay, so they keep the bucket's lower bound (always 0).

    The labels still use delay_floor_minutes, which is exact at every threshold.
    delay_minutes is used only to count the delay burden (modeling/costs.py).
    It is an outcome, so never a model input.
    """
    floor = incidents["delay_floor_minutes"]
    ceiling = floor.map(DELAY_BUCKET_CEILINGS)
    recorded = incidents["recorded_minutes"]
    agrees = (incidents["incident_type"] == "T") & recorded.between(floor, ceiling)
    return incidents.assign(delay_minutes=recorded.where(agrees, floor).astype(int))
