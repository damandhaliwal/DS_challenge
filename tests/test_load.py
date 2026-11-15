"""Tests for cleaning/load.py.

Two groups:
    * raw readers: the published files are read faithfully, nothing is changed
    * load / override: the cleaned file is built when missing, reused when present,
      and rebuilt when override=True
"""

import pandas as pd
import pytest

import config
from cleaning import load
from cleaning.load import INCIDENT_COLUMNS, KM_COLUMNS, read_raw_incidents, read_raw_km


# ---------------------------------------------------------------------------
# Raw readers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def raw_incidents():
    return read_raw_incidents()


@pytest.fixture(scope="module")
def raw_km():
    return read_raw_km()


def test_incidents_shape_and_names(raw_incidents):
    assert raw_incidents.shape == (45_561, 26)
    assert list(raw_incidents.columns) == list(INCIDENT_COLUMNS.values())


def test_km_shape_and_names(raw_km):
    assert raw_km.shape == (8_472, 11)
    assert list(raw_km.columns) == list(KM_COLUMNS.values())


def test_every_column_is_text(raw_incidents, raw_km):
    for frame in (raw_incidents, raw_km):
        assert all(pd.api.types.is_string_dtype(dtype) for dtype in frame.dtypes)


def test_placeholders_are_kept_as_text(raw_incidents):
    # The byte-order mark must not leak into the first value, and placeholders must
    # survive as text rather than being turned into NaN by pandas.
    assert raw_incidents["incident_id"].str.match(r"^[TS]\d{8}$").all()
    assert raw_incidents.isna().sum().sum() == 0
    assert (raw_incidents["evacuation"] == "#").sum() == 44_190
    assert (raw_incidents["primary_cause"] == "#N/A").sum() == 482


def test_decimal_comma_values_stay_in_one_field(raw_km):
    # A quoted "16315,92" must not be split across two columns by the "," separator.
    assert raw_km["planned_km"].str.fullmatch(r"\d+(,\d+)?").all()


def test_unexpected_header_fails_loudly(tmp_path):
    bad_file = tmp_path / "bad.csv"
    bad_file.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected columns"):
        load._read_as_text(bad_file, {"a": "x", "c": "y"}, sep=",")


# ---------------------------------------------------------------------------
# load_incidents / load_km with the override switch
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_cleaning(tmp_path, monkeypatch):
    """Point the cleaned-file path at a temporary folder and replace the real cleaning
    with a stand-in that records how often it runs. This tests the caching logic on its
    own, independently of the cleaning rules (which have their own tests)."""
    monkeypatch.setattr(config, "INCIDENTS_CLEAN", tmp_path / "processed" / "incidents.parquet")

    calls = []

    def stand_in(raw: pd.DataFrame) -> pd.DataFrame:
        calls.append(len(raw))
        return raw.head(3)

    monkeypatch.setattr(load, "clean_incidents", stand_in)
    return calls


def test_builds_and_saves_when_no_cleaned_file(fake_cleaning):
    result = load.load_incidents()
    assert fake_cleaning == [45_561]          # cleaning ran once, on the full raw file
    assert config.INCIDENTS_CLEAN.exists()    # and its output was saved
    assert len(result) == 3


def test_reuses_saved_file_without_cleaning_again(fake_cleaning):
    first = load.load_incidents()
    second = load.load_incidents()
    assert len(fake_cleaning) == 1            # the second call read the saved file
    pd.testing.assert_frame_equal(first, second)


def test_override_reruns_the_cleaning(fake_cleaning):
    load.load_incidents()
    load.load_incidents(override=True)
    assert len(fake_cleaning) == 2
