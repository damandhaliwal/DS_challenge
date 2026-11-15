"""Smoke test: the project layout and raw inputs are where config expects."""

import pytest

import config


def test_raw_files_exist():
    assert config.INCIDENTS_RAW.is_file()
    assert config.KM_RAW.is_file()


@pytest.mark.slow
def test_main_runs():
    import main

    main.main()
    assert config.OUTPUTS_DIR.is_dir()
    assert config.PROCESSED_DIR.is_dir()
