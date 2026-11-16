"""Tests for modeling/train.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid

from cleaning.clean_incidents import clean_incidents
from cleaning.load import read_raw_incidents
from modeling.costs import sample_weights
from modeling.features import LEAK_SUSPECT_FEATURES, MAIN_FEATURES, build_features
from modeling.target import add_targets, model_population
from modeling.train import (
    ALTERNATIVE_MODEL_NAMES,
    MODEL_NAMES,
    PARAM_GRIDS,
    LinearProbability,
    SymptomLineTable,
    fit_model,
    make_model,
    predict_risk,
)


@pytest.fixture(scope="module")
def year_2019():
    labelled = add_targets(model_population(clean_incidents(read_raw_incidents())))
    labelled = labelled[labelled["service_date"].dt.year == 2019]
    return build_features(labelled), labelled["y_major"], sample_weights(labelled)


@pytest.mark.parametrize("name", MODEL_NAMES + ALTERNATIVE_MODEL_NAMES)
def test_every_model_fits_and_scores_each_incident(year_2019, name):
    X, y, weight = year_2019
    model = fit_model(make_model(name, MAIN_FEATURES), X[MAIN_FEATURES], y, weight)
    risk = predict_risk(model, X[MAIN_FEATURES])
    assert risk.index.equals(X.index)
    assert risk.between(0, 1).all()


@pytest.mark.parametrize("name", ["logistic", "gradient_boosting"])
def test_models_accept_the_leak_suspects_for_the_sensitivity_run(year_2019, name):
    X, y, weight = year_2019
    columns = MAIN_FEATURES + LEAK_SUSPECT_FEATURES
    model = fit_model(make_model(name, columns), X[columns], y, weight)
    assert predict_risk(model, X[columns]).between(0, 1).all()


@pytest.mark.parametrize("name", MODEL_NAMES + ALTERNATIVE_MODEL_NAMES)
def test_every_tuning_setting_applies_to_its_model(name):
    for setting in ParameterGrid(PARAM_GRIDS[name]):
        make_model(name, MAIN_FEATURES).set_params(**setting)


def test_linear_probability_is_least_squares_clipped_to_0_1():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    # Least squares line: 0.4x - 0.1, so x = 0 gives -0.1 (clipped to 0).
    risk = LinearProbability().fit(X, y).predict_proba(X)[:, 1]
    assert risk == pytest.approx([0.0, 0.3, 0.7, 1.0])


def test_unknown_model_name_fails():
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("random_forest", MAIN_FEATURES)


# ---------------------------------------------------------------------------
# The lookup table, checked by hand on a tiny example
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny():
    X = pd.DataFrame({
        "symptom": ["fire"] * 4 + ["customer"] * 4,
        "line_group": ["green", "green", "orange", "orange", "green", "green", "green", "green"],
    })
    y = pd.Series([1, 1, 0, 0, 0, 0, 0, 1])
    return X, y


def test_table_shrinks_small_cells_toward_their_symptom(tiny):
    X, y = tiny
    table = SymptomLineTable(prior_strength=2).fit(X, y)
    overall = 3 / 8
    fire = (2 + 2 * overall) / (4 + 2)
    fire_green = (2 + 2 * fire) / (2 + 2)
    assert table.overall_ == pytest.approx(overall)
    assert table.symptom_share_["fire"] == pytest.approx(fire)
    assert table.cell_share_[("fire", "green")] == pytest.approx(fire_green)


def test_table_falls_back_for_unseen_cells_and_symptoms(tiny):
    X, y = tiny
    table = SymptomLineTable(prior_strength=2).fit(X, y)
    new = pd.DataFrame({"symptom": ["fire", "smoke"], "line_group": ["blue", "green"]})
    risk = table.predict_proba(new)[:, 1]
    assert risk[0] == pytest.approx(table.symptom_share_["fire"])   # unseen line
    assert risk[1] == pytest.approx(table.overall_)                 # unseen symptom


def test_table_uses_the_sample_weights(tiny):
    X, y = tiny
    unweighted = SymptomLineTable(prior_strength=2).fit(X, y)
    weighted = SymptomLineTable(prior_strength=2).fit(X, y, sample_weight=1 + y)  # positives x2
    assert weighted.overall_ == pytest.approx(6 / 11)
    assert weighted.overall_ > unweighted.overall_
