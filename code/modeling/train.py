"""The model ladder: from the simplest possible prediction to gradient boosting.

Each rung must beat the one below it to earn its place:
    base_rate            everyone gets the training share of major incidents
    symptom_line_table   share of major incidents per (symptom, line group), shrunk
                         toward the symptom's share when a cell is small
    logistic             logistic regression on every main input (one-hot categories)
    gradient_boosting    histogram gradient boosting, native categorical support

Alternatives, compared with the ladder in the backtest only (model_comparison.py):
    ols                  linear probability model: least squares on the same one-hot
                         inputs as the logistic, scores clipped to [0, 1]
    neural_net           small feed-forward network on the same one-hot inputs
    xgboost              XGBoost, histogram trees, native categorical support, tuned on
                         a grid that mirrors the gradient boosting one

Every model is trained with the sample weights from costs.py (1x in the main run; 2x
and 4x for safety responses in the sensitivity runs) and returns a risk score in
[0, 1] for each incident. The scores are risk *rankings*; whether they can be quoted
as probabilities is checked in a separate calibration step. Settings are fixed
defaults here; tuning happens inside the backtest, on the training years only.

Usage:
    model = make_model("logistic", feature_names)
    fit_model(model, X_train, y_train, sample_weight)
    risk = predict_risk(model, X_test)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

import config
from modeling.features import CATEGORICAL_FEATURES

MODEL_NAMES = ["base_rate", "symptom_line_table", "logistic", "gradient_boosting"]
ALTERNATIVE_MODEL_NAMES = ["ols", "neural_net", "xgboost"]

# Settings tried when tuning (backtest.py picks one per fold, on training years only).
# Small grids on purpose: about 1,600 positives cannot tell many settings apart.
PARAM_GRIDS = {
    "base_rate": {},
    "symptom_line_table": {"prior_strength": [2, 10, 50]},
    "logistic": {"model__C": [0.1, 1.0, 10.0, 100.0]},
    "gradient_boosting": {
        "learning_rate": [0.01, 0.03, 0.1],
        "max_leaf_nodes": [3, 7, 15],
        "min_samples_leaf": [100, 200],
    },
    "ols": {},
    "neural_net": {
        "model__hidden_layer_sizes": [(8,), (32,), (32, 16)],
        "model__alpha": [0.001, 0.1],
    },
    # Mirrors the gradient boosting grid: depth 2/3/4 holds at most 4/8/16 leaves, and a
    # leaf weight of 5/10 is roughly 100/200 incidents at a ~5% major rate (each
    # incident adds p(1 - p) to it).
    "xgboost": {
        "learning_rate": [0.01, 0.03, 0.1],
        "max_depth": [2, 3, 4],
        "min_child_weight": [5, 10],
    },
}


class SymptomLineTable(ClassifierMixin, BaseEstimator):
    """Weighted share of positives per (symptom, line group) cell.

    A small cell is pulled toward its symptom's share, and a rare symptom toward the
    overall share, by adding `prior_strength` pseudo-incidents at that share:
        cell share = (weighted positives + m x symptom share) / (weight + m)
    This is the transparent benchmark an operations team could build in a spreadsheet.
    """

    def __init__(self, prior_strength: float = 50.0):
        self.prior_strength = prior_strength

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        weight = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        frame = pd.DataFrame({
            "symptom": X["symptom"].astype(str).to_numpy(),
            "line_group": X["line_group"].astype(str).to_numpy(),
            "positive_weight": np.asarray(y, dtype=float) * weight,
            "weight": weight,
        })
        m = self.prior_strength
        self.overall_ = frame["positive_weight"].sum() / frame["weight"].sum()

        by_symptom = frame.groupby("symptom")[["positive_weight", "weight"]].sum()
        self.symptom_share_ = (by_symptom["positive_weight"] + m * self.overall_) / (by_symptom["weight"] + m)

        by_cell = frame.groupby(["symptom", "line_group"])[["positive_weight", "weight"]].sum()
        symptom_prior = self.symptom_share_.reindex(by_cell.index.get_level_values("symptom")).to_numpy()
        self.cell_share_ = (by_cell["positive_weight"] + m * symptom_prior) / (by_cell["weight"] + m)

        self.classes_ = np.array([False, True])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cells = pd.MultiIndex.from_arrays(
            [X["symptom"].astype(str), X["line_group"].astype(str)], names=["symptom", "line_group"]
        )
        # Unseen cell -> its symptom's share; unseen symptom -> the overall share.
        share = self.cell_share_.reindex(cells).to_numpy()
        symptom_fallback = self.symptom_share_.reindex(X["symptom"].astype(str)).to_numpy()
        share = np.where(np.isnan(share), symptom_fallback, share)
        share = np.where(np.isnan(share), self.overall_, share)
        return np.column_stack([1 - share, share])


class LinearProbability(ClassifierMixin, BaseEstimator):
    """Plain least squares on a 0/1 label (the linear probability model).

    The fitted value is read as a risk score. It can fall outside [0, 1], so it is
    clipped there; the clipped incidents sit at the very bottom or top of the ranking,
    where they would be anyway.
    """

    def fit(self, X, y, sample_weight=None):
        self.regression_ = LinearRegression().fit(X, np.asarray(y, dtype=float), sample_weight=sample_weight)
        self.classes_ = np.array([False, True])
        return self

    def predict_proba(self, X) -> np.ndarray:
        share = np.clip(self.regression_.predict(X), 0, 1)
        return np.column_stack([1 - share, share])


def make_model(name: str, feature_names: list[str]):
    """A new, unfitted model. `feature_names` are the columns it will be given."""
    categorical = [column for column in feature_names if column in CATEGORICAL_FEATURES]
    flags = [column for column in feature_names if column not in CATEGORICAL_FEATURES]

    if name == "base_rate":
        return DummyClassifier(strategy="prior")

    if name == "symptom_line_table":
        return SymptomLineTable()

    # Every level becomes a 0/1 column; a level absent from training is ignored.
    encode = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("flags", "passthrough", flags),
    ])

    if name == "logistic":
        return Pipeline([
            ("encode", encode),
            ("model", LogisticRegression(C=1.0, max_iter=2000)),
        ])

    if name == "ols":
        return Pipeline([("encode", encode), ("model", LinearProbability())])

    if name == "neural_net":
        # The inputs are all 0/1 columns, so no scaling is needed. Early stopping is off:
        # it would hold out a random tenth of the rows, not a later year.
        return Pipeline([
            ("encode", encode),
            ("model", MLPClassifier(
                hidden_layer_sizes=(32,), alpha=0.001, max_iter=500,
                early_stopping=False, random_state=config.RANDOM_SEED,
            )),
        ])

    if name == "gradient_boosting":
        # Small trees and a slow learning rate: about 1,600 positives cannot support
        # deep interactions. Categories are used natively ("from_dtype").
        return HistGradientBoostingClassifier(
            categorical_features="from_dtype",
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=config.RANDOM_SEED,
        )

    if name == "xgboost":
        # Same number of trees as gradient boosting, so the two differ by library and
        # tree shape, not by how long they train.
        return XGBClassifier(
            tree_method="hist",
            enable_categorical=True,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=5,
            reg_lambda=1.0,
            random_state=config.RANDOM_SEED,
        )

    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES + ALTERNATIVE_MODEL_NAMES}.")


def fit_model(model, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series):
    """Fit any model of the ladder with sample weights."""
    weight = np.asarray(sample_weight, dtype=float)
    if isinstance(model, Pipeline):
        # A pipeline passes the weights to its final step by name.
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": weight})
    return model.fit(X, y, sample_weight=weight)


def predict_risk(model, X: pd.DataFrame) -> pd.Series:
    """Risk score of each incident (the model's probability for the positive class)."""
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="risk")
