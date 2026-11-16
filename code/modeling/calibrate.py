"""Calibration: turning risk scores into probabilities that can be quoted.

A risk score ranks incidents, but it is not automatically a probability that is
true on average. Two things can break it: sample weights (safety responses counted
2x overstate the chance of a major incident; the main run no longer uses them, see
costs.py) and drift (the share of major incidents fell from 7.1% in 2021 to 4.6% in
2023, so a model trained on earlier years overstates it). Before anyone is told
"this incident has a 30% chance of becoming major", the score is checked, and mapped
to a calibrated probability if needed.

How, inside each backtest fold (no test-year data is ever used):
  1. The model is trained on the training years minus the last one, and scores that
     last training year.
  2. A calibrator learns the map score -> share of major incidents on that year,
     without weights (a probability must count every incident once).
  3. The model is refitted on all training years (as in the backtest) and the
     calibrator is applied to its test-year scores.

Two calibrators are compared with no calibration at all:
    platt       a logistic curve on the log-odds of the score: 2 parameters, smooth
    isotonic    any increasing step function: flexible, but needs more data
Both keep the order of the incidents, so the ranking metrics do not change (isotonic
can create ties, which capture() splits exactly).

Metrics on the test year:
    brier            mean squared error of the probability; lower is better
    brier_reference  the same for a constant forecast of the training share
    log_loss         penalises confident wrong forecasts
    mean_predicted vs observed_rate   calibration "in the large"
    calibration_error  average |predicted - observed| over 10 equal-size groups
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from modeling.backtest import FOLDS, tune
from modeling.costs import SAFETY_WEIGHT, sample_weights
from modeling.features import MAIN_FEATURES, build_features
from modeling.train import fit_model, make_model, predict_risk

CALIBRATION_METHODS = ["none", "platt", "isotonic"]
# Chosen on the backtest. With safety responses weighted 2x, the raw score overstated
# the rate and Platt fixed it (Decision 5). Without the weight (Decision 9) the raw
# score is already calibrated: lower calibration error than Platt in 2 of 3 folds
# (0.016 / 0.012 / 0.011 against 0.014 / 0.021 / 0.012). The rule set before that run
# was to drop Platt in that case, so no calibrator is used.
CALIBRATION_METHOD = "none"
RELIABILITY_GROUPS = 10
# Scores are clipped away from 0 and 1 before taking log-odds.
SCORE_CLIP = 1e-6


class PlattCalibrator:
    """Logistic regression of the outcome on the log-odds of the score."""

    def fit(self, scores: np.ndarray, outcome: np.ndarray) -> "PlattCalibrator":
        # A very weak penalty: two parameters, thousands of incidents.
        self.model_ = LogisticRegression(C=1e6).fit(_log_odds(scores), outcome)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(_log_odds(scores))[:, 1]


class IsotonicCalibrator:
    """Increasing step function from score to probability."""

    def fit(self, scores: np.ndarray, outcome: np.ndarray) -> "IsotonicCalibrator":
        self.model_ = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(scores, outcome)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self.model_.predict(scores)


class NoCalibration:
    """The raw score, for comparison."""

    def fit(self, scores, outcome) -> "NoCalibration":
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores)


CALIBRATORS = {"none": NoCalibration, "platt": PlattCalibrator, "isotonic": IsotonicCalibrator}


def run_calibration_backtest(
    labelled: pd.DataFrame,
    model_name: str = "gradient_boosting",
    feature_names: list[str] = MAIN_FEATURES,
    target: str = "y_major",
    safety_weight: float = SAFETY_WEIGHT,
) -> dict[str, pd.DataFrame]:
    """Calibrate one model in every backtest fold; see the module docstring.

    Returns:
        metrics       one row per (fold, method)
        reliability   predicted vs observed rate in 10 equal-size groups, per (fold, method)
        predictions   calibrated probability of every test incident, per (fold, method)
    """
    X = build_features(labelled)[feature_names]
    y = labelled[target]
    weight = sample_weights(labelled, safety_weight)
    year = labelled["service_date"].dt.year

    metrics, reliability, predictions = [], [], []
    for test_year, (first_year, last_year) in FOLDS.items():
        train = year.between(first_year, last_year)
        early = year.between(first_year, last_year - 1)
        validation = year == last_year
        test = year == test_year

        chosen = tune(model_name, feature_names, X[train], y[train], weight[train], year[train])

        # Steps 1-2: scores on the last training year from a model that has not seen it.
        early_model = make_model(model_name, feature_names).set_params(**chosen)
        fit_model(early_model, X[early], y[early], weight[early])
        validation_scores = predict_risk(early_model, X[validation]).to_numpy()

        # Step 3: the backtest's model, refitted on all training years.
        model = make_model(model_name, feature_names).set_params(**chosen)
        fit_model(model, X[train], y[train], weight[train])
        test_scores = predict_risk(model, X[test]).to_numpy()

        for method in CALIBRATION_METHODS:
            calibrator = CALIBRATORS[method]().fit(validation_scores, y[validation].to_numpy())
            probability = calibrator.transform(test_scores)
            outcome = y[test].to_numpy()

            metrics.append({"fold": test_year, "method": method,
                            **calibration_metrics(probability, outcome, y[train].mean())})
            reliability.append(reliability_table(probability, outcome).assign(fold=test_year, method=method))
            predictions.append(pd.DataFrame({
                "fold": test_year, "method": method,
                "incident_id": labelled.loc[test, "incident_id"].to_numpy(),
                "probability": probability,
            }))

    return {
        "metrics": pd.DataFrame(metrics),
        "reliability": pd.concat(reliability, ignore_index=True),
        "predictions": pd.concat(predictions, ignore_index=True),
    }


def calibration_metrics(probability: np.ndarray, outcome: np.ndarray, training_rate: float) -> dict:
    """Brier score (and its reference), log loss, mean predicted vs observed rate, and
    calibration error over 10 equal-size groups; see the module docstring."""
    groups = reliability_table(probability, outcome)
    return {
        "brier": brier_score_loss(outcome, probability),
        "brier_reference": brier_score_loss(outcome, np.full(len(outcome), training_rate)),
        "log_loss": log_loss(outcome, np.clip(probability, SCORE_CLIP, 1 - SCORE_CLIP)),
        "mean_predicted": probability.mean(),
        "observed_rate": outcome.mean(),
        "calibration_error": (groups["mean_predicted"] - groups["observed_rate"]).abs().mean(),
    }


def reliability_table(probability: np.ndarray, outcome: np.ndarray) -> pd.DataFrame:
    """Mean predicted probability vs observed rate in equal-size groups of incidents,
    from the lowest to the highest predicted probability."""
    order = np.argsort(probability, kind="stable")
    group = np.empty(len(order), dtype=int)
    group[order] = np.arange(len(order)) * RELIABILITY_GROUPS // len(order)
    return (
        pd.DataFrame({"group": group + 1, "predicted": probability, "outcome": outcome})
        .groupby("group")
        .agg(incidents=("outcome", "size"), mean_predicted=("predicted", "mean"),
             observed_rate=("outcome", "mean"))
        .reset_index()
    )


def _log_odds(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=float), SCORE_CLIP, 1 - SCORE_CLIP)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)
