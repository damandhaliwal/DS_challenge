"""Other model families against the chosen model, on the rolling backtest only.

Would a different method have done better? Each alternative in
train.ALTERNATIVE_MODEL_NAMES (plain OLS, a small neural net, XGBoost) goes through the
same folds, tuning and test years as the ladder, next to the two models it challenges:

    gradient_boosting    the chosen model; every alternative is compared with it
    logistic             the explainable benchmark; OLS is also compared with it,
                         linear model against linear model

The holdout is not used. If an alternative were clearly better here, switching to it
would be a new decision, to be made on this evidence before any holdout rerun.

Every model is judged on the backtest metrics (backtest.evaluate) plus how well its
scores read as probabilities (calibrate.calibration_metrics), since OLS and the
neural net are not built to give calibrated scores.
"""

import pandas as pd

from modeling.backtest import FOLDS, compare_models, run_backtest
from modeling.calibrate import calibration_metrics
from modeling.train import ALTERNATIVE_MODEL_NAMES

CHOSEN_MODEL = "gradient_boosting"
BENCHMARK_MODEL = "logistic"
COMPARED_MODELS = [BENCHMARK_MODEL, CHOSEN_MODEL] + ALTERNATIVE_MODEL_NAMES

# (better, simpler) pairs for the paired bootstrap comparisons.
PAIRS = [(name, CHOSEN_MODEL) for name in ALTERNATIVE_MODEL_NAMES] + [("ols", BENCHMARK_MODEL)]

# Columns of the backtest metrics kept in the summary.
SUMMARY_METRICS = [
    "pr_auc",
    "roc_auc",
    "y_major_caught_5%",
    "y_major_caught_10%",
    "safety_caught_10%",
    "minutes_caught_10%",
]


def run_model_comparison(labelled: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Backtest every compared model; return the backtest tables plus two more.

        comparisons   per (pair, fold): pr_auc difference, with its 95% interval
        summary       one row per model: mean over folds of the backtest metrics and
                      of the calibration error, and, against gradient boosting, the
                      mean pr_auc difference and the number of folds where it is
                      clearly positive / negative
    """
    results = run_backtest(labelled, model_names=COMPARED_MODELS)
    results["comparisons"] = pd.concat(
        [compare_models(results, labelled, better, simpler) for better, simpler in PAIRS],
        ignore_index=True,
    )

    summary = results["metrics"].groupby("model", sort=False)[SUMMARY_METRICS].mean()
    calibration = _calibration_by_fold(results["predictions"], labelled)
    summary["calibration_error"] = calibration.groupby("model")["calibration_error"].mean()

    against_chosen = results["comparisons"][results["comparisons"]["comparison"].str.endswith(f" - {CHOSEN_MODEL}")]
    by_model = against_chosen.assign(model=against_chosen["comparison"].str.split(" - ").str[0]).groupby("model")
    summary["pr_auc_difference_vs_chosen"] = by_model["pr_auc_difference"].mean()
    # Folds where the 95% interval of the difference lies entirely above / below zero.
    summary["folds_clearly_better"] = by_model["low"].apply(lambda low: (low > 0).sum())
    summary["folds_clearly_worse"] = by_model["high"].apply(lambda high: (high < 0).sum())
    summary.loc[CHOSEN_MODEL, ["pr_auc_difference_vs_chosen", "folds_clearly_better", "folds_clearly_worse"]] = 0

    results["summary"] = summary.reset_index()
    return results


def _calibration_by_fold(predictions: pd.DataFrame, labelled: pd.DataFrame) -> pd.DataFrame:
    """Calibration metrics of each (fold, model) on the major label."""
    outcomes = labelled.set_index("incident_id")["y_major"]
    year = labelled["service_date"].dt.year

    rows = []
    for (fold, model), scored in predictions.groupby(["fold", "model"], sort=False):
        first_year, last_year = FOLDS[fold]
        training_rate = labelled.loc[year.between(first_year, last_year), "y_major"].mean()
        outcome = outcomes.loc[scored["incident_id"]].to_numpy(dtype=float)
        rows.append({"fold": fold, "model": model,
                     **calibration_metrics(scored["risk"].to_numpy(), outcome, training_rate)})
    return pd.DataFrame(rows)
