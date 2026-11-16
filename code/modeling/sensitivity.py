"""Sensitivity runs: does a choice we made change the conclusions?

Each variant reruns the backtest of the chosen model (gradient boosting) with one
choice changed, and is compared with the main run on the same test incidents:

    main                  main inputs, target y_major, no safety weight
    with_leak_suspects    + run number present, + whole-line location, + door numbers
                          listed (features.py). A large gain means these fields matter,
                          which is a question for STM about when they are recorded,
                          not a reason to use them.
    without_flags         - door flag, - emergency brake. The main model assumes both
                          are set when the incident is reported; the data does not
                          say when. This is the model if neither is known that early.
    safety_weight_2       safety responses count 2x (the main setting until Decision 9)
    safety_weight_4       safety responses count 4x
    target_5plus          trained on y_5plus (the usual "disruption" threshold), without
                          a safety weight: the comparison model of the trade-off

Every variant is judged on the main question, finding major incidents (y_major), plus
the share of safety responses and delay minutes it puts in the top 10%.
"""

import pandas as pd
from sklearn.metrics import average_precision_score

from modeling.backtest import compare_predictions, run_backtest
from modeling.features import LEAK_SUSPECT_FEATURES, MAIN_FEATURES, TIMING_ASSUMED_FEATURES

MODEL = "gradient_boosting"

VARIANTS = {
    "main": {},
    "with_leak_suspects": {"feature_names": MAIN_FEATURES + LEAK_SUSPECT_FEATURES},
    "without_flags": {
        "feature_names": [name for name in MAIN_FEATURES if name not in TIMING_ASSUMED_FEATURES]
    },
    "safety_weight_2": {"safety_weight": 2.0},
    "safety_weight_4": {"safety_weight": 4.0},
    "target_5plus": {"target": "y_5plus", "safety_weight": 1.0},
}

# Columns of the backtest metrics kept in the summary.
SUMMARY_METRICS = [
    "y_major_caught_5%",
    "y_major_caught_10%",
    "safety_caught_10%",
    "minutes_caught_10%",
]


def run_sensitivity(labelled: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run every variant; return per-fold metrics, a summary and paired comparisons.

        metrics       one row per (variant, fold), backtest metrics
        summary       one row per variant: mean over folds, the pr_auc for y_major,
                      its paired difference with the main run, and the number of
                      folds where that difference is clearly positive / negative
        comparisons   per (variant, fold): pr_auc difference with the main run on
                      y_major, with its 95% interval
    """
    runs = {
        name: run_backtest(labelled, model_names=[MODEL], **settings)
        for name, settings in VARIANTS.items()
    }

    metrics = pd.concat(
        [run["metrics"].assign(variant=name) for name, run in runs.items()], ignore_index=True
    )
    main_predictions = runs["main"]["predictions"]
    comparisons = pd.concat(
        [
            compare_predictions(run["predictions"], main_predictions, labelled, label=f"{name} - main")
            .assign(variant=name)
            for name, run in runs.items() if name != "main"
        ],
        ignore_index=True,
    )

    summary = metrics.groupby("variant", sort=False)[SUMMARY_METRICS].mean()
    summary["pr_auc_y_major"] = [
        _mean_pr_auc_on_major(run["predictions"], labelled) for run in runs.values()
    ]
    summary["pr_auc_difference_vs_main"] = (
        comparisons.groupby("variant")["pr_auc_difference"].mean().reindex(summary.index).fillna(0.0)
    )
    # Folds where the 95% interval of the difference lies entirely above / below zero.
    by_variant = comparisons.groupby("variant")
    summary["folds_clearly_better"] = (
        by_variant["low"].apply(lambda low: (low > 0).sum()).reindex(summary.index).fillna(0).astype(int)
    )
    summary["folds_clearly_worse"] = (
        by_variant["high"].apply(lambda high: (high < 0).sum()).reindex(summary.index).fillna(0).astype(int)
    )
    return {"metrics": metrics, "summary": summary.reset_index(), "comparisons": comparisons}


def _mean_pr_auc_on_major(predictions: pd.DataFrame, labelled: pd.DataFrame) -> float:
    """Average over folds of the pr_auc for y_major, whatever the variant was trained on."""
    labels = labelled.set_index("incident_id")["y_major"]
    return predictions.groupby("fold").apply(
        lambda fold: average_precision_score(labels.loc[fold["incident_id"]], fold["risk"])
    ).mean()
