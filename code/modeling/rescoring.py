"""A second look: score again the incidents still delaying service after 5 or 10 minutes.

The main model scores an incident once, when it is reported. Most incidents are over
within a few minutes, so a second score given only to the incidents that are still
delaying service works on a much smaller and much riskier pool:

    stage              pool: train incidents with a delay of at least   target
    after_5_minutes    5 min                                            y_30plus
    after_10_minutes   10 min                                           y_30plus

The pools come from the delay bucket, whose bounds are exact at 5 and 10 minutes.

The target is a delay of 30+ min, not the major label. The data does not say when an
evacuation was called, so by minute 5 or 10 a safety response may already have
happened and "predicting" it would be reading the past. A delay of 30+ min is always
still ahead at minute 10.

The inputs are the same report-time inputs as the main model: the only new
information is that the incident is still going on.

Rankings of each pool, on the backtest folds only (the holdout is not used):
    base_rate            every incident gets the pool's rate: what "still delayed"
                         says on its own
    first_score          the main model's risk score from report time, reused as is
    first_score_30plus   the main model trained at report time on all incidents for
                         the 30+ target, reused on the pool
    logistic             retrained on the pool for the 30+ target
    gradient_boosting    retrained on the pool for the 30+ target
Each retrained model is compared with both first scores on the same incidents. The
first score aims at the major label, so a gain over it mixes two things: the new
target and the new pool. first_score_30plus has the new target but not the pool, so
    retrained - first_score_30plus     what the pool adds (knowing the incident is
                                       still delaying service)
    first_score_30plus - first_score   what the target adds
"""

import pandas as pd

from modeling.backtest import FOLDS, compare_predictions, evaluate, run_backtest
from modeling.costs import ALERT_BUDGETS
from modeling.holdout import CHOSEN_MODEL

STAGES = {"after_5_minutes": 5, "after_10_minutes": 10}   # stage: minutes of delay so far
TARGET = "y_30plus"
FIRST_SCORE = "first_score"
FIRST_SCORE_30PLUS = "first_score_30plus"
RETRAINED_MODELS = ["base_rate", "logistic", "gradient_boosting"]
BUDGET = max(ALERT_BUDGETS)
CAUGHT = f"{TARGET}_caught_{BUDGET:.0%}"


def run_rescoring(labelled: pd.DataFrame, first_predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Backtest every stage; `first_predictions` are the main backtest's predictions.

    The report-time 30+ score is a backtest of the chosen model on every incident with
    the 30+ target, run here once for both stages.

        pools         per (stage, fold): size of the pool, and the share of the year's
                      incidents, major incidents and 30+ incidents it holds
        metrics       per (stage, fold, model): the backtest metrics on the pool, plus
                      the share of 30+ incidents among the riskiest 10% of the pool
        comparisons   per (stage, fold, model, reference): pr_auc of the model minus
                      the reference, with its 95% interval
        summary       per (stage, model): mean over folds, and against each first score
                      the mean pr_auc difference and the folds where it is clearly
                      positive / negative
    """
    first_scores = {
        FIRST_SCORE: first_predictions[first_predictions["model"] == CHOSEN_MODEL],
        FIRST_SCORE_30PLUS: run_backtest(labelled, target=TARGET, model_names=[CHOSEN_MODEL])["predictions"],
    }
    year = labelled["service_date"].dt.year

    pools, metrics, comparisons = [], [], []
    for stage, minutes in STAGES.items():
        pool = labelled[labelled["delay_floor_minutes"] >= minutes].reset_index(drop=True)
        retrained = run_backtest(pool, target=TARGET, model_names=RETRAINED_MODELS)
        reused = {
            name: scores[scores["incident_id"].isin(pool["incident_id"])]
            for name, scores in first_scores.items()
        }

        metrics.append(retrained["metrics"].assign(stage=stage))
        for fold in FOLDS:
            in_year = labelled[year == fold]
            test = pool[pool["service_date"].dt.year == fold]
            for name, scores in reused.items():
                risk = test["incident_id"].map(scores.set_index("incident_id")["risk"])
                metrics.append(pd.DataFrame([
                    {"fold": fold, "model": name, "stage": stage, **evaluate(test, risk, TARGET)}
                ]))
            pools.append({
                "stage": stage, "fold": fold,
                "incidents": len(test),
                "share_of_incidents": len(test) / len(in_year),
                "share_of_major": test["y_major"].sum() / in_year["y_major"].sum(),
                "share_of_30plus": test[TARGET].sum() / in_year[TARGET].sum(),
            })
        predictions = retrained["predictions"]
        pairs = [
            (name, predictions[predictions["model"] == name], reference)
            for name in RETRAINED_MODELS[1:] for reference in reused
        ] + [(FIRST_SCORE_30PLUS, reused[FIRST_SCORE_30PLUS], FIRST_SCORE)]
        for name, scores, reference in pairs:
            comparisons.append(
                compare_predictions(scores, reused[reference], pool,
                                    label=f"{name} - {reference}", target=TARGET)
                .assign(stage=stage, model=name, reference=reference)
            )

    metrics = pd.concat(metrics, ignore_index=True)
    # Share of 30+ incidents among the riskiest 10% of the pool.
    metrics[f"precision_{BUDGET:.0%}"] = metrics[CAUGHT] * metrics["base_rate"] / BUDGET
    comparisons = pd.concat(comparisons, ignore_index=True)
    return {
        "pools": pd.DataFrame(pools),
        "metrics": metrics,
        "comparisons": comparisons,
        "summary": _summary(metrics, comparisons),
    }


def _summary(metrics: pd.DataFrame, comparisons: pd.DataFrame) -> pd.DataFrame:
    """Mean over folds per (stage, model), with the paired results against each first score."""
    columns = ["incidents", "base_rate", "pr_auc", CAUGHT, f"precision_{BUDGET:.0%}"]
    summary = metrics.groupby(["stage", "model"], sort=False)[columns].mean()

    for reference, against in comparisons.groupby("reference"):
        by_model = against.groupby(["stage", "model"])
        summary[f"pr_auc_difference_vs_{reference}"] = by_model["pr_auc_difference"].mean()
        # Folds where the 95% interval of the difference lies entirely above / below zero.
        summary[f"folds_clearly_better_vs_{reference}"] = by_model["low"].apply(lambda low: (low > 0).sum())
        summary[f"folds_clearly_worse_vs_{reference}"] = by_model["high"].apply(lambda high: (high < 0).sum())
    return summary.reset_index()
