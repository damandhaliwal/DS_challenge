"""Rolling yearly backtest: train on the past, test on the next year, three times.

    fold   train years    test year
    2021   2019-2020      2021
    2022   2019-2021      2022
    2023   2019-2022      2023

The final holdout (train 2019-2023, test 2024-2025) is not part of the backtest. It is
run once, after the model has been chosen on these folds, so its result cannot steer
any choice.

Inside each fold, every model is tuned on the training years only: the last training
year is held back for validation, each setting in train.PARAM_GRIDS is scored on it,
and the best setting is refitted on all training years. The test year is never seen
until the final scoring.

Metrics, per fold and model (the "major" label is the target, y_major by default):
    pr_auc              average precision for the target; the base rate is its floor
    roc_auc             ranking quality, 0.5 = random
    {target}_caught_5% / _10%   share of target incidents in the top 5% / 10% of risk
    safety_caught_*     share of safety responses in the same top group
    minutes_caught_*    share of delay minutes (lower bound) in the same top group
Each pr_auc and 10% capture comes with a 95% interval from a bootstrap that resamples
service days rather than incidents, because incidents on the same day are related.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

import config
from modeling.costs import ALERT_BUDGETS, SAFETY_WEIGHT, delay_minutes, sample_weights
from modeling.features import MAIN_FEATURES, build_features
from modeling.train import MODEL_NAMES, PARAM_GRIDS, fit_model, make_model, predict_risk

FOLDS = {2021: (2019, 2020), 2022: (2019, 2021), 2023: (2019, 2022)}   # test year: train years
BOOTSTRAP_SAMPLES = 500


def run_backtest(
    labelled: pd.DataFrame,
    feature_names: list[str] = MAIN_FEATURES,
    target: str = "y_major",
    safety_weight: float = SAFETY_WEIGHT,
    model_names: list[str] = MODEL_NAMES,
) -> dict[str, pd.DataFrame]:
    """Run every fold for every model.

    Returns three tables:
        metrics      one row per (fold, model), with bootstrap intervals
        settings     the setting chosen by tuning, per (fold, model)
        predictions  the risk score of every test incident, per (fold, model)
    """
    X = build_features(labelled)[feature_names]
    y = labelled[target]
    weight = sample_weights(labelled, safety_weight)
    year = labelled["service_date"].dt.year

    metrics, settings, predictions = [], [], []
    for test_year, (first_year, last_year) in FOLDS.items():
        train = year.between(first_year, last_year)
        test = year == test_year
        for name in model_names:
            chosen = tune(name, feature_names, X[train], y[train], weight[train], year[train])
            model = make_model(name, feature_names).set_params(**chosen)
            fit_model(model, X[train], y[train], weight[train])
            risk = predict_risk(model, X[test])

            metrics.append({"fold": test_year, "model": name,
                            **evaluate(labelled[test], risk, target)})
            settings.append({"fold": test_year, "model": name, **chosen})
            predictions.append(pd.DataFrame({
                "fold": test_year, "model": name,
                "incident_id": labelled.loc[test, "incident_id"], "risk": risk,
            }))

    return {
        "metrics": pd.DataFrame(metrics),
        "settings": pd.DataFrame(settings),
        "predictions": pd.concat(predictions, ignore_index=True),
    }


def tune(name, feature_names, X, y, weight, year) -> dict:
    """Best setting from PARAM_GRIDS, judged on the last training year.

    Score: average precision weighted like the training (safety responses count
    safety_weight times), so tuning follows the same priority as the model.
    """
    grid = list(ParameterGrid(PARAM_GRIDS[name]))
    if len(grid) == 1:
        return grid[0]

    validation = year == year.max()
    scores = []
    for setting in grid:
        model = make_model(name, feature_names).set_params(**setting)
        fit_model(model, X[~validation], y[~validation], weight[~validation])
        risk = predict_risk(model, X[validation])
        scores.append(average_precision_score(y[validation], risk, sample_weight=weight[validation]))
    return grid[int(np.argmax(scores))]


def evaluate(test: pd.DataFrame, risk: pd.Series, target: str) -> dict:
    """Every metric for one (fold, model), with bootstrap intervals for the key two."""
    minutes = delay_minutes(test)

    result = {
        "incidents": len(test),
        "base_rate": test[target].mean(),
        "pr_auc": average_precision_score(test[target], risk),
        "roc_auc": roc_auc_score(test[target], risk),
    }
    # Major incidents are always reported, so a model trained on another target
    # (e.g. y_5plus) can be compared with the main one on the same footing.
    caught_labels = list(dict.fromkeys([target, "y_major", "y_safety"]))
    for budget in ALERT_BUDGETS:
        label = f"{budget:.0%}"
        for caught in caught_labels:
            name = "safety" if caught == "y_safety" else caught
            result[f"{name}_caught_{label}"] = capture(risk, test[caught], budget)
        result[f"minutes_caught_{label}"] = capture(risk, minutes, budget)

    low_high = _bootstrap_by_day(test, risk, target, budget=max(ALERT_BUDGETS))
    return result | low_high


def capture(risk: pd.Series, values: pd.Series, budget: float) -> float:
    """Share of `values` (a 0/1 label or delay minutes) in the top `budget` of risk.

    When the cut-off falls inside a group of tied scores, each tied incident is
    counted for the same fraction of a slot. This makes the result exact and
    deterministic; breaking ties by row order (which is chronological) or at random
    would make a model with many ties, like the base rate, look better or worse by
    chance.
    """
    groups = (
        pd.DataFrame({"risk": np.asarray(risk), "value": np.asarray(values, dtype=float)})
        .groupby("risk")["value"].agg(["sum", "size"])
        .sort_index(ascending=False)
    )
    slots = budget * len(risk)
    slots_before = groups["size"].cumsum() - groups["size"]
    taken = (slots - slots_before).clip(lower=0, upper=groups["size"])
    return (groups["sum"] * taken / groups["size"]).sum() / groups["sum"].sum()


def _bootstrap_by_day(test: pd.DataFrame, risk: pd.Series, target: str, budget: float) -> dict:
    """95% intervals for pr_auc and the capture at `budget`, resampling service days."""
    labels, scores = test[target].to_numpy(), risk.to_numpy()

    pr_aucs, captures = [], []
    for sample in _day_resamples(test):
        pr_aucs.append(average_precision_score(labels[sample], scores[sample]))
        captures.append(capture(scores[sample], labels[sample], budget))

    label = f"{target}_caught_{budget:.0%}"
    return {
        "pr_auc_low": np.percentile(pr_aucs, 2.5),
        "pr_auc_high": np.percentile(pr_aucs, 97.5),
        f"{label}_low": np.percentile(captures, 2.5),
        f"{label}_high": np.percentile(captures, 97.5),
    }


def compare_models(
    results: dict[str, pd.DataFrame], labelled: pd.DataFrame, better: str, simpler: str,
    target: str = "y_major",
) -> pd.DataFrame:
    """Paired comparison of two models of one backtest run (see compare_predictions)."""
    predictions = results["predictions"]
    return compare_predictions(
        predictions[predictions["model"] == better],
        predictions[predictions["model"] == simpler],
        labelled,
        label=f"{better} - {simpler}",
        target=target,
    )


def compare_predictions(
    first: pd.DataFrame, second: pd.DataFrame, labelled: pd.DataFrame, label: str,
    target: str = "y_major",
) -> pd.DataFrame:
    """Paired comparison, fold by fold: difference in pr_auc (first minus second).

    `first` and `second` hold the columns fold, incident_id and risk, for the same
    test incidents (two models, or two variants of one model). Both are scored on the
    *same* resampled days, so the interval is for the difference itself. Two separate
    intervals that overlap do not show that two models are equal.
    """
    paired = first.merge(second, on=["fold", "incident_id"], suffixes=("_first", "_second"))
    if len(paired) != len(first) or len(paired) != len(second):
        raise ValueError("The two sets of predictions do not cover the same incidents.")

    rows = []
    for fold, pair in paired.groupby("fold"):
        test = labelled.set_index("incident_id").loc[pair["incident_id"]].reset_index()
        labels = test[target].to_numpy()
        risk_first, risk_second = pair["risk_first"].to_numpy(), pair["risk_second"].to_numpy()
        differences = np.array([
            average_precision_score(labels[sample], risk_first[sample])
            - average_precision_score(labels[sample], risk_second[sample])
            for sample in _day_resamples(test)
        ])
        rows.append({
            "fold": fold,
            "comparison": label,
            "pr_auc_difference": average_precision_score(labels, risk_first)
            - average_precision_score(labels, risk_second),
            "low": np.percentile(differences, 2.5),
            "high": np.percentile(differences, 97.5),
            "share_of_samples_better": np.mean(differences > 0),
        })
    return pd.DataFrame(rows)


def _day_resamples(test: pd.DataFrame):
    """BOOTSTRAP_SAMPLES resamples of the test rows, drawn as whole service days.

    Positions refer to the rows of `test` in order. The fixed seed makes every call
    draw the same days, so two models scored on these samples are paired.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    positions_by_day = list(test.reset_index(drop=True).groupby("service_date").indices.values())
    for _ in range(BOOTSTRAP_SAMPLES):
        days = rng.integers(len(positions_by_day), size=len(positions_by_day))
        yield np.concatenate([positions_by_day[day] for day in days])


def save_backtest(results: dict[str, pd.DataFrame], name: str = "main") -> None:
    """Write the three tables to outputs/backtest/<name>_*.csv."""
    config.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    for table, frame in results.items():
        frame.to_csv(config.BACKTEST_DIR / f"{name}_{table}.csv", index=False)
