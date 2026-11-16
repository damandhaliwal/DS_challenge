"""Final holdout: train on 2019-2023, test once on 2024-2025.

The protocol was written down before this was first run (documentation.md,
Decision 7) and is followed exactly:
  * every model of the ladder is tuned as in the backtest (trained on 2019-2022,
    scored on 2023) and refitted on 2019-2023 with the chosen setting;
  * the chosen model, gradient boosting, is calibrated with the method chosen on the
    backtest (calibrate.py; none since Decision 9), fitted on the 2023 scores of its
    2019-2022 version;
  * results are reported for 2024-2025 together and for each year on its own.

No choice in the project may be revisited because of what this file reports.
"""

import pandas as pd

from modeling.backtest import compare_predictions, evaluate, tune
from modeling.calibrate import CALIBRATION_METHOD, CALIBRATORS, calibration_metrics, reliability_table
from modeling.costs import SAFETY_WEIGHT, sample_weights
from modeling.features import MAIN_FEATURES, build_features
from modeling.train import MODEL_NAMES, fit_model, make_model, predict_risk

TRAIN_YEARS = (2019, 2023)
TEST_YEARS = (2024, 2025)
CHOSEN_MODEL = "gradient_boosting"
TARGET = "y_major"
HOLDOUT_LABEL = "2024-2025"


def run_holdout(labelled: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run the holdout once; see the module docstring.

    Returns:
        metrics       ranking metrics per (period, model); period = "2024-2025",
                      "2024" or "2025"
        calibration   calibration metrics of the chosen model, per period
        reliability   predicted vs observed rate in 10 groups (chosen model, 2024-2025)
        comparisons   paired pr_auc differences between each model and the one below it
        settings      the setting chosen by tuning, per model
        predictions   risk score of every holdout incident per model, plus the
                      calibrated probability of the chosen model
    """
    X = build_features(labelled)[MAIN_FEATURES]
    y = labelled[TARGET]
    weight = sample_weights(labelled, SAFETY_WEIGHT)
    year = labelled["service_date"].dt.year

    train = year.between(*TRAIN_YEARS)
    early = year.between(TRAIN_YEARS[0], TRAIN_YEARS[1] - 1)
    validation = year == TRAIN_YEARS[1]
    test = year.between(*TEST_YEARS)
    # The whole holdout, then each of its years on its own.
    periods = {HOLDOUT_LABEL: test}
    for test_year in range(TEST_YEARS[0], TEST_YEARS[1] + 1):
        periods[str(test_year)] = year == test_year

    metrics, settings, predictions = [], [], []
    for name in MODEL_NAMES:
        chosen = tune(name, MAIN_FEATURES, X[train], y[train], weight[train], year[train])
        model = make_model(name, MAIN_FEATURES).set_params(**chosen)
        fit_model(model, X[train], y[train], weight[train])
        risk = predict_risk(model, X[test])

        settings.append({"model": name, **chosen})
        predictions.append(pd.DataFrame({
            "fold": HOLDOUT_LABEL, "model": name,
            "incident_id": labelled.loc[test, "incident_id"], "risk": risk,
        }))
        for period, in_period in periods.items():
            rows = in_period[test]
            metrics.append({"period": period, "model": name,
                            **evaluate(labelled[test][rows], risk[rows], TARGET)})

        if name == CHOSEN_MODEL:
            # Calibrator from the 2023 scores of the same setting trained on 2019-2022.
            early_model = make_model(name, MAIN_FEATURES).set_params(**chosen)
            fit_model(early_model, X[early], y[early], weight[early])
            calibrator = CALIBRATORS[CALIBRATION_METHOD]().fit(
                predict_risk(early_model, X[validation]).to_numpy(), y[validation].to_numpy()
            )
            probability = pd.Series(calibrator.transform(risk.to_numpy()), index=risk.index)

    predictions = pd.concat(predictions, ignore_index=True)
    chosen_rows = predictions["model"] == CHOSEN_MODEL
    predictions.loc[chosen_rows, "probability"] = probability.to_numpy()

    calibration = pd.DataFrame([
        {"period": period, "method": CALIBRATION_METHOD,
         **calibration_metrics(probability[in_period[test]].to_numpy(),
                               y[test][in_period[test]].to_numpy(), y[train].mean())}
        for period, in_period in periods.items()
    ])

    comparisons = pd.concat([
        compare_predictions(predictions[predictions["model"] == better],
                            predictions[predictions["model"] == simpler],
                            labelled, label=f"{better} - {simpler}", target=TARGET)
        for better, simpler in zip(MODEL_NAMES[1:], MODEL_NAMES[:-1])
    ], ignore_index=True)

    return {
        "metrics": pd.DataFrame(metrics),
        "calibration": calibration,
        "reliability": reliability_table(probability.to_numpy(), y[test].to_numpy()),
        "comparisons": comparisons,
        "settings": pd.DataFrame(settings),
        "predictions": predictions,
    }
