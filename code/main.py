"""Run the full pipeline end to end and write every output.

Usage, from the repository root:
    python code/main.py

The exploration notebooks (code/exploration/) are run by hand and are not part of
this pipeline. The stages below run in order, each reading what the previous wrote:
    2. cleaning     -> processed datasets in data/processed/
    3. modeling     -> backtest, calibration, sensitivity, holdout and decision
                       tables in outputs/backtest/
    4. reporting    -> README figures in outputs/figures/
"""

import pandas as pd

import config
from cleaning.load import load_incidents, load_line_month
from modeling.backtest import compare_models, run_backtest, save_backtest
from modeling.calibrate import run_calibration_backtest
from modeling.decision import run_decision
from modeling.holdout import run_holdout
from modeling.sensitivity import run_sensitivity
from modeling.target import add_targets, model_population
from reporting.figures import make_figures


def main() -> None:
    # Make sure the output folders exist before any stage writes to them.
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 2, cleaning: rebuild every processed dataset from the raw files.
    # override=True runs the whole chain (cleaned incidents, cleaned km, line-month
    # table), so a full run never reuses a dataset saved by older cleaning code.
    load_line_month(override=True)

    # Stage 3, modelling: rolling backtest of the model ladder on the main target,
    # plus paired comparisons of each model with the rung below it.
    labelled = add_targets(model_population(load_incidents()))
    results = run_backtest(labelled)
    results["comparisons"] = pd.concat([
        compare_models(results, labelled, "symptom_line_table", "base_rate"),
        compare_models(results, labelled, "logistic", "symptom_line_table"),
        compare_models(results, labelled, "gradient_boosting", "logistic"),
    ])
    save_backtest(results, name="main")

    # Calibration of the chosen model, and the sensitivity runs of its choices.
    save_backtest(run_calibration_backtest(labelled), name="calibration")
    save_backtest(run_sensitivity(labelled), name="sensitivity")

    # The final holdout (train 2019-2023, test 2024-2025), following the protocol
    # written before it was first run (documentation.md, Decision 7).
    holdout = run_holdout(labelled)
    save_backtest(holdout, name="holdout")

    # Decision layer on the holdout: alert budgets and the cost grid.
    save_backtest(run_decision(labelled, holdout["predictions"]), name="decision")

    # Stage 4, reporting: the README figures, drawn from the tables saved above.
    make_figures(load_incidents())


if __name__ == "__main__":
    main()
