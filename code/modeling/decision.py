"""Decision layer: from calibrated probabilities to "which incidents to escalate".

The model gives every train incident a probability of becoming major. This file turns
those probabilities into decisions and into what each decision would have caught, on
the final holdout (2024-01 to 2025-09), whose incidents no model or setting has seen.

Two views, both fixed in advance in costs.py:

1. Alert budgets (main view, for VP Operations). Escalate the riskiest 5% or 10% of
   train incidents. Reported per budget: the probability cut-off it implies, how many
   escalations a year and a week that means, and what they catch (major incidents,
   safety responses, delay minutes), plus how many escalations it takes per major
   incident found.

2. Cost grid (sensitivity view, for the CFO). Escalating an incident pays off when
       P(major) x avoidable share x delay minutes of a major incident > false-alarm cost
   so the cut-off is  false-alarm cost / (avoidable share x minutes per major).
   Neither assumption can be measured, so each spans a range (costs.py). For every
   pair the grid reports the cut-off, the share of incidents escalated, what is
   caught, and the net delay minutes saved (minutes saved on the major incidents
   escalated, minus the false-alarm cost of the others).

Delay minutes are the bucket's lower bound (costs.delay_minutes), so minute figures are
"at least" values. Safety responses are always reported next to them, never converted.
"""

import numpy as np
import pandas as pd

from modeling.backtest import capture
from modeling.costs import (
    ALERT_BUDGETS,
    AVOIDABLE_SHARE_GRID,
    FALSE_ALARM_COST_MINUTES_GRID,
    delay_minutes,
)

# Budgets for the full curve (1% to 30% of incidents), used for the figures.
CURVE_BUDGETS = np.round(np.arange(0.01, 0.31, 0.01), 2)
DAYS_PER_YEAR = 365.25


def score_holdout(labelled: pd.DataFrame, holdout_predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per holdout incident: calibrated probability, outcomes and delay minutes."""
    chosen = holdout_predictions.dropna(subset=["probability"])
    scored = labelled.merge(chosen[["incident_id", "probability"]], on="incident_id", how="inner")
    if len(scored) != len(chosen):
        raise ValueError("Some holdout predictions do not match a labelled incident.")
    return scored.assign(delay_minutes=delay_minutes(scored))


def run_decision(labelled: pd.DataFrame, holdout_predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Budget table, budget curve and cost grid on the holdout.

    The minutes of a typical major incident come from the training years (before
    2024), not from the holdout, so the cut-offs of the cost grid are set in advance.
    """
    scored = score_holdout(labelled, holdout_predictions)
    training = labelled[labelled["service_date"] < scored["service_date"].min()]
    minutes_per_major = delay_minutes(training)[training["y_major"]].mean()

    return {
        "budgets": budget_table(scored, ALERT_BUDGETS),
        "curve": budget_table(scored, CURVE_BUDGETS),
        "cost_grid": cost_grid(scored, minutes_per_major),
    }


def budget_table(scored: pd.DataFrame, budgets) -> pd.DataFrame:
    """What escalating the riskiest share of incidents would have caught, per budget."""
    years = _years_covered(scored)
    probability = scored["probability"]
    rows = []
    for budget in budgets:
        escalated = budget * len(scored)
        majors_caught = capture(probability, scored["y_major"], budget) * scored["y_major"].sum()
        rows.append({
            "budget": budget,
            "probability_cut_off": np.quantile(probability, 1 - budget),
            "escalations_per_year": escalated / years,
            "escalations_per_week": escalated / years / 52,
            "major_caught": capture(probability, scored["y_major"], budget),
            "safety_caught": capture(probability, scored["y_safety"], budget),
            "minutes_caught": capture(probability, scored["delay_minutes"], budget),
            "majors_found_per_year": majors_caught / years,
            "escalations_per_major_found": escalated / majors_caught,
        })
    return pd.DataFrame(rows)


def cost_grid(scored: pd.DataFrame, minutes_per_major: float) -> pd.DataFrame:
    """Escalate when P(major) is above the cost-based cut-off; one row per assumption pair."""
    years = _years_covered(scored)
    rows = []
    for avoidable_share in AVOIDABLE_SHARE_GRID:
        for false_alarm_cost in FALSE_ALARM_COST_MINUTES_GRID:
            cut_off = false_alarm_cost / (avoidable_share * minutes_per_major)
            escalate = scored["probability"] >= cut_off
            major = scored["y_major"]
            saved = avoidable_share * scored.loc[escalate & major, "delay_minutes"].sum()
            false_alarm_cost_total = false_alarm_cost * (escalate & ~major).sum()
            rows.append({
                "avoidable_share": avoidable_share,
                "false_alarm_cost_minutes": false_alarm_cost,
                "probability_cut_off": cut_off,
                "share_escalated": escalate.mean(),
                "escalations_per_year": escalate.sum() / years,
                "major_caught": (escalate & major).sum() / major.sum(),
                "safety_caught": (escalate & scored["y_safety"]).sum() / scored["y_safety"].sum(),
                "net_minutes_saved_per_year": (saved - false_alarm_cost_total) / years,
            })
    return pd.DataFrame(rows).assign(minutes_per_major=minutes_per_major)


def _years_covered(scored: pd.DataFrame) -> float:
    """Length of the holdout in years, from its first to its last service day."""
    days = (scored["service_date"].max() - scored["service_date"].min()).days + 1
    return days / DAYS_PER_YEAR
