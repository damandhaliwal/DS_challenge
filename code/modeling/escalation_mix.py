"""What the escalated incidents are, and which major incidents are missed.

The backtest and the holdout say how many major incidents the riskiest 10% contains.
This file opens that number up for the chosen model, in two tables:

    by_symptom   per first symptom: its share of the major incidents, and how many of
                 them the model escalates. A symptom with many major incidents and
                 none escalated is where the report-time data runs out.
    by_outcome   per kind of outcome (what was evacuated, or how long the delay was
                 when there was no safety response): how long the group's delays are,
                 and how much of it the model escalates. The major label treats a
                 single car evacuated and a train and its station evacuated alike;
                 this table shows whether the model finds the heavier ones.
    by_customer_cause   customer-symptom incidents only, per secondary cause. The
                 cause is recorded after the incident, so it is never an input; the
                 table asks whether a finer customer code, if it were known at report
                 time, could separate the major incidents.

Each table starts with a row for all incidents, and has one block of rows for the backtest (test years 2021-2023, each year
escalating its own riskiest 10%) and one for the holdout. No model is trained here
and no choice depends on these tables: they are read-outs of scores already saved.
"""

import pandas as pd

from modeling.costs import ALERT_BUDGETS, delay_minutes
from modeling.features import UNKNOWN
from modeling.holdout import CHOSEN_MODEL, HOLDOUT_LABEL
from modeling.target import (
    EMERGENCY_TEAM_ONLY,
    EVACUATION_KINDS,
    LONG,
    MINOR,
    MODERATE,
    VERY_LONG,
)

BUDGET = max(ALERT_BUDGETS)
ALL = "all"                     # the row of a table that covers every incident in it
CUSTOMER_SYMPTOM = "Clientèle"
BACKTEST_LABEL = "backtest 2021-2023"
HOLDOUT_PERIOD = f"holdout {HOLDOUT_LABEL}"

# Outcome kinds, most serious first: the safety responses, then the delay tiers of the
# incidents without one.
OUTCOME_KINDS = [*EVACUATION_KINDS.values(), EMERGENCY_TEAM_ONLY, VERY_LONG, LONG, MODERATE, MINOR]


def run_escalation_mix(
    labelled: pd.DataFrame, backtest_predictions: pd.DataFrame, holdout_predictions: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """The three tables, for the backtest and the holdout scores of the chosen model."""
    scored = pd.concat([
        score(labelled, backtest_predictions, BACKTEST_LABEL),
        score(labelled, holdout_predictions, HOLDOUT_PERIOD),
    ], ignore_index=True)
    customer = scored[scored["symptom"] == CUSTOMER_SYMPTOM]
    return {
        "by_symptom": mix_by(scored, "symptom"),
        "by_outcome": mix_by(scored, "outcome"),
        "by_customer_cause": mix_by(
            customer.assign(secondary_cause=customer["secondary_cause"].fillna(UNKNOWN)),
            "secondary_cause",
        ),
    }


def score(labelled: pd.DataFrame, predictions: pd.DataFrame, period: str) -> pd.DataFrame:
    """One row per scored incident: outcome, delay minutes and the share of it escalated."""
    chosen = predictions[predictions["model"] == CHOSEN_MODEL]
    scored = labelled.merge(chosen[["fold", "incident_id", "risk"]], on="incident_id", how="inner")
    if len(scored) != len(chosen):
        raise ValueError("Some predictions do not match a labelled incident.")
    return scored.assign(
        period=period,
        symptom=scored["symptom"].fillna(UNKNOWN),
        outcome=outcome_kind(scored),
        delay_minutes=delay_minutes(scored),
        # Each test year (fold) escalates its own riskiest incidents.
        escalated=scored.groupby("fold")["risk"].transform(escalated_share, budget=BUDGET),
    )


def outcome_kind(labelled: pd.DataFrame) -> pd.Series:
    """The kind of safety response, or the delay tier when there was none."""
    kind = labelled["safety_response_kind"].astype(str)
    outcome = kind.where(labelled["y_safety"], labelled["severity_tier"].astype(str))
    return pd.Series(pd.Categorical(outcome, categories=OUTCOME_KINDS), index=labelled.index)


def escalated_share(risk: pd.Series, budget: float) -> pd.Series:
    """How much of each incident falls in the riskiest `budget` (1 = escalated, 0 = not).

    Tied scores at the cut-off share the remaining slots equally, the same rule as
    backtest.capture, so summing these shares over a group gives that group's exact
    part of the capture.
    """
    slots = budget * len(risk)
    riskier = risk.rank(method="min", ascending=False) - 1   # incidents with a higher score
    tied = risk.groupby(risk).transform("size")
    return ((slots - riskier) / tied).clip(lower=0, upper=1)


def mix_by(scored: pd.DataFrame, group: str) -> pd.DataFrame:
    """Per period: one row for all incidents, then one per group.

        share_of_major        the group's share of the period's major incidents
        share_under_5 / share_30plus / mean_delay_minutes   how long its delays are
        share_escalated       share of the group's incidents in the riskiest 10%
        share_of_escalations  the group's share of everything escalated
        major_caught          share of the group's major incidents escalated
        delay_30plus_caught   share of the group's 30+ min incidents escalated

    Groups are listed by seriousness when they have an order (outcome kinds), and by
    their number of major incidents otherwise.
    """
    scored = scored.assign(
        under_5=~scored["y_5plus"],
        major_escalated=scored["escalated"] * scored["y_major"],
        delay_30plus_escalated=scored["escalated"] * scored["y_30plus"],
    )
    ordered = isinstance(scored[group].dtype, pd.CategoricalDtype)

    blocks = []
    for period, rows in scored.groupby("period", sort=False):
        total = _summarise(rows.groupby(pd.Series(ALL, index=rows.index)))
        by_group = _summarise(rows.groupby(group, observed=True))
        by_group = by_group.sort_index() if ordered else by_group.sort_values("major", ascending=False)

        block = pd.concat([total, by_group]).rename_axis(group).reset_index()
        block.insert(0, "period", period)
        block.insert(4, "share_of_major", block["major"] / total["major"].iloc[0])
        block["share_escalated"] = block["escalated"] / block["incidents"]
        block["share_of_escalations"] = block["escalated"] / total["escalated"].iloc[0]
        block["major_caught"] = block["major_escalated"] / block["major"].where(block["major"] > 0)
        block["delay_30plus_caught"] = (
            block["delay_30plus_escalated"] / block["delay_30plus"].where(block["delay_30plus"] > 0)
        )
        blocks.append(block)

    return pd.concat(blocks, ignore_index=True).drop(
        columns=["escalated", "major_escalated", "delay_30plus", "delay_30plus_escalated"]
    )


def _summarise(grouped) -> pd.DataFrame:
    """Counts and averages of each group of scored incidents."""
    return grouped.agg(
        incidents=("incident_id", "size"),
        major=("y_major", "sum"),
        share_under_5=("under_5", "mean"),
        share_30plus=("y_30plus", "mean"),
        mean_delay_minutes=("delay_minutes", "mean"),
        escalated=("escalated", "sum"),
        major_escalated=("major_escalated", "sum"),
        delay_30plus=("y_30plus", "sum"),
        delay_30plus_escalated=("delay_30plus_escalated", "sum"),
    )
