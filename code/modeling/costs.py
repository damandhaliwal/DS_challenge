"""How incidents are weighted and how escalation is judged, fixed before any modelling.

Every value in this file was set before a model was trained or a test year was looked
at, so the evaluation cannot be tuned to make a model look good. Changing a value
here after seeing results would have to be reported as such.

Two dimensions are kept apart and never merged into one number:
  * safety: how many safety-response incidents are caught (Public Affairs / Safety)
  * delay burden: how many delay minutes are caught (CFO, VP Operations)
Danger to people is not delay, so safety is not converted into minutes.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Safety weight
# ---------------------------------------------------------------------------
# A safety-response incident counts this many times as much as any other incident,
# both as a sample weight when training on y_major (so the model works harder to rank
# safety events at the top) and in the weighted capture score. Unitless.
SAFETY_WEIGHT = 2.0
# Values reported side by side, to show whether the choice changes the conclusions.
SAFETY_WEIGHT_GRID = [1.0, 2.0, 4.0]

# ---------------------------------------------------------------------------
# Alert budgets (main decision view)
# ---------------------------------------------------------------------------
# Share of train incidents escalated, taken from the top of the predicted-risk
# ranking. A budget is easy to staff: "the team can escalate 1 incident in 10".
ALERT_BUDGETS = [0.05, 0.10]

# ---------------------------------------------------------------------------
# Escalation cost grid (CFO sensitivity view)
# ---------------------------------------------------------------------------
# Escalating early pays off when
#     P(major) x avoidable share x delay minutes of a major incident > false-alarm cost
# The data cannot measure either assumption, so each spans a wide range instead of
# a single guess. These are illustrative ranges, not estimates.
#   avoidable share: fraction of a major incident's delay minutes that early
#                    escalation would save
#   false-alarm cost: delay-minute equivalent of escalating an incident that turns
#                     out not to be major (staff time, held trains)
AVOIDABLE_SHARE_GRID = [0.10, 0.25, 0.50]
FALSE_ALARM_COST_MINUTES_GRID = [0.5, 1.0, 2.0]


def sample_weights(labelled: pd.DataFrame, safety_weight: float = SAFETY_WEIGHT) -> pd.Series:
    """Weight of each incident: safety_weight for a safety response, 1 otherwise."""
    return labelled["y_safety"].map({True: safety_weight, False: 1.0}).rename("weight")


def delay_minutes(labelled: pd.DataFrame) -> pd.Series:
    """Delay minutes of each incident, counted at the lower bound of its delay bucket.

    A 10-14 min incident counts as 10 and a 30+ min incident as 30. This needs no
    assumption about where inside a bucket a delay falls, or how long the open-ended
    30+ bucket really runs. Totals are therefore "at least" figures: the true burden
    is higher, mostly because 30+ incidents run longer than 30 min.
    """
    return labelled["delay_floor_minutes"].astype(float).rename("delay_minutes")
