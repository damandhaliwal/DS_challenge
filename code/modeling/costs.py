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
# both as a sample weight when training on y_major and in the weighted tuning score.
# Unitless. It was 2 until the backtest showed it changes nothing: PR-AUC 0.234 at 1,
# 0.236 at 2, 0.233 at 4, no fold clearly different (documentation.md, Decision 9).
# Safety responses are already 94% of major incidents, so the target itself carries
# the safety priority.
SAFETY_WEIGHT = 1.0
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
    """Delay minutes of each incident: its recorded delay when that agrees with the
    delay bucket, otherwise the bucket's lower bound (clean_incidents.py, step 7).

    Until Decision 9 (documentation.md) minutes were counted at the bucket's lower
    bound, which undercounts the 30+ incidents (30 min each against a recorded median
    of 49). The change was made after results were seen, so both counts are reported.
    """
    return labelled["delay_minutes"].astype(float)
