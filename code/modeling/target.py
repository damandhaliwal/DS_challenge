"""What the models predict: the population of incidents and the labels.

Priority order agreed for this project: safety first, then the longest disruptions,
then shorter ones. The data has no injury or fatality field, so "safety" is measured
by the STM's response to the incident (notebook 04, "Severity ladder" section):

    tier                 definition                                   train incidents
    safety_response      evacuation or metro emergency team called           1,515
    very_long_30plus     delay of 30+ min, no safety response                  102
    long_10_to_29        delay of 10-29 min                                  1,041
    moderate_5_to_9      delay of 5-9 min                                    4,074
    minor_under_5        delay under 5 min                                  22,435

Each incident goes into the highest tier it qualifies for. The main target, a "major
interruption", is the top two tiers together (see add_targets).

Two caveats to state wherever these labels are reported:
  * "Safety response" is not "danger to life": about half of the tier is a train
    breakdown where the passengers were taken off the train.
  * Every label is an outcome, known only after the incident. The fields behind them
    (evacuation, emergency team, delay bucket) must never be model inputs.
"""

import numpy as np
import pandas as pd

from cleaning.build_datasets import DELAY_THRESHOLDS
from cleaning.clean_incidents import NO_EVACUATION

# Ordered from least to most severe, so comparisons read naturally:
# severity_tier >= LONG means "long or worse".
MINOR = "minor_under_5"
MODERATE = "moderate_5_to_9"
LONG = "long_10_to_29"
VERY_LONG = "very_long_30plus"
SAFETY_RESPONSE = "safety_response"
SEVERITY_TIERS = [MINOR, MODERATE, LONG, VERY_LONG, SAFETY_RESPONSE]


def model_population(incidents: pd.DataFrame) -> pd.DataFrame:
    """The incidents the models learn from: train incidents only.

    Station incidents are always in the shortest delay bucket and have no cause
    (notebook 01, to-do item 2), so they can never be a major interruption.
    """
    return incidents[incidents["incident_type"] == "T"].reset_index(drop=True)


def add_targets(incidents: pd.DataFrame) -> pd.DataFrame:
    """Add the severity tier and one yes/no label per question we ask of the data.

        y_major       MAIN TARGET, a major interruption: a safety response or a delay of
                      30+ min (the top two tiers, 1,617 train incidents, ~240 a year).
                      It follows the priority order: safety first, then the longest.
        y_safety      the safety tier on its own
        y_30plus      the longest disruptions, whether or not there was a safety response
        y_5plus       comparison label: the usual "service disruption" threshold
        y_10plus, y_15plus   robustness checks between the two
    """
    is_safety_response = (incidents["evacuation"] != NO_EVACUATION) | incidents["emergency_team"]
    delay = incidents["delay_floor_minutes"]

    # np.select takes the first condition that is true, so the order is the ladder.
    tier = np.select(
        [is_safety_response, delay >= 30, delay >= 10, delay >= 5],
        [SAFETY_RESPONSE, VERY_LONG, LONG, MODERATE],
        default=MINOR,
    )

    severity_tier = pd.Categorical(tier, categories=SEVERITY_TIERS, ordered=True)

    labels = {"y_major": severity_tier >= VERY_LONG, "y_safety": is_safety_response}
    labels |= {f"y_{threshold}plus": delay >= threshold for threshold in DELAY_THRESHOLDS}
    return incidents.assign(severity_tier=severity_tier, **labels)
