# Documentation

A record of analytical findings and the decisions they led to. Each entry says what
was asked, how it was tested, what came out, and what it changes in the project.
Numbers come from the cleaned data (`code/cleaning/`) and the labels in
`code/modeling/target.py`; "major" means `y_major` (a safety response or a delay of
30+ min).

---

## Finding 1: Station incidents are not a precursor of train incidents

**Question.** Station incidents (type `S`, 16,394 rows) never delay service themselves:
they are always in the shortest delay bucket. Could they still be an early warning, or a
model input, for train incidents, and in particular for major ones?

**Short answer.** No. The two kinds of incident look correlated, but almost all of it
is volume: busy lines and busy days have more of both. Once that is controlled for,
station incidents predict neither more train incidents nor more severe ones.

### Test 1: daily co-movement

Counts of station and train incidents per line per day (single-line incidents,
2019-01-01 to 2025-09-30).

| Comparison | Correlation |
|---|---|
| Raw | 0.55 |
| Within the same line, weekday and month | 0.09 |
| Yesterday's station incidents vs today's train incidents (same control) | 0.01 |
| Station incidents vs **major** train incidents (same control) | 0.03 |

The raw 0.55 mostly reflects line size: the green and orange lines carry most
passengers and have the most incidents of every kind. Comparing each day only with
days of the same line, weekday and month removes that, and almost nothing is left.

### Test 2: same station, shortly after (time-matched controls)

For each station incident, count the train incidents at the same station in the next
15, 60 and 180 minutes. The expected count comes from the same station and the same
time window 7 and 14 days earlier and later, so the station, the weekday and the time
of day are held fixed. The ratio is observed / expected; the 95% interval uses the
normal approximation on the log scale.

| Window | Train incidents | Observed | Expected | Ratio [95% CI] |
|---|---|---|---|---|
| next 15 min | all | 74 | 65.8 | 1.13 [0.87, 1.46] |
| next 15 min | major | 6 | 2.8 | 2.18 [0.81, 5.90] |
| next 60 min | all | 285 | 254.8 | 1.12 [0.98, 1.28] |
| next 60 min | major | 18 | 10.8 | 1.67 [0.97, 2.90] |
| next 180 min | all | 795 | 723.5 | 1.10 [1.02, 1.19] |
| next 180 min | major | 38 | 29.5 | 1.29 [0.89, 1.86] |

* The effect on all train incidents is small (about +10%).
* The ratio for major incidents looks larger, but the counts are tiny: about 16,000
  station incidents are followed by about **7 extra major train incidents in almost 7
  years (about one a year)**, against roughly 240 major incidents a year. None of the
  major-incident intervals clearly excludes "no effect".
* By station-incident symptom, only fire / smoke stands out: 5 train incidents in the
  next 15 minutes against 1 expected. This is most likely **one event recorded twice**
  (smoke in a station, trains held), not an early warning. "Customer" station
  incidents give about +15%; station-operations incidents give nothing.

### Test 3: as a model input

This is how a station incident would enter the classifier: when a train incident is
reported, was there a station incident shortly before it? Major rate of train
incidents with and without one (Wilson 95% intervals):

| Station incident in the previous 60 min | Train incidents | Major rate [95% CI] |
|---|---|---|
| Same station: yes | 281 | 6.05% [3.81, 9.47] |
| Same station: no | 28,886 | 5.54% [5.28, 5.81] |
| Same line: yes | 3,395 | 5.45% [4.73, 6.26] |
| Same line: no | 25,772 | 5.56% [5.28, 5.84] |

The intervals overlap almost completely. The 30- and 180-minute windows give the same
picture: the same-station rate is 7.80% [4.41, 13.43] over 30 minutes (141 incidents)
and 4.44% [3.19, 6.14] over 180 minutes, so the direction is not even stable. Only
about 1% of train incidents have a same-station station incident in the hour before,
so even a real effect would barely move a model.

### Decision

* **Station incidents are not used as model inputs or alert triggers.** This agrees with
  notebook 04, where earlier incidents cluster in time but do not raise the severe risk.
* **Extra reason: a timing risk.** The data does not say whether station incidents are
  logged in real time. If they are entered after the fact, using them would leak
  future information into the model.
* **For stakeholders:** an early warning from station activity would catch at most about
  one extra major incident a year, which is too few to support an alert procedure.

### Caveats

* The controls hold the station, weekday and time of day fixed, but not other
  conditions on the day itself (weather, events, crowding). Those raise both kinds of
  incident at once, so they would make the observed link look *stronger* than the true
  one. The true precursor effect is therefore at most as large as measured here,
  which only strengthens the conclusion.
* Train incidents are matched to a station through `station_from`, so incidents on a
  whole line, a facility or an ancillary structure are not part of tests 2 and 3.
* A station incident on several lines counts on each of them for the "same line"
  comparison.

**Reproduce.** `code/exploration/04_consistency_and_signals.ipynb`, section 8.

---

## Decision 2: Weights and pricing, fixed before modelling

**Why now.** The rules that score a model have to be set before any model is trained or
any test year is looked at. Otherwise the weights could be adjusted until a model looks
good. Everything below lives in `code/modeling/costs.py`; changing it after seeing
results would have to be reported.

**Two dimensions, never merged.**

| Dimension | Measured as | Main audience |
|---|---|---|
| Safety | safety-response incidents caught | Public Affairs / Safety |
| Delay burden | delay minutes caught | CFO, VP Operations |

Danger to people is not delay, so safety is not converted into minutes.

**Safety weight.** A safety-response incident counts **2x** any other incident, as a
sample weight when training on `y_major` and in the weighted capture score. The weight
is unitless. Results are reported for 1x, 2x and 4x to show whether the choice matters.
*Changed after results were seen (Decision 9): the main run now uses 1x, because the
weight changed nothing.*

**Delay minutes at the bucket's lower bound.** A 10-14 min incident counts as 10, a 30+
min incident as 30. This needs no assumption about where a delay falls inside its
bucket, or how long the open-ended 30+ bucket runs (recorded durations there have a
median of 52 min). Every total is therefore an **"at least"** figure. On this measure,
major incidents are 5.6% of train incidents but **31.9% of delay minutes**
(2019-2023, before the final holdout). *Changed after results were seen (Decision 9):
for train incidents the recorded delay agrees with the bucket 98% of the time, so
minutes are now the recorded delay (bucket lower bound where the two disagree). On
that count major incidents are 38.7% of delay minutes.*

**Decision rule.**
* Main view: escalate the top **5%** and **10%** of incidents by predicted risk, and
  report the safety events and delay minutes caught.
* CFO sensitivity view: escalating pays off when
  P(major) x avoidable share x delay minutes of a major incident > false-alarm cost.
  Neither assumption can be measured from the data, so both span a wide range:
  avoidable share 10% / 25% / 50%, false-alarm cost 0.5 / 1 / 2 delay-minute
  equivalents. These ranges are illustrative, not estimates.

---

## Decision 3: Model inputs, and leak suspects held out

**Rule.** Inputs are a whitelist of what is known when an incident is reported
(`code/modeling/features.py`). Everything set afterwards (causes, rolling stock, end
time, delay, evacuation, emergency team, power cut, damage) is excluded, and a test
checks that none of them, nor any label, is an input.

**Main inputs.** Symptom, line group, location kind, at or next to a terminal, door
flag (as recorded), emergency brake and time of day (service-day clock bands). Two
more inputs, at or next to a transfer station and weekend, were in the first version
and were dropped for carrying no signal (Decision 8).

**Leak suspects, held out of the main model.** Each carries a strong signal, but the
data does not say *when* it is recorded. The third was added in Decision 9:

| Field | Signal (train incidents, 2019-2023) | Timing doubt |
|---|---|---|
| Run number present | major rate 6.6% with, 1.4% without; the gap holds within every symptom (fire / smoke: 21.9% vs 6.6%) | recorded at report time, or added after a train was held? |
| Location "whole line / network" | major rate 12.4% (vs 5.3% at a station) | known at report time, or only once the incident has spread? |
| Door numbers listed | major rate 3.3% with, 6.0% without | written at report time, or after the doors are inspected? Until Decision 9 they were merged into the door flag |

The main model leaves them out (whole-line locations become "unspecified"). A
sensitivity run adds them back and reports the difference. A large gain is a question
for STM ("is this logged in real time?"), not a reason to trust the field.

**Left out on purpose.** The station itself (68 values with few severe events each; the
terminal flag keeps its most robust signal), car numbers, month
and holidays (no seasonality in the severe rate, notebook 04), and station incidents
(Finding 1).

---

## Decision 4: Gradient boosting goes forward, logistic stays as the benchmark

**How it was chosen.** Rolling yearly backtest (`code/modeling/backtest.py`): train
2019-20 / test 2021, train 2019-21 / test 2022, train 2019-22 / test 2023. Each model
was tuned on its training years only (last training year held back for validation).
Intervals come from a bootstrap that resamples whole service days. The 2024-25 holdout
was not used.

**Results, averaged over the three folds** (target `y_major`):

| Model | PR-AUC | Top 5%: major | Top 10%: major | Top 10%: safety | Top 10%: delay minutes |
|---|---|---|---|---|---|
| Base rate | 0.057 | 5% | 10% | 10% | 10% |
| Symptom x line table | 0.136 | 15% | 25% | 27% | 13% |
| Logistic regression | 0.221 | 29% | 38% | 39% | 15% |
| Gradient boosting | 0.230 | 28% | 38% | 40% | 16% |

**Paired differences in PR-AUC** (same resampled days for both models):

| Comparison | 2021 | 2022 | 2023 |
|---|---|---|---|
| Table - base rate | +0.106 [0.073, 0.148] | +0.060 [0.042, 0.091] | +0.073 [0.050, 0.104] |
| Logistic - table | +0.074 [0.041, 0.113] | +0.077 [0.048, 0.113] | +0.103 [0.061, 0.157] |
| Gradient boosting - logistic | -0.004 [-0.031, 0.019] | +0.008 [-0.006, 0.026] | +0.022 [-0.007, 0.045] |

**Decision.** Gradient boosting is the model that goes forward to calibration and the
final holdout. The choice was made on the first setup (nine inputs, 2x safety weight),
where it was ahead of logistic regression in every fold, by about +0.02 PR-AUC, clearly
in 2023. With the current setup (Decisions 8 and 9) the backtest gain is smaller:
+0.009 on average, slightly behind in 2021, ahead in 86% and 93% of the resamples in
2022 and 2023, and no fold's interval excludes zero. At the 5% budget logistic catches
slightly more (28.8% against 28.3%). On the holdout the gain is clear (+0.031
[0.013, 0.045], Finding 4).
Logistic regression is therefore reported next to it as the **explainable benchmark**:
the extra complexity buys a little, and the reader can see exactly how much.

**What the numbers mean for stakeholders.** Escalating 1 incident in 10 flags about 38%
of major incidents and 40% of safety responses, against 10% by chance. Only about 16%
of delay minutes are in that group: a model aimed at major events does not chase the
many 5-9 min delays, which is the trade-off shown in notebook 03.

---

## Decision 5: Calibration with a Platt curve (no longer needed since Decision 9)

**Why.** The models were trained with safety responses weighted 2x, so their scores
ranked incidents well but overstated the chance of a major incident (about 0.10 on
average, against a real rate of about 0.057). A probability given to a stakeholder
must be true on average.

**How** (`code/modeling/calibrate.py`). In each backtest fold the model is trained on
the training years minus the last one; a calibrator learns score -> share of major
incidents on that last training year, without weights; it is then applied to the test
year scores of the model refitted on all training years. No test-year data is used.

**Results with the 2x weight** (gradient boosting, averaged over the 2021-2023 folds):

| Method | Brier | Log loss | Mean predicted | Observed | Calibration error |
|---|---|---|---|---|---|
| None (raw scores) | 0.0512 | 0.207 | 0.101 | 0.057 | 0.044 |
| Platt | 0.0489 | 0.193 | 0.065 | 0.057 | 0.017 |
| Isotonic | 0.0491 | 0.196 | 0.067 | 0.057 | 0.017 |
| Constant training rate (reference) | 0.0536 | | | | |

**Decision (with the 2x weight).** Platt. It ties with isotonic (slightly ahead on every
measure), and has 2 parameters, a smooth curve and no ties. Calibration does not change
the ranking, so every capture figure stays the same.

**Results without the weight (Decision 9).** The rule, set before this run, was to drop
Platt if the raw scores were already calibrated:

| Method | Brier | Log loss | Mean predicted | Observed | Calibration error (2021 / 2022 / 2023) |
|---|---|---|---|---|---|
| None (raw scores) | 0.0491 | 0.193 | 0.058 | 0.057 | 0.013 (0.016 / 0.012 / 0.011) |
| Platt | 0.0492 | 0.194 | 0.065 | 0.057 | 0.016 (0.014 / 0.021 / 0.012) |
| Isotonic | 0.0493 | 0.196 | 0.066 | 0.057 | 0.015 |

The raw scores are as good as Platt on Brier and log loss and better on calibration
error in 2 of 3 folds, so **no calibrator is used** (`calibrate.CALIBRATION_METHOD`).
The calibration check stays in the pipeline.

**Remaining bias (2x run).** Probabilities still ran about 0.8 points high. The
major-incident rate fell from 7.1% (2021) to 5.3% (2022) and 4.7% (2023), and the
calibrator always learns from the year before. Probabilities should be read as "the
rate of last year's incidents that looked like this one".

---

## Finding 2: Sensitivity runs: the main conclusions do not depend on our choices

Each variant reruns the gradient-boosting backtest with one choice changed
(`code/modeling/sensitivity.py`), and is compared with the main run on the same test
incidents (paired, day-level bootstrap). Averages over the 2021-2023 folds:

| Variant | PR-AUC (major) | Difference vs main (folds clearly better / worse) | Top 10%: major | Top 10%: safety | Top 10%: delay minutes |
|---|---|---|---|---|---|
| Main (no safety weight) | 0.230 | | 38.1% | 39.6% | 16.2% |
| + leak suspects (run number, whole line, door numbers) | 0.251 | +0.021 (1 / 0) | 40.5% | 42.1% | 18.1% |
| Safety weight 2x | 0.236 | +0.006 (0 / 0) | 38.0% | 39.5% | 14.7% |
| Safety weight 4x | 0.228 | -0.002 (0 / 0) | 36.8% | 38.2% | 15.0% |
| Trained on 5+ min | 0.099 | -0.131 (0 / 3) | 15.7% | 15.5% | 18.4% |

**What it shows.**
* **Leak suspects add a little** (+0.021 PR-AUC, clear in one fold of three, 2022).
  The main model is not missing much by leaving them out, so the question of when STM
  records them is worth asking but not decisive.
* **The safety weight does not matter** between 1x and 4x: no fold differs clearly.
  Safety responses are already 94% of major incidents, so weighting them more
  changes little. 2x is slightly ahead of 1x in every fold (+0.011, +0.004, +0.004),
  but inside the interval each time; the main run uses 1x (Decision 9).
* **The trade-off is real and large.** A model trained on the usual 5+ min threshold
  finds less than half as many major incidents and safety responses, and gains only
  about 2 points of delay minutes (18.4% vs 16.2%). Even that model puts less than a
  fifth of delay minutes in its top 10%.

---

## Decision 6: No Bayesian model for the main prediction

**Question.** Should the risk model be Bayesian, so that each prediction comes with its
own uncertainty?

**Short answer.** Not for the main model. A Bayesian model helps when an estimate rests
on little data, and the main model's estimates do not. The one place it would help is
the station-level effect, which is left as a next step.

**Why it would change little here** (train incidents, 2019-2023):
* **The estimates rest on a lot of data.** 21,058 incidents and 1,192 major ones, spread
  over 7 coarse inputs. Of the 45 symptom x line cells, the ones with fewer than 100
  incidents hold only 1.8% of incidents. With this much data, a Bayesian fit with weak
  priors gives almost the same probabilities as the current fit.
* **The decision uses one number.** An incident is escalated when its probability is
  above a cut-off. An interval around that probability does not change the action,
  except for incidents right at the cut-off. The uncertainty that matters for the
  conclusions is in the metrics, and the day-level bootstrap already measures it
  (Decision 4, Finding 2).
* **It would not rank incidents better.** Ranking quality is limited by what is known at
  report time, not by the model: gradient boosting beats logistic regression by only
  about 0.01 PR-AUC in the backtest and 0.03 on the holdout (Decision 4, Finding 4). A Bayesian model sees the same inputs.

**Where it would help: the station.** The station itself is left out (Decision 3)
because most stations have few major incidents: the median station has 13 in five
years, and 22 of 68 have fewer than 10. A hierarchical logistic model would give each
station its own effect, pulled toward the average of its line, and pulled harder
the less data the station has. A station with many incidents keeps its own rate; a
station with few keeps mostly its line's rate. This is the principled version of the
shrinkage already used in the symptom x line table (`code/modeling/train.py`).

**Decision.** The main model stays as it is. A hierarchical model with station effects
is a possible extension, judged on the same backtest and paired comparison as every
other variant: it goes forward only if it beats the main model on the same test
incidents.

---

## Finding 3: Delay minutes are a secondary metric, and hard to predict

**Question.** The main model puts only about 16% of delay minutes in its top 10%. Is
delay minutes a good metric for this data, and can a model aimed at minutes catch more?

**Is it a good metric?** (train incidents 2019-2023, minutes as in Decision 9)
* For train incidents the recorded delay falls inside the published bucket 98% of the
  time, so it is a measurement. The first version of this finding said the length of
  30+ incidents "cannot be settled from the data"; that was wrong (Decision 9).
* The old lower-bound count missed a third of the minutes (54,446 against 81,340).
  Most of the gap is in the 30+ incidents (recorded median 52 min, mean 65, against a
  flat 30). Major incidents are **38.7%** of delay minutes (31.9% on the old count).
* **9.7%** of safety responses count zero minutes, and 23.1% count fewer than 5.

**Can a model aimed at minutes do better?** Rolling backtest, average of 2021-2023:

| Ranking | Top 10%: delay minutes | Top 10%: major | Top 10%: safety | Deviance explained |
|---|---|---|---|---|
| Major model (main) | 16.2% | 38.1% | 39.6% | |
| 5+ model | 18.4% | 15.7% | 15.5% | |
| Symptom mean minutes | 14.9% | 12.0% | 11.5% | 4.4% |
| Minutes model (Poisson gradient boosting) | 18.9% | 16.1% | 14.4% | 6.9% |

The minutes model gains about 3 points of minutes over the main model and loses about
22 points of major incidents. It explains about 7% of the deviance: what is known at
report time says little about how many minutes an incident will cost.

**Decision.** Safety responses and major incidents stay the target and the headline.
Delay minutes are reported as a secondary measure. A classifier aimed at delays (5+ or
10+) can be added later as a separate operations view, with this trade-off shown next
to it.

**Reproduce.** `code/exploration/05_delay_minutes.ipynb`.

---

## Decision 7: Final holdout protocol (written before the holdout was run)

Fixed before any model saw 2024 or 2025. The holdout is run **once**; nothing below is
changed after seeing its result. (Two exceptions, made after the first run, each on
evidence from 2019-2023 only and each with both results kept: Decision 8 dropped two
inputs with no signal; Decision 9 changed the door flag, the safety weight, the
calibration and the delay-minutes count.)

* **Data.** Train on train incidents of 2019-2023; test on 2024-01-01 to 2025-09-30
  (2025 ends in September, so it is a partial year).
* **Model.** Gradient boosting on the main inputs, target `y_major`, safety weight 2x
  (Decisions 2-4; 1x since Decision 9). Its setting is tuned exactly as in the backtest: trained on
  2019-2022, scored on 2023, best setting refitted on 2019-2023.
* **Calibration.** Platt (Decision 5): the tuned model trained on 2019-2022 scores 2023,
  the Platt curve is fitted on those scores, and applied to the holdout scores of the
  model trained on 2019-2023. (None since Decision 9.)
* **Benchmarks.** Base rate, symptom x line table and logistic regression, each tuned
  and trained the same way, with paired comparisons (day-level bootstrap) between each
  model and the one below it.
* **Reported.** For 2024-2025 together, and for 2024 and 2025 separately: PR-AUC with
  its 95% interval; the share of major incidents, safety responses and delay minutes
  (lower bound; recorded delay since Decision 9) in the top 5% and 10% of risk; and for the calibrated probabilities the
  Brier score (against a constant forecast of the 2019-2023 rate), the mean predicted
  vs observed rate, and the calibration error.
* **Known risks, stated in advance.** The share of rolling-stock symptoms confirmed as
  rolling-stock causes fell sharply in 2024 (notebook 04), and the major-incident rate
  has moved year to year (7.1%, 5.3%, 4.7% in 2021-2023), so calibration may be off in
  the holdout even if the ranking holds.

---

## Finding 4: Final holdout results (2024-2025)

Following Decision 7, with the current setup (Decisions 8 and 9; earlier runs are
kept in those decisions). Train incidents 2019-2023 for training; 8,109 train
incidents from 2024-01-01 to 2025-09-30 for testing (major-incident rate 5.2%).

**Ranking** (95% intervals from the day-level bootstrap):

| Model | PR-AUC | Top 5%: major | Top 10%: major | Top 10%: safety | Top 10%: delay minutes |
|---|---|---|---|---|---|
| Base rate | 0.052 | 5.0% | 10.0% | 10.0% | 10.0% |
| Symptom x line table | 0.205 | 20.3% | 30.8% | 32.9% | 14.4% |
| Logistic regression | 0.287 | 33.6% | 42.1% | 44.4% | 15.5% |
| **Gradient boosting** | **0.318 [0.278, 0.365]** | **35.6%** | **43.7% [40.0, 48.2]** | **46.2%** | **15.8%** |

| Gradient boosting by year | 2024 | 2025 (Jan-Sep) |
|---|---|---|
| Share of major incidents (PR-AUC of chance) | 5.2% | 5.3% |
| PR-AUC | 0.321 | 0.324 |
| Top 10%: major | 42.3% | 46.5% |
| Top 10%: safety | 44.9% | 48.6% |

**Paired differences in PR-AUC:** table - base rate +0.152 [0.122, 0.187]; logistic -
table +0.083 [0.058, 0.118]; gradient boosting - logistic **+0.031 [0.013, 0.045]**,
better in every resample. In the backtest that last gain was uncertain; on the
holdout it is clear.

**Calibration** (raw scores, Decision 9): mean predicted probability 5.6% against an
observed rate of 5.2%; calibration error 0.010; Brier score 0.043 against 0.050 for a
constant forecast of the 2019-2023 rate (about 14% better). The probabilities run
slightly **high**: they follow the 2019-2023 average rate (5.7%), and 2024-25 is a
little below it. The top 10% is close (20.9% predicted, 23.0% observed); groups 7 and
9 are over-predicted most (4.3% against 3.2%, 8.0% against 5.1%).

**Reading.**
* The holdout is **better** than the backtest average (PR-AUC 0.318 vs 0.230; 43.7% vs
  38.1% of major incidents in the top 10%). The likely reason is more training data
  (five years against two to four), and nothing was tuned on the holdout. The fair
  claim is the range: **about 40% of major incidents and 45% of safety responses in
  the top 10% of incidents, about 4x chance**, stable across 2021-2025.
* The risk stated in advance (the 2024 shift in how rolling-stock incidents are
  confirmed) did not hurt the ranking: 2024 is in line with the other years.
* **From April 2025 far fewer short incidents are recorded** (notebook 07; caveat set
  before the Decision 9 rerun). Train incidents fall by about a third against the same
  months of 2024 (850 fewer), mostly under 5 minutes (708 of the 850) and mostly
  fixed equipment. The
  share of major incidents stays in its usual range (5.9% in April-September 2025,
  4.8-7.3% in 2019-2024), so 2025 is kept, reported on its own, and read next to its
  share of major incidents (5.3%). Why the short incidents disappear is a question for
  STM.
* Delay minutes stay secondary: 15.8% in the top 10% (Finding 3).

---

## Finding 5: What escalation would deliver (decision layer, holdout 2024-2025)

Probabilities of the chosen model on the holdout (8,109 train incidents in
1.75 years: about 243 major incidents and 225 safety responses a year). Computed by
`code/modeling/decision.py`.

**Alert budgets (main view, VP Operations).**

| Escalate | Probability cut-off | Escalations | Major caught | Safety caught | Delay minutes caught | Escalations per major found |
|---|---|---|---|---|---|---|
| Top 5% | 14.4% | about 230 a year (4-5 a week) | 35.6% (about 86 a year) | 37.4% | 13.0% | 2.7 |
| Top 10% | 9.8% | about 460 a year (9 a week) | 43.7% (about 106 a year) | 46.2% | 15.8% | 4.4 |

Wider budgets keep adding major incidents but more slowly: 15% catches 50%, 20%
catches 54%, 30% catches 65%. The top 1% (about 46 escalations a year) already catches
14% of major incidents, and nearly 3 of every 4 escalations there are major.

**Cost grid (sensitivity view, CFO).** Escalate when P(major) > false-alarm cost /
(avoidable share x 26.4 min), where 26.4 is the average delay of a major incident in
2019-2023 (recorded delay, Decision 9). Net delay minutes saved per year:

| Avoidable share \ false-alarm cost | 0.5 min | 1 min | 2 min |
|---|---|---|---|
| 10% | 149 (4% escalated) | 94 (1%) | 0 (never pays) |
| 25% | 362 (16%) | 398 (5%) | 259 (2%) |
| 50% | 1,580 (40%) | 724 (16%) | 796 (5%) |

**Reading.**
* Escalation is **cheap to staff**: the 10% budget is about 9 escalations a week, and
  1 in 4.4 of them turns out major.
* **In delay minutes alone, the gain is small.** The holdout years average about 18,900
  delay minutes a year, 8,000 of them from major incidents. Even the most generous
  assumptions save about 1,580 net minutes a year (8% of all minutes); the middle one
  (25% avoidable, 1-minute false alarm) saves about 400 (2%). With a high false-alarm
  cost and a low avoidable share, escalating never pays.
* **The break-even rule escalates too much when false alarms are cheap.** In two rows a
  higher false-alarm cost saves *more* net minutes (25%: 362 at 0.5 min against 398 at
  1 min; 50%: 724 at 1 min against 796 at 2 min, each net of its own cost). The rule
  gives every major incident the average 26.4 minutes, but the major incidents between
  those cut-offs (probability 7.6%-15.2%, 69 incidents) are 99% safety responses with
  13.7 minutes on average, and the probabilities run slightly high on the holdout. A rule built on each incident's expected minutes would
  fix this; it is a next step, not done here.
* So the case for the model rests on **safety**, not on minutes: escalating 1 incident
  in 10 puts about 44% of safety responses, about 100 a year, in front of the team
  early. This matches Findings 3 and 4, and should lead the recommendation to all
  three stakeholders, with minutes as the secondary, conservative figure.

---

## Decision 8: Two inputs with no signal dropped (weekend, transfer station)

**Evidence** (`code/exploration/06_feature_importance.ipynb`, training years 2019-2023
only). Three methods that fail in different ways agree:

| Input | Random forest: PR-AUC lost when shuffled | Lasso: enters at C (smaller = stronger) | Gradient boosting: PR-AUC change when dropped |
|---|---|---|---|
| Symptom | 0.149 | 0.002 | -0.140 (clearly worse in 3 of 3 folds) |
| Door involved | 0.080 | 0.010 | -0.054 (3 of 3) |
| At a terminal | 0.075 | 0.006 | -0.049 (3 of 3) |
| Time of day, brake, location, line | 0.01-0.02 | 0.002-0.018 | -0.01 to -0.02 (1 of 3 each) |
| **Weekend** | 0.000 | 0.316 | -0.001 (0 of 3) |
| **Transfer station** | -0.005 | 0.316 | +0.001 (0 of 3) |

**Decision.** Both are dropped from the main inputs: a simpler model that scores the
same is easier to explain and has fewer things to go wrong. They are still built by
`features.py` (`NO_SIGNAL_FEATURES`) so notebook 06 can be rerun.

**Protocol.** This was decided after the holdout run (Decision 7), but on training-year
evidence only: no 2024-25 number was looked at to make it. The whole pipeline was then
rerun, holdout included. Both results are kept here so nothing is hidden:

| Gradient boosting | 9 inputs (first run) | 7 inputs (current) |
|---|---|---|
| Backtest PR-AUC, mean of 2021-2023 | 0.237 | 0.236 |
| Backtest top 10%: major / safety | 39.2% / 41.1% | 37.9% / 39.6% |
| Holdout PR-AUC | 0.313 [0.270, 0.364] | 0.314 [0.273, 0.366] |
| Holdout top 10%: major / safety | 42.9% / 45.0% | 42.4% / 44.4% |
| Holdout gradient boosting - logistic | +0.023 [0.011, 0.033] | +0.026 [0.014, 0.037] |

**Reading.** PR-AUC does not move. The share caught in the top 10% is about 1 point
lower in the backtest (lower in all three folds, but inside the paired bootstrap
interval in each) and 0.5 points lower on the holdout. A single cut-off is a noisier
measure than PR-AUC, so this is no clear loss, but it is not a gain either: dropping
the two inputs buys simplicity, not accuracy. Every number elsewhere in this file is
the 7-input run with the changes of Decision 9. (The evidence table above is as it was
when the decision was made; notebook 06 now reruns it on the Decision 9 setup, with
the same verdict for weekend and transfer station.)

---

## Decision 9: Four fixes after a review (door flag, safety weight, calibration, delay minutes)

**Why.** A review of the finished pipeline found four problems. Each fix below was
decided on 2019-2023 evidence only, before the holdout was rerun, and every rule for
choosing was written down before the run it applied to.

1. **Door flag: the raw flag only.** The door input was "flag OR door numbers listed".
   Door numbers may be written in after the doors are inspected, the same doubt that
   kept run number out. The main model now uses the raw flag (`clean_incidents.py`,
   step 5) and "door numbers listed" joins the leak suspects (Decision 3). On its own
   the change costs nothing: with the 2x weight, the backtest PR-AUC is 0.236 with
   either door input.
2. **Safety weight: 1x.** PR-AUC was 0.234 at 1x, 0.236 at 2x and 0.233 at 4x, with no
   fold clearly different (Finding 2, first version). A setting that changes nothing
   is one more thing to explain, and it was the only reason a calibration step was
   needed. With the raw door flag, 2x is slightly ahead of 1x in every fold (+0.006 on
   average) but inside the interval each time: no clear loss, not a gain.
3. **Calibration: none.** Rule set before the run: drop Platt if the unweighted scores
   are already calibrated on the backtest. They are (Decision 5, second table).
4. **Delay minutes: the recorded delay.** For train incidents the recorded start and end
   times bracket the delay itself: the recorded duration falls inside the published
   bucket for 97.8% of them (98.0% in 2019-2023), against 4.5% for station incidents.
   Minutes are now the recorded delay, or the bucket's lower bound where the two
   disagree (a typo such as 1,605 min in the "2 min and less" bucket). Station
   incidents keep 0 (`clean_incidents.py`, step 7). The labels do not change: they use
   the bucket, which is exact at every threshold.

**Also checked before the rerun: the April 2025 drop** (notebook 07). Far fewer short
incidents are recorded from April 2025. The caveat was fixed before the rerun: 2025 is
reported on its own, next to its share of major incidents, and is not dropped
(Finding 4).

**Protocol.** Fixes 1-3 change the model and fix 4 changes a cost measure that
Decision 2 fixed before modelling, so all four are reported as made after results
were seen. The whole pipeline was rerun once, holdout included. Both results:

| Gradient boosting | Before (Decision 8 run) | After (current) |
|---|---|---|
| Backtest PR-AUC, mean of 2021-2023 | 0.236 | 0.230 |
| Backtest top 10%: major / safety | 37.9% / 39.6% | 38.1% / 39.6% |
| Backtest gradient boosting - logistic, by fold | +0.013 / +0.015 / +0.020 | -0.004 / +0.008 / +0.022 |
| Backtest top 10%: delay minutes | 14.9% (lower bound) | 16.2% (recorded) |
| Holdout PR-AUC | 0.314 [0.273, 0.366] | 0.318 [0.278, 0.365] |
| Holdout top 10%: major / safety | 42.4% / 44.4% | 43.7% / 46.2% |
| Holdout gradient boosting - logistic | +0.026 [0.014, 0.037] | +0.031 [0.013, 0.045] |
| Holdout calibration error | 0.008 (Platt) | 0.010 (none) |
| Holdout top 10%: delay minutes | 18.0% (lower bound) | 15.8% (recorded) |

**Reading.** The ranking is essentially unchanged: backtest PR-AUC 0.006 lower, inside
the noise, holdout PR-AUC and capture slightly higher. The model is simpler (no weight,
no calibrator) and leans less on a field that may be recorded late. The delay-minute
shares are not comparable across the two columns, because the count itself changed:
the new count adds the long tail of 30+ incidents, which is harder to rank. The one
real cost is the case for gradient boosting over logistic regression, which is now
inside the noise in the backtest and rests on the holdout (Decision 4).

---

## Finding 6: Other model families do no better (OLS, neural net, XGBoost)

**Question.** Would a different method have ranked major incidents better than
gradient boosting? Three alternatives: plain OLS (the linear probability model), a
small neural net, and XGBoost.

**How it was tested.** `code/modeling/model_comparison.py`. Each alternative runs
through the same rolling backtest as the ladder: same folds, same tuning on the last
training year, same main inputs, no safety weight. The holdout is not used, so a
better alternative would be a new decision made before any rerun. The rule is the
one every variant faces (Decision 6): an alternative goes forward only if it beats
gradient boosting clearly, on the same test incidents.
* **OLS**: least squares on the logistic's one-hot inputs, scores clipped to [0, 1].
* **Neural net**: scikit-learn's feed-forward network on the same inputs, tuned over
  3 sizes (8; 32; 32 then 16 units) and 2 penalty strengths. Early stopping is off,
  because it would hold out random rows, not a later year.
* **XGBoost**: 300 trees, like gradient boosting, and a grid that mirrors its grid
  (depth 2/3/4 for 3/7/15 leaves, leaf weight 5/10 for 100/200 incidents).

**Results** (backtest, mean of 2021-2023; difference with gradient boosting, by fold,
with a 95% day-bootstrap interval):

| Model | PR-AUC | ROC-AUC | Top 5% major | Top 10% major / safety / minutes | Calibration error | PR-AUC vs gradient boosting, by fold |
|---|---|---|---|---|---|---|
| Gradient boosting | 0.230 | 0.736 | 28.3% | 38.1% / 39.6% / 16.2% | 0.013 | |
| XGBoost | 0.236 | 0.738 | 28.0% | 39.3% / 40.9% / 15.3% | 0.015 | +0.016 [0.000, 0.034] / 0.000 / +0.004 |
| Neural net | 0.232 | 0.737 | 28.9% | 38.4% / 39.7% / 14.4% | 0.016 | +0.011 / -0.002 / -0.004 |
| Logistic | 0.221 | 0.735 | 28.8% | 37.7% / 39.2% / 14.6% | 0.013 | (Decision 4) |
| OLS | 0.196 | 0.736 | 27.1% | 37.8% / 39.2% / 14.3% | 0.020 | -0.021 / -0.024 / -0.057 |

**Reading.**
* **XGBoost ties gradient boosting.** It is +0.007 PR-AUC on average, and no fold is
  clearly better (2021 sits on the edge of its interval). The two rank incidents
  almost identically (rank correlation 0.97 on the test incidents): two
  implementations of the same idea, on the same 7 inputs, find the same splits.
* **The neural net is a smoothed logistic regression.** Tuning picked the smallest
  network (8 units) in all three folds, and its ranking is closer to the logistic's
  (rank correlation 0.97) than to gradient boosting's (0.90). It is +0.002 on
  average, inside the noise in every fold.
* **OLS is clearly worse, but only at the very top.** It is clearly behind gradient
  boosting in 2 of 3 folds and behind logistic regression in all 3 (-0.025 /
  -0.016 / -0.035, every interval below zero). Its ROC-AUC and top-10% capture equal
  the logistic's; the loss is in the top 5%. A linear model in the probability adds
  risks: an incident with two risky traits gets the sum of their effects. The
  logistic multiplies odds, which is how the riskiest combinations behave, so it
  ranks the very top better. OLS also scores about 9% of test incidents below zero
  (clipped), and is the worst calibrated.
* **The limit is the inputs, not the method.** Five methods of very different
  flexibility land within 0.04 PR-AUC, and the four that handle the top of the
  ranking well land within 0.015. What is known when the incident is reported
  carries only so much signal (Decision 6 makes the same point for a Bayesian model).

**Decision.** Gradient boosting stays the chosen model and logistic regression the
benchmark. No alternative clears the rule, so the holdout is not rerun.

---

## Finding 7: Next month's major incidents per line: a VAR does not help

**Question.** A time-series model such as a VAR cannot score single incidents, so it
cannot join the comparison in Finding 6. Its natural question is a different one: how
many major incidents will each line have next month? Does a VAR forecast that better
than simple baselines?

**How it was tested.** `code/exploration/08_monthly_counts.ipynb` looks at the monthly
counts first; `code/modeling/forecast.py` runs the forecasts. The test years are the
same as the incident model's (2021, 2022, 2023). Each month is forecast one month
ahead, refitted every month on all earlier months, from 2019. The holdout is not used.
A multi-line incident counts on each line it involves. Six forecasters:
* **Long-run mean** and **last-12-month mean** of the line.
* **Same month last year.**
* **Seasonal Poisson**: Poisson regression on line + calendar month.
* **AR(1)**: each line regressed on its own last month.
* **VAR(1)**: every line regressed on every line's last month. The only difference
  with the AR(1) is the cross-line terms, so VAR(1) minus AR(1) is the test of what a
  VAR adds. Only one lag: with 24 to 59 past months, each extra lag adds 4 numbers
  per line.

**What the counts look like** (notebook 08, 2019-2023):
* **Small and close to pure chance.** About 1 major incident a month on yellow, 3 on
  blue, 7 on orange and 9 on green. Variance is 0.9 to 1.8 times the mean; random
  (Poisson) counts give 1.
* **No reliable seasonality.** The calendar month does not improve a Poisson fit with
  a level per line (likelihood ratio 14.1 on 11 df, p = 0.23).
* **Some memory within a line, none across.** A line's count correlates with its own
  last month at 0.23 (green) and 0.31 (orange), near zero on blue and yellow. One
  line's last month against another line's this month: between -0.14 and 0.18, with
  no pattern.

**Results** (error per line and month, mean of the three test years; test-year average
4.9 major incidents per line-month). The difference is with the last-12-month mean, by
test year; negative is better:

| Forecaster | MAE | RMSE | MAE difference by year | Years clearly better / worse |
|---|---|---|---|---|
| AR(1) | 2.03 | 2.76 | -0.09 / -0.08 / -0.09 | 0 / 0 |
| Long-run mean | 2.07 | 2.81 | -0.11 / 0.00 / -0.01 | 0 / 0 |
| Last-12-month mean | 2.11 | 2.87 | (reference) | |
| Seasonal Poisson | 2.20 | 2.96 | -0.04 / +0.28 / +0.02 | 0 / 0 |
| VAR(1) | 2.23 | 3.03 | +0.17 / +0.12 / +0.08 | 0 / 0 |
| Same month last year | 2.59 | 3.53 | +0.33 / +0.81 / +0.31 | 0 / 1 |

VAR(1) minus AR(1): +0.26 [-0.03, 0.51] / +0.20 [-0.03, 0.44] / +0.17 [-0.03, 0.33].
The 95% intervals resample whole test months, so the four lines of a month stay together.

**Reading.**
* **The VAR is worse than the AR(1) in all three years.** Each interval only just
  includes zero. Its cross-line terms fit noise, as notebook 08 predicted, and that
  noise costs accuracy.
* **Nothing beats a flat average clearly.** The AR(1) is best, by about 0.08 a month,
  in every year, but inside the noise each time. Its edge comes from the memory on
  green and orange.
* **Seasonal forecasts hurt.** Same month last year is the worst forecaster, and
  clearly so in 2022; the seasonal Poisson adds nothing.
* **Most of the error cannot be forecast.** If each line's true monthly rate were
  known exactly, pure Poisson chance would still leave an error of about 1.7 a
  month (test-year rates). The best forecaster, at 2.0, is not far above that.
  Next month's count is mostly chance around a line's usual level.

**What this changes.** Nothing in the incident model. For planning, a line's recent
average is as good a forecast of its major incidents as any model tried. The useful
lever is the incident-level ranking (Findings 4 and 5), not a monthly forecast.

---

## Outputs for the README

`python code/main.py` rebuilds everything from the raw files (about 5 minutes).
Figures are drawn by `code/reporting/figures.py` from the tables in
`outputs/backtest/`, which are also their table view.

| Figure (`outputs/figures/`) | Shows | Table behind it (`outputs/backtest/`) |
|---|---|---|
| `capture_curve.png` | % of major incidents, safety responses and delay minutes caught by escalating the riskiest 1-30% (holdout) | `decision_curve.csv` |
| `model_ladder.png` | PR-AUC of each model: three backtest years and the holdout with its interval | `main_metrics.csv`, `holdout_metrics.csv` |
| `calibration.png` | Predicted probability vs observed rate in 10 groups (holdout) | `holdout_reliability.csv` |
| `cost_grid.png` | Net delay minutes saved per year under each cost assumption | `decision_cost_grid.csv` |
| `risk_by_symptom.png` | Share of train incidents that became major, by first symptom (2019-2023) | computed from the cleaned data |
| `risk_by_location.png` | The same by kind of location (2019-2023) | computed from the cleaned data |

Other tables: `main_comparisons.csv` and `holdout_comparisons.csv` (paired model
comparisons), `calibration_*.csv` (Decisions 5 and 9), `sensitivity_*.csv` (Finding 2),
`holdout_calibration.csv` (Finding 4), `decision_budgets.csv` (Finding 5),
`alternatives_*.csv` (Finding 6), `forecast_*.csv` (Finding 7).
