# Documentation

A record of analytical findings and the decisions they led to. Each entry says what
was asked, how it was tested, what came out, and what it changes in the project.
Numbers come from the cleaned data (`code/cleaning/`) and the labels in
`code/modeling/target.py`; "major" means `y_major` (a safety response or a delay of
30+ min).

**Where to start.** The record is long because it keeps every decision and both
results whenever something was changed. The short version:

| Question | Answer | Entry |
|---|---|---|
| What is predicted? | Whether a train incident becomes major, from seven things known when it is reported | Decisions 2 and 3 |
| How well? | Escalating the riskiest 1 incident in 10 catches about 40% of major incidents and 45% of safety responses, about 4x chance (holdout: 43.7% and 46.2%) | Findings 4 and 5 |
| With which model? | Gradient boosting. Logistic regression is close; OLS, a neural net and XGBoost do no better | Decision 4, Finding 6 |
| What does it depend on? | The symptom first, then the door flag, the terminal flag and the emergency brake. Without the two flags, whose recording time is assumed, it catches about 30% in the backtest, not 38% | Decisions 3 and 8, Finding 2 |
| What does it miss? | Customer incidents (a quarter of major incidents, none escalated) and long delays without a safety response | Finding 9 |
| What would improve it? | A second score for incidents still delaying service after 5 and 10 minutes | Finding 10 |
| How should reliability be measured, and what are the trends made of? | Incidents of 5+ min per million planned km, by line. Volume explains little; short incidents drive the counts; 5+ min incidents rose in 2023; major incidents did not | Finding 8 |
| What should STM do? | Measure, prioritise, collect, and the dollar view | Recommendations |
| What was changed after results were seen? | Reported with before and after results | Decisions 8, 9 and 10 |

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

**Kept in the main model, with the same doubt stated.** The door flag and the
emergency brake have no time stamp either. They stay in because each describes how an
incident starts (a door is involved; a customer pulled the emergency brake), which
is what the control room is told first, while a run number, a whole-line location or
a list of door numbers can be filled in later. That is an assumption; the data cannot
confirm it. The symptom is on firmer ground: the data dictionary defines it as the
cause given at the moment of the incident. A sensitivity run drops both flags
(Finding 2, added in Decision 10): the top 10% then holds 29.8% of major incidents,
not 38.1%, about 3x chance, not 4x. So "are these two flags entered when the incident
is reported?" is the first question to put to STM before the model is used.

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

## Finding 2: Sensitivity runs: which choices change the conclusions

Each variant reruns the gradient-boosting backtest with one choice changed
(`code/modeling/sensitivity.py`), and is compared with the main run on the same test
incidents (paired, day-level bootstrap). Averages over the 2021-2023 folds:

| Variant | PR-AUC (major) | Difference vs main (folds clearly better / worse) | Top 10%: major | Top 10%: safety | Top 10%: delay minutes |
|---|---|---|---|---|---|
| Main (no safety weight) | 0.230 | | 38.1% | 39.6% | 16.2% |
| + leak suspects (run number, whole line, door numbers) | 0.251 | +0.021 (1 / 0) | 40.5% | 42.1% | 18.1% |
| - door flag and emergency brake | 0.158 | -0.071 (0 / 3) | 29.8% | 31.0% | 16.4% |
| Safety weight 2x | 0.236 | +0.006 (0 / 0) | 38.0% | 39.5% | 14.7% |
| Safety weight 4x | 0.228 | -0.002 (0 / 0) | 36.8% | 38.2% | 15.0% |
| Trained on 5+ min | 0.099 | -0.131 (0 / 3) | 15.7% | 15.5% | 18.4% |

**What it shows.**
* **Leak suspects add a little** (+0.021 PR-AUC, clear in one fold of three, 2022).
  The main model is not missing much by leaving them out, so the question of when STM
  records them is worth asking but not decisive.
* **The two flags carry a lot** (added in Decision 10). Without the door flag and the
  emergency brake the model is clearly worse in every fold (-0.069, -0.058, -0.087),
  and the top 10% holds 29.8% of major incidents, not 38.1%. The model still ranks
  about 3x better than chance on symptom, line, location and time alone, but about
  a fifth of the major incidents it catches depend on the assumption that both flags
  are known when the incident is reported (Decision 3). This is the one choice here that changes the
  size of the result.
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
calibration and the delay-minutes count. Decision 10 added read-outs of the same
holdout scores, with no rerun and no result changed. Decision 11 tested a different
question, the second look, on the same holdout under its own protocol, reusing the
main model's scores unchanged; Finding 11.)

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
* **Late-arriving incidents are not a concern for this extract.** The data dictionary
  warns that train incidents can appear more than 30 days after a month closes. The
  files are dated 2026-07-07 (their names, `config.py`) and the last incident is on
  2025-09-30, so the last months had about nine months to fill in.
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
* **What is inside that 10%, and what is not,** is in Finding 9; the conversion of
  these minutes into dollars, and the two numbers it needs from STM, is in the
  Recommendations (part 4).

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

## Finding 8: How to measure reliability, and what the trends are made of

**Question.** The brief says direction cannot tell whether the trends in the data come
from reliability, service volume, reporting practices or classification consistency,
and asks how STM should measure reliability. This entry has no model: it counts.

**How it was tested.** `code/reporting/reliability.py` builds two tables from the
cleaned incidents and the line-month table (`code/cleaning/build_datasets.py`): one row
per year for the network, and one row per line and year. Rates are per million planned
km. The rates by line come with an exact 95% interval for a count (Poisson), because a
small line has few incidents a year and its rate moves by chance. Planned km ends on
2023-09-30: rates on real km stop there; later months use the carried-forward km and
are marked "est.". A multi-line incident counts on each of its lines in the line
table and once in the network table.

**The network, year by year** (`reliability_trend_by_year.csv`; train incidents; 2025
is January to September, and the 2023 rates are January to September, the months with
real km):

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| Planned km (million) | 89.9 | 88.0 | 84.3 | 84.4 | 61.6 (9 months) | none | none |
| Train incidents | 5,081 | 3,709 | 3,564 | 4,207 | 4,497 | 5,063 | 3,046 |
| Share under 5 min | 81.0% | 78.5% | 77.6% | 78.7% | 75.6% | 77.9% | 75.8% |
| Incidents of 5+ min | 964 | 799 | 800 | 896 | 1,097 | 1,121 | 736 |
| 5+ min per million km | 10.7 | 9.1 | 9.5 | 10.6 | 13.2 | 13.5 (est.) | 12.0 (est.) |
| Major incidents | 262 | 246 | 254 | 221 | 209 | 263 | 162 |
| Major per million km | 2.9 | 2.8 | 3.0 | 2.6 | 2.6 | 3.2 (est.) | 2.6 (est.) |
| Recorded duration inside its delay bucket | 98.5% | 98.0% | 97.9% | 97.7% | 97.6% | 97.5% | 97.1% |
| First symptom confirmed by the final cause | 77.3% | 75.0% | 78.1% | 77.0% | 76.2% | 69.3% | 77.2% |
| Same, rolling-stock symptoms only | 53.7% | 52.5% | 52.1% | 48.3% | 44.3% | 28.4% | 46.2% |

**Reading, one explanation at a time.**
* **Service volume explains almost nothing.** Planned km fell 6% from 2019 to 2021 and
  then stayed flat. Over the same years the number of incidents fell 30%, and then
  rose 42% by 2024. Planned km is the schedule, not the trains actually run or the
  riders carried: in 2020 it fell 2% while incidents fell 27%.
* **The total count moves with short incidents.** Incidents under 5 minutes are 76% to
  81% of every year. They are 89% of the fall from 2019 to 2021 (1,353 of 1,517) and
  79% of the rise from 2021 to 2024 (1,178 of 1,499). They are also the incidents
  whose recording changed in April 2025 (notebook 07). The count of all incidents is
  the weakest measure of reliability in this data.
* **Interruptions of 5+ minutes have become more frequent since 2023.** 9.1 to 10.7 per
  million km in 2019-2022, 13.2 in January-September 2023, then 13.5 and 12.0 on
  estimated km. The counts say the same without any km: 1,097 and 1,121 in 2023 and
  2024, against 799 to 964 before. This is the part of the trend that looks like
  reliability.
* **The most serious incidents have not.** Major incidents stay between 2.6 and 3.0
  per million km in 2019-2023 (3.2 and 2.6 afterwards on estimated km), with no trend.
* **Reporting drifts slowly, and jumped once.** The share of recorded durations that
  agree with the published delay bucket falls a little every year, from 98.5% to
  97.1%. The one sharp break is April 2025: about a third fewer train incidents,
  mostly under 5 minutes (notebook 07).
* **Classification was different in 2024.** A rolling-stock first symptom is confirmed
  by the final cause 44% to 54% of the time in the other years, and 28% in 2024. The
  other symptoms stay in their usual range (notebook 04). Counts *by cause* for 2024
  are not comparable with other years; counts by delay or by safety response do not
  use the cause.

**By line** (`reliability_rates_by_line_year.csv`, summed over the months with real km,
2019-01 to 2023-09):

| Line | Share of planned km | 5+ min per million km | Major per million km | Delay minutes per million km |
|---|---|---|---|---|
| Orange | 48.7% | 8.6 | 2.0 | 156 |
| Green | 36.2% | 13.2 | 3.7 | 241 |
| Blue | 9.4% | 12.6 | 4.9 | 228 |
| Yellow | 5.7% | 10.0 | 2.5 | 197 |

* **Volume and reliability rank the lines differently.** Orange runs about half the
  planned km and has nearly as many incidents as green (8,079 against 8,396), but the
  lowest rate on every measure. Blue has the highest rate of major incidents, more
  than twice orange's.
* **Green, 2023.** 18.3 incidents of 5+ min per million km [16.5, 20.2] in
  January-September 2023, against 11.5 to 13.2 in 2019-2022, whose intervals end at
  14.6 at most: a clear rise. On estimated km it is 14.9 in 2024 and 14.5 in 2025,
  below the 2023 peak and above every year before it.
* **Yellow, 2024.** 144 incidents of 5+ min, against 37 to 70 in the other full years;
  65 of them in one month, September 2024 (line-month table). The rate picks this up
  (25.3 per million estimated km [21.4, 29.8]); a network count would not.
* **Small lines need long windows.** A yearly rate on yellow has an interval of about
  plus or minus a third (2022: 9.4 [6.8, 12.7]), and a monthly rate is mostly noise:
  yellow has a median of 3 incidents of 5+ min a month (notebook 02).

**What this changes: the indicator to use.** Recommendations, part 1, below. Figure:
`outputs/figures/reliability_rates.png`.

**Limits.** Planned km is not delivered km, and it barely moved during COVID (notebook
02). The km file ends two years before the incidents. The yellow line's planned km
rose by about a third in January 2023, so its rates before and after are not on the
same footing.

---

## Finding 9: What the model escalates, and what it misses

**Question.** "The riskiest 10% holds about 40% of major incidents" is one number.
Which major incidents are in it, and which never are? And since "major" covers very
different events, from a single car evacuated to a train and its station evacuated,
does the model find the heavier ones?

**How it was tested.** `code/modeling/escalation_mix.py` reads the saved scores of the
chosen model, in the backtest (test years 2021-2023, each year escalating its own
riskiest 10%) and on the holdout. No model is trained. Both questions were set on the
backtest; the holdout columns are read-outs of the same, unchanged scores
(Decision 10). `target.py` now keeps what the safety response was
(`safety_response_kind`); it is reported, not predicted.

**By first symptom** (`escalation_by_symptom.csv`):

| First symptom | Share of major incidents (backtest / holdout) | Share of escalations (backtest / holdout) | Its major incidents escalated: backtest | holdout |
|---|---|---|---|---|
| Rolling stock | 43.7% / 43.3% | 57.1% / 67.0% | 54.5% | 60.3% |
| Customer | 25.1% / 28.0% | 2.0% / 0.0% | **0 of 172** | **0 of 119** |
| Fire, smoke, odour | 15.2% / 16.5% | 28.8% / 26.4% | 77.9% | 94.3% |
| Fixed equipment | 12.4% / 11.3% | 9.7% / 6.2% | 12.9% | 18.8% |
| All other symptoms | 3.5% / 0.9% | 2.5% / 0.5% | 3 of 24 | 0 of 4 |

* **The model is a rolling-stock and fire detector.** Those two symptoms are about
  59% of major incidents and 86% to 93% of what it escalates.
* **It never escalates a customer incident that turns out major.** Customer incidents
  are a quarter of major incidents. Only 3.0% of them become major (172 of 5,747 in
  the backtest years, against 5.6% for all train incidents), and nothing known at
  report time separates those from the rest, so their risk never reaches the top 10%
  (a probability of about 10%, Finding 5).
* **A finer customer code built from today's fields would not fix it**
  (`escalation_by_customer_cause.csv`). The secondary cause is recorded after the
  incident. Even if it were known at the start, its three large groups become major
  0.7% of the time (accidental nuisance, 8 of 1,122), 3.1% (deliberate act, 105 of
  3,383) and 5.2% (injured or ill, 52 of 1,002): none is near 10%. What separates a
  major customer incident is not in the data at all (Recommendations, part 3).

**By kind of outcome** (`escalation_by_outcome.csv`; the first four columns are the
backtest years):

| Outcome | Incidents | Under 5 min | 30+ min | Mean delay (min) | Escalated: backtest | Escalated: holdout |
|---|---|---|---|---|---|---|
| Train and station evacuated | 134 | 0.7% | 76.9% | 55.8 | 45.5% | 72.6% (of 84) |
| Station evacuated | 19 | 57.9% | 5.3% | 8.1 | 52.6% | 30.0% (of 10) |
| Train evacuated | 415 | 17.1% | 19.0% | 19.8 | 36.1% | 39.2% (of 248) |
| Car evacuated | 17 | 29.4% | 0% | 6.9 | 11.8% | 0% (of 4) |
| Emergency team only | 62 | 66.1% | 6.5% | 6.8 | 48.3% | 43.2% (of 48) |
| 30+ min, no safety response | 37 | 0% | 100% | 58.0 | 13.5% | 12.9% (of 31) |
| **All major incidents** | 684 | | | | 37.7% | 43.7% |
| **All incidents of 30+ min** | | | | | 31.7% | 39.1% |

(The backtest figure for all major incidents pools the three years; the average of
the three yearly figures is 38.1%, as in Decision 4.)

* **The major label is not one level of severity.** A train and its station evacuated
  together is the heavy event: 77% last 30+ min, 56 minutes on average. A single car
  evacuated never lasts 30+ min. Two-thirds of "emergency team only" incidents delay
  service under 5 minutes, which STM's own indicators would not count.
* **The model finds the heavy kind at least as well as the label as a whole.** It
  escalates 45.5% of train-and-station evacuations in the backtest (37.7% for all
  major incidents) and 72.6% on the holdout (43.7%). The holdout figure rests on 84
  incidents, so the fair claim is "about half, or better".
* **It does not find long delays that have no safety response.** 13% of them are
  escalated in both periods, barely above the 10% of a random choice. Most of them
  start with a customer or fixed-equipment symptom (notebook 04), the two symptoms
  the model cannot rank.
* **Incidents of 30+ min as a whole:** 31.7% escalated in the backtest and 39.1% on
  the holdout, a little below the major label.

**Decision.** The major label stays. It follows the priority agreed before modelling
(safety first), and changing a target after seeing results would need a new protocol
and a new holdout run. What is inside it is now reported next to it, so a reader can
see that "major" is mostly evacuations of a train, that almost a quarter of safety
responses delay service under 5 minutes (Finding 3), and that the model does best on
the heaviest ones.

---

## Finding 10: A second score after 5 or 10 minutes (backtest only)

**Question.** The model scores an incident once, when it is reported. Most incidents
are over within a few minutes. Does scoring again the incidents that are still
delaying service find the long ones better?

**How it was tested.** `code/modeling/rescoring.py`, on the backtest folds only; the
holdout is not used.
* **Pools.** Train incidents with a delay of at least 5 minutes, and of at least 10.
  Both come from the delay bucket, which is exact at those bounds.
* **Target.** A delay of 30+ min, not the major label. The data does not say when an
  evacuation was called, so by minute 5 or 10 it may already have happened, and
  "predicting" it would be reading the past. A delay of 30+ min is always still ahead.
* **Inputs.** The same seven report-time inputs. The only new information is that the
  incident is still going on.
* **Rankings.** The pool's rate for everyone; the first score reused as it is; a
  report-time 30+ score (the main model trained on every incident for the 30+ target,
  then reused on the pool); a logistic regression and a gradient boosting retrained on
  the pool, tuned as in the main backtest. Each retrained model is compared with both
  reused scores on the same incidents (paired, day-level bootstrap).
* **Why two reused scores** (added after this entry was first written). The first score
  aims at the major label, so a gain over it mixes the new target with the new pool.
  The report-time 30+ score has the new target but not the pool, so the gain splits
  in two: report-time 30+ score minus first score is what the target adds, retrained
  minus report-time 30+ score is what knowing "still delaying service" adds.

**The pools** (`rescoring_pools.csv`, test years 2021-2023):

| Stage | Incidents a year | Share of train incidents | Holds: incidents of 30+ min | Holds: major incidents | Share that goes on to 30+ min |
|---|---|---|---|---|---|
| After 5 minutes | about 930 | 21% to 24% | all | 78% to 84% | 8.2% |
| After 10 minutes | about 295 | 7% to 8% | all | 57% to 66% | 25.8% |

**Results** (`rescoring_summary.csv`, mean of the three folds; the difference is with
the first score, by fold):

| Stage | Ranking | PR-AUC | Riskiest 10% of the pool: share that is 30+ min | Share of all 30+ min incidents in it | PR-AUC difference: 2021 / 2022 / 2023 |
|---|---|---|---|---|---|
| After 5 min | Pool rate | 0.082 | 8.2% | 10.0% | |
| | First score, reused | 0.189 | 24.5% | 30.0% | |
| | Report-time 30+ score, reused | 0.259 | 22.3% | 28.1% | +0.057 [0.002, 0.113] / +0.088 [0.042, 0.141] / +0.068 [-0.020, 0.164] |
| | Logistic, retrained | 0.328 | 31.7% | 38.3% | +0.115 [0.050, 0.177] / +0.243 [0.175, 0.332] / +0.058 [-0.035, 0.151] |
| | Gradient boosting, retrained | 0.314 | 32.0% | 38.8% | +0.121 [0.050, 0.195] / +0.208 [0.146, 0.295] / +0.048 [-0.041, 0.155] |
| After 10 min | Pool rate | 0.258 | 25.8% | 10.0% | |
| | First score, reused | 0.352 | 39.4% | 15.9% | |
| | Report-time 30+ score, reused | 0.432 | 51.7% | 20.7% | +0.073 [0.010, 0.141] / +0.111 [0.058, 0.179] / +0.055 [-0.036, 0.148] |
| | Logistic, retrained | 0.485 | 68.8% | 26.9% | +0.112 [0.028, 0.204] / +0.236 [0.158, 0.318] / +0.048 [-0.063, 0.162] |
| | Gradient boosting, retrained | 0.464 | 64.4% | 25.4% | +0.087 [0.019, 0.158] / +0.197 [0.124, 0.278] / +0.050 [-0.050, 0.164] |

**What the pool adds** (retrained minus the report-time 30+ score, same incidents):

| Stage | Ranking | Mean | 2021 / 2022 / 2023 |
|---|---|---|---|
| After 5 min | Logistic | +0.068 | +0.058 [0.003, 0.115] / +0.156 [0.090, 0.218] / -0.010 [-0.062, 0.041] |
| | Gradient boosting | +0.055 | +0.064 [0.001, 0.132] / +0.120 [0.069, 0.187] / -0.020 [-0.059, 0.033] |
| After 10 min | Logistic | +0.053 | +0.040 [-0.028, 0.114] / +0.125 [0.061, 0.190] / -0.007 [-0.072, 0.059] |
| | Gradient boosting | +0.032 | +0.014 [-0.053, 0.073] / +0.086 [0.024, 0.142] / -0.006 [-0.069, 0.062] |

**Reading.**
* **Waiting 5 minutes shrinks the problem to under a quarter and loses no long
  incident.** Every incident of 30+ min is still in the pool, with about four in five
  major incidents.
* **About half the gain is the target, half the pool.** After 5 minutes the retrained
  logistic beats the first score by 0.139 PR-AUC on average. Aiming at 30+ from report
  time already gives 0.071 of it, clearly in 2021 and 2022. The other 0.068 is what
  knowing "still delaying service" adds: clear in 2021 and 2022, nothing in 2023.
  After 10 minutes the pool's part is smaller (+0.053) and clear in 2022 only. So the
  second look helps in two years of three, and the size of its gain was overstated
  when it was measured against the major-label score alone.
* **Fewer escalations, as many long incidents.** A report-time model aimed at 30+
  escalates about 410 incidents a year (the riskiest 10%) and holds 39.3% of the
  incidents of 30+ min (backtest mean; 37% to 43% by year). The second look after 5
  minutes escalates about 93 a year, a tenth of the pool, and holds 38% to 39%. It
  catches about as many long incidents with under a quarter of the escalations. The
  price is five minutes. (Against the main model, which aims at the major label, it
  is 31.7% at report time, Finding 9; that comparison mixes in the target.)
* **After 10 minutes it is close to a real alarm.** The riskiest 10% of that pool is
  about 29 incidents a year, and about two in three of them go on to 30+ min.
* **A simple model is enough here.** Logistic regression is at least as good as
  gradient boosting on average at both stages (the two were not compared as a pair).
  The pool has about 76 incidents of 30+ min a year, too few for anything richer.

**Limits.** This is a backtest, not a holdout result. The inputs are still the
report-time ones: what the control room knows at minute 5 (was an evacuation called,
is the train moving, are passengers on board) is not in the data. With it the second
score would do better, and it could aim at the major label.

**What this changes.** Nothing in the main model. It is the next step to propose:
escalate in two stages, a first score when the incident is reported and a second for
the incidents still delaying service at 5 and 10 minutes. It should be tested on the
holdout, under a protocol written first, before it is claimed (Decision 11). On the
holdout it was not confirmed (Finding 11).

---

## Decision 10: What the second review changed, and what it did not

**Why.** A second review of the finished pipeline listed what was still missing: the
reliability half of the brief, what is inside the major label, what the model misses,
a second score, and the timing of two inputs.

**Nothing in the main model changed.** Same label, inputs, tuning, model and holdout
protocol. `python code/main.py` was rerun after the changes: every table of the main
backtest, the calibration, the model comparison, the forecast, the holdout and the
decision layer is identical to the version before, and so are the six earlier
figures (git shows no change in those files). The sensitivity tables gained the rows
of one new variant; their earlier rows are identical.

**Added.**
1. **Reliability indicators** (Finding 8). No model.
2. **What is escalated and what is missed** (Finding 9). These tables include new
   read-outs of the holdout. They were chosen on backtest evidence (no customer
   incident among 172 escalated; 45% of train-and-station evacuations escalated),
   they are computed from the unchanged holdout scores, and they are reported
   whatever they showed. No choice depends on them.
3. **A second score at 5 and 10 minutes** (Finding 10). Backtest only.
4. **A sensitivity run without the door flag and the emergency brake** (Decision 3,
   Finding 2): the cost of being wrong about when they are recorded.

**Checked and closed.**
* **Late-arriving incidents** (Finding 4): not a concern for this extract.
* **Dollars** (Recommendations, part 4): a formula and the two numbers STM has to
  supply. No dollar figure is given, because the data has none to build it from.

**Left as it is, on purpose.**
* **The major label** (Finding 9).
* **Delay minutes stay hard to rank** (Finding 3): 15.8% in the top 10% on the
  holdout. That is what the report-time data allows, not a setting to fix.

---

## Decision 11: Holdout protocol for the second look (written before it is run)

**Why.** Finding 10 was designed while reading the backtest: the two stages, the 30+
target, the second reference score and the preference for logistic regression were
all chosen on 2021-2023. A result found that way flatters itself. The 2024-2025
holdout has never been used for the second look, so it can test it fairly, once,
under rules fixed here. Nothing below is changed after seeing the result.

**The one question that decides.** After 5 minutes, does a logistic regression
retrained on the incidents still delaying service rank the 30+ min incidents better
than the report-time 30+ score reused on the same incidents? This is "what the pool
adds" in Finding 10, the part that is new. Everything else below is reported, but
does not decide.

* **Data.** As in Decision 7: train on train incidents of 2019-2023, test on
  2024-01-01 to 2025-09-30. Pools are the incidents with a delay of at least 5 min,
  and of at least 10 min (delay bucket, as in Finding 10).
* **Target.** `y_30plus`.
* **Rankings**, each tuned as in the holdout (trained on 2019-2022, scored on 2023,
  chosen setting refitted on 2019-2023):
    * **report-time 30+ score:** gradient boosting on every train incident, target
      `y_30plus`, the seven main inputs, then reused on the pool;
    * **first score:** the holdout scores of the main model (Finding 4), reused as they
      are, no rerun;
    * **logistic regression, retrained on the pool** (the decisive model; chosen
      because it was at least as good as gradient boosting in the backtest and is
      simpler);
    * **gradient boosting, retrained on the pool**, and the pool's rate, reported
      alongside.
* **Decision rule**, on 2024-2025 together, PR-AUC of the retrained logistic minus the
  report-time 30+ score after 5 minutes, 95% interval from the paired day-level
  bootstrap:
    * interval entirely above 0: **confirmed**; the recommendation says so and keeps
      the live test (shadow run) as the next step;
    * interval includes 0: **not confirmed**; the recommendation says "promising in
      the backtest, not confirmed on 2024-2025" and the second look stays an idea
      for the shadow run, not a finding;
    * interval entirely below 0: **rejected**; the recommendation is withdrawn.
* **Reported whatever it shows.** For 2024-2025 together and for 2024 and 2025
  separately: the pools; PR-AUC of every ranking at both stages; every paired
  difference of Finding 10 (each retrained model against both reused scores, and
  report-time 30+ score minus first score); and the efficiency read-out: escalations
  a year and share of all 30+ min incidents caught by the riskiest 10% of the pool,
  next to the same for the riskiest 10% of all incidents under the report-time 30+
  score. The efficiency read-out is descriptive; it does not decide.
* **Nothing else moves.** The main model, its holdout scores and every earlier table
  stay as they are. Whatever the result, the stages, target and models above are not
  changed and the run is not repeated.

**Known risks, stated in advance.**
* **Small sample.** From the backtest, the pool after 5 minutes has about 930
  incidents and 76 of 30+ min a year, so about 1,600 and 130 over the 21 holdout months
  (estimated from 2021-2023, not counted on the holdout). A single backtest year gave
  intervals about +/-0.06 wide; the holdout should be somewhat narrower. A gain the size
  of the backtest mean (+0.068) would likely show; one the size of 2023 (-0.010) would
  not. "Not confirmed" would mean "cannot tell", not "does not work".
* **One backtest year in three showed nothing** (2023), so a null result is plausible.
* **April 2025 recording change** (notebook 07). The incidents that disappeared are
  mostly under 5 minutes, so the pools should change little, but 2025 is reported on
  its own.
* **What the holdout cannot show.** The pool is built after the fact from the final
  delay; live, the control room must know at minute 5 that the incident is still
  going on, on the same clock. And a better ranking is not proof that escalating
  changes the outcome. Both need the live test in the recommendations.

---

## Finding 11: The second look on the holdout: not confirmed

Following Decision 11, run once (`rescoring.run_rescoring_holdout`, tables
`rescoring_holdout_*.csv`). Train incidents 2019-2023 for training, 2024-01-01 to
2025-09-30 for testing.

**The deciding comparison.** After 5 minutes, logistic regression retrained on the
pool minus the report-time 30+ score reused on the same incidents, PR-AUC, 2024-2025:
**-0.000 [-0.039, +0.042]**, better in 51% of resamples. By the rule of Decision 11:
**not confirmed**.

**The pools** (`rescoring_holdout_pools.csv`): as in the backtest. After 5 minutes,
1,857 incidents (23% of train incidents), holding 86% of major incidents and every
one of the 156 incidents of 30+ min; 8.4% of the pool goes on to 30+ min (8.2% in
the backtest). After 10 minutes, 625 incidents (8%), 70% of major incidents, 25.0% go
on to 30+ min.

**Results** (2024-2025; PR-AUC with its 95% interval; the riskiest 10% of each group):

| Stage | Ranking | PR-AUC | Share of all 30+ min incidents caught | Escalations a year |
|---|---|---|---|---|
| At report | First score (major label) | 0.161 [0.110, 0.237] | 39.1% | about 463 |
| | Report-time 30+ score | 0.172 [0.115, 0.245] | 40.5% | about 463 |
| After 5 min | Pool rate | 0.084 | 10.0% | about 106 |
| | First score, reused | 0.252 [0.188, 0.335] | 35.9% | about 106 |
| | Report-time 30+ score, reused | 0.281 [0.206, 0.363] | 34.0% | about 106 |
| | **Logistic, retrained** | **0.281 [0.216, 0.359]** | **39.9%** | about 106 |
| | Gradient boosting, retrained | 0.292 [0.215, 0.374] | 34.6% | about 106 |
| After 10 min | Report-time 30+ score, reused | 0.424 [0.352, 0.513] | 22.8% | about 36 |
| | Logistic, retrained | 0.426 [0.356, 0.518] | 22.4% | about 36 |

**Paired differences in PR-AUC** (`rescoring_holdout_comparisons.csv`, 2024-2025):

| Stage | Comparison | Difference |
|---|---|---|
| After 5 min | Logistic - report-time 30+ score (**decides**) | -0.000 [-0.039, +0.042] |
| | Gradient boosting - report-time 30+ score | +0.011 [-0.020, +0.039] |
| | Logistic - first score | +0.029 [-0.021, +0.081] |
| | Report-time 30+ score - first score (the target) | +0.029 [-0.013, +0.064] |
| After 10 min | Logistic - report-time 30+ score | +0.001 [-0.042, +0.047] |
| | Gradient boosting - first score | +0.058 [+0.004, +0.112] |

By year, the deciding difference is -0.000 in 2024 and +0.000 in 2025. Of the 30
paired comparisons, five intervals lie above zero and none below, none of them the
deciding one: four are against the first score (gradient boosting at 5 minutes in
2025; at 10 minutes, gradient boosting in 2024-2025 and 2025, and logistic in 2025),
and one against the report-time 30+ score (gradient boosting at 5 minutes in 2025,
+0.060 [+0.015, +0.109]). That is a little more than chance alone would give, so
gradient boosting may do somewhat better in 2025 and at 10 minutes. But the five
share incidents and models, four include the target effect, and none was set in
advance to count, so they are a lead for the live test, not a result.

**Reading.**
* **Retraining on the pool adds nothing on 2024-2025.** The point estimate is zero,
  and the interval's upper end (+0.042) is below the backtest's mean gain (+0.068).
  So this is not only a small sample: the backtest gain was mostly the years it was
  chosen on, as the risk stated in Decision 11 (2023 showed nothing) allowed for.
* **Aiming at 30+ min is not clearly better either** (+0.029, interval includes 0),
  so the "half target, half pool" split of Finding 10 does not hold on new years.
* **What does hold is the filter itself** (descriptive, not tested, Decision 11).
  Waiting 5 minutes removes three in four incidents and no incident of 30+ min.
  Escalating the riskiest tenth of what is left is about 106 incidents a year,
  against about 463 at report time, and it held 34% to 40% of the 30+ min incidents
  depending on the ranking, against 39% to 41% at report time. The gap between the
  rankings in that column was not tested and is within what the PR-AUC intervals
  allow, so the claim is only "far fewer escalations for a similar share".
* **Nothing else changes.** The main model, its holdout scores and every earlier
  table are identical (`python code/main.py` rerun; git shows only the new files).

**What this changes.** The recommendation follows the rule set in advance: the
second score is "promising in the backtest, not confirmed on 2024-2025", and stays
an idea for the live test, not a finding. Finding 10 is kept as it was, as the
backtest result.

---

## Recommendations to STM

The three things the brief asks for, plus the dollar view. Each line points to the
entry that supports it.

**1. How to measure reliability** (Finding 8)
* **Headline:** train incidents of 5+ minutes per million planned km, by line, over a
  rolling 12 months, shown with its interval. It removes service volume, follows
  STM's own 5-minute convention, and is the measure least exposed to changes in what
  gets recorded.
* **Next to it:** major incidents per million km (safety), and delay minutes per
  million km from the recorded delay (burden).
* **Not the count of all incidents.** Four in five are under 5 minutes, and that is
  where recording changes show up.
* **Causes on their own page.** The way first symptoms are confirmed changed in 2024,
  so a cause count mixes reliability with classification.

**2. How to prioritise action**
* **When an incident is reported:** escalate the riskiest 1 in 10, about 9 a week. On
  the holdout that group held 44% of major incidents, 46% of safety responses and 73%
  of the evacuations of a train and its station (Findings 4, 5 and 9). The case rests
  on safety; in delay minutes the gain is small (Finding 5).
* **Know what it cannot see:** customer incidents and long delays without a safety
  response (Finding 9). A quiet score there is not reassurance.
* **A second score at 5 and 10 minutes: promising in the backtest, not confirmed on
  2024-2025** (Findings 10 and 11). Retraining on the incidents still delaying service
  ranked them no better than the report-time score. What held up is simpler: by
  minute 5 three in four incidents are over and no long one is, so a second look at
  what is left needs far fewer escalations (about 106 a year against 463) for a
  similar share of the 30+ min incidents. That is worth trying in the live test, with
  the time each incident is still delaying service logged as it happens.
* **By line:** blue has the highest rate of major incidents, green the highest rate of
  5+ minute incidents and a clear rise in 2023; orange is the most reliable per km
  (Finding 8). A monthly forecast of major incidents adds nothing to a line's recent
  average (Finding 7).

**3. What to collect**

| Add or confirm | Why | Evidence |
|---|---|---|
| The time each field is entered: symptom, door flag, emergency brake, run number, location, door numbers | The model may only use what is known when the incident is reported, and the data cannot say what that is. Without the two flags the model drops from about 38% to 30% of major incidents in the top 10%; with the three held-out fields it gains 2 points | Decision 3, Finding 2 |
| The time an evacuation or the emergency team was called | A second score could then aim at safety responses, and the warning time could be measured | Finding 10 |
| Riders affected by each incident (or the load of the trains held) | Delay minutes weigh a full train at rush hour like an empty one at 1 a.m.; this is also the missing term for a dollar figure | Finding 5, part 4 below |
| What is happening in a customer incident, at report time | A quarter of major incidents, none escalated; the causes recorded today do not separate them even after the fact. Worth testing: whether someone is on the track, whether the train is stopped between stations | Finding 9 |
| Delivered km, and planned km after 2023-09 | Rates stop in 2023-09, and planned km barely moved in 2020 | Finding 8, notebook 02 |
| Harm to people (injury, person struck) | The data has no such field, so "safety" is measured by STM's response, half of which is a broken-down train being emptied | `target.py`, notebook 04 |
| A dated note whenever recording or classification rules change | April 2025 (short incidents) and 2024 (rolling-stock causes) can only be guessed at from the data | notebook 07, Finding 8 |

**4. In dollars (CFO).** The cost of an incident is

    delay minutes x riders delayed x value of a rider's minute  +  direct costs

(direct costs: staff, the evacuation itself, damage). The data has the first term
only. With STM's own riders-delayed figure and the value of time it already uses in
business cases, the cost grid of Finding 5 converts directly: net delay minutes saved
a year (about 400 in the middle case, about 1,600 at most) times riders delayed per
incident times the value of a minute. No dollar figure is given here: both missing
numbers would be guesses.

---

## Outputs for the README

`python code/main.py` rebuilds everything from the raw files (about 8 minutes).
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
| `reliability_rates.png` | Incidents of 5+ min per million planned km, by line and year, with 95% intervals; hollow marks use estimated km | `reliability_rates_by_line_year.csv` |

Other tables: `main_comparisons.csv` and `holdout_comparisons.csv` (paired model
comparisons), `calibration_*.csv` (Decisions 5 and 9), `sensitivity_*.csv` (Finding 2),
`holdout_calibration.csv` (Finding 4), `decision_budgets.csv` (Finding 5),
`alternatives_*.csv` (Finding 6), `forecast_*.csv` (Finding 7), `reliability_*.csv`
(Finding 8), `escalation_*.csv` (Finding 9), `rescoring_*.csv` (Finding 10),
`rescoring_holdout_*.csv` (Finding 11).
