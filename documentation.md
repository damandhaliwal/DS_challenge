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

**Delay minutes at the bucket's lower bound.** A 10-14 min incident counts as 10, a 30+
min incident as 30. This needs no assumption about where a delay falls inside its
bucket, or how long the open-ended 30+ bucket runs (recorded durations there have a
median of 52 min). Every total is therefore an **"at least"** figure. On this measure,
major incidents are 5.6% of train incidents but **31.9% of delay minutes**
(2019-2023, before the final holdout).

**Decision rule.**
* Main view: escalate the top **5%** and **10%** of incidents by predicted risk, and
  report the safety events and delay minutes caught.
* CFO sensitivity view: escalating pays off when
  P(major) x avoidable share x delay minutes of a major incident > false-alarm cost.
  Neither assumption can be measured from the data, so both span a wide range:
  avoidable share 10% / 25% / 50%, false-alarm cost 0.5 / 1 / 2 delay-minute
  equivalents. These ranges are illustrative, not estimates.

---

## Decision 3: Model inputs, and two leak suspects held out

**Rule.** Inputs are a whitelist of what is known when an incident is reported
(`code/modeling/features.py`). Everything set afterwards (causes, rolling stock, end
time, delay, evacuation, emergency team, power cut, damage) is excluded, and a test
checks that none of them, nor any label, is an input.

**Main inputs.** Symptom, line group, location kind, at or next to a terminal, at or
next to a transfer station, door involved, emergency brake, time of day (service-day
clock bands) and weekend.

**Two leak suspects, held out of the main model.** Both carry a strong signal, but the
data does not say *when* they are recorded:

| Field | Signal (train incidents, 2019-2023) | Timing doubt |
|---|---|---|
| Run number present | major rate 6.6% with, 1.4% without; the gap holds within every symptom (fire / smoke: 21.9% vs 6.6%) | recorded at report time, or added after a train was held? |
| Location "whole line / network" | major rate 12.4% (vs 5.3% at a station) | known at report time, or only once the incident has spread? |

The main model leaves both out (whole-line locations become "unspecified"). A
sensitivity run adds them back and reports the difference. A large gain is a question
for STM ("is this logged in real time?"), not a reason to trust the field.

**Left out on purpose.** The station itself (68 values with few severe events each; the
terminal and transfer flags keep its most robust signal), car and door numbers, month
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
| Symptom x line table | 0.137 | 15% | 25% | 27% | 14% |
| Logistic regression | 0.217 | 29% | 38% | 39% | 14% |
| Gradient boosting | 0.237 | 29% | 39% | 41% | 15% |

**Paired differences in PR-AUC** (same resampled days for both models):

| Comparison | 2021 | 2022 | 2023 |
|---|---|---|---|
| Table - base rate | +0.106 [0.072, 0.148] | +0.058 [0.041, 0.090] | +0.076 [0.051, 0.107] |
| Logistic - table | +0.064 [0.032, 0.102] | +0.079 [0.051, 0.117] | +0.097 [0.057, 0.148] |
| Gradient boosting - logistic | +0.017 [-0.005, 0.037] | +0.015 [-0.002, 0.037] | +0.028 [0.001, 0.053] |

**Decision.** Gradient boosting is the model that goes forward to calibration and the
final holdout. It is ahead of logistic regression in every fold and in 94-98% of the
resamples, although the gain is small (about +0.02 PR-AUC) and the interval excludes
zero only in 2023; at the 5% budget the two catch the same share of major incidents.
Logistic regression is therefore reported next to it as the **explainable benchmark**:
the extra complexity buys a little, and the reader can see exactly how much.

**What the numbers mean for stakeholders.** Escalating 1 incident in 10 flags about 39%
of major incidents and 41% of safety responses, against 10% by chance. Only about 15%
of delay minutes are in that group: a model aimed at major events does not chase the
many 5-9 min delays, which is the trade-off shown in notebook 03.

---

## Decision 5: Calibration with a Platt curve

**Why.** The models are trained with safety responses weighted 2x, so their scores rank
incidents well but overstate the chance of a major incident (about 0.10 on average,
against a real rate of about 0.057). A probability given to a stakeholder must be true
on average.

**How** (`code/modeling/calibrate.py`). In each backtest fold the model is trained on
the training years minus the last one; a calibrator learns score -> share of major
incidents on that last training year, without weights; it is then applied to the test
year scores of the model refitted on all training years. No test-year data is used.

**Results** (gradient boosting, averaged over the 2021-2023 folds):

| Method | Brier | Log loss | Mean predicted | Observed | Calibration error |
|---|---|---|---|---|---|
| None (raw scores) | 0.0512 | 0.206 | 0.101 | 0.057 | 0.044 |
| Platt | 0.0490 | 0.193 | 0.065 | 0.057 | 0.017 |
| Isotonic | 0.0490 | 0.195 | 0.067 | 0.057 | 0.017 |
| Constant training rate (reference) | 0.0536 | | | | |

**Decision.** Platt. It ties with isotonic, and has 2 parameters, a smooth curve and no
ties. Calibration does not change the ranking, so every capture figure stays the same.

**Remaining bias.** Probabilities still run about 0.8 points high on average. The
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
| Main | 0.237 | | 39.2% | 41.1% | 14.7% |
| + leak suspects | 0.252 | +0.014 (1 / 0) | 40.0% | 41.6% | 16.0% |
| Safety weight 1x | 0.233 | -0.004 (0 / 0) | 39.1% | 40.6% | 14.9% |
| Safety weight 4x | 0.235 | -0.002 (0 / 0) | 38.9% | 40.7% | 14.7% |
| Trained on 5+ min, no safety weight | 0.092 | -0.145 (0 / 3) | 14.8% | 14.5% | 18.6% |

**What it shows.**
* **Leak suspects add little** (+0.014 PR-AUC, clear in one fold of three). The main
  model is not missing much by leaving them out, so the question of when STM records
  them is worth asking but not decisive.
* **The safety weight does not matter** between 1x and 4x: no fold differs clearly.
  Safety responses are already 94% of major incidents, so weighting them more
  changes little. The 2x choice is harmless rather than important.
* **The trade-off is real and large.** A model trained on the usual 5+ min threshold
  finds a third as many major incidents and safety responses, and gains only about 4
  points of delay minutes (18.6% vs 14.7%). Even that model puts less than a fifth of
  delay minutes in its top 10%.

---

## Decision 6: No Bayesian model for the main prediction

**Question.** Should the risk model be Bayesian, so that each prediction comes with its
own uncertainty?

**Short answer.** Not for the main model. A Bayesian model helps when an estimate rests
on little data, and the main model's estimates do not. The one place it would help is
the station-level effect, which is left as a next step.

**Why it would change little here** (train incidents, 2019-2023):
* **The estimates rest on a lot of data.** 21,058 incidents and 1,192 major ones, spread
  over 9 coarse inputs. Of the 45 symptom x line cells, the ones with fewer than 100
  incidents hold only 1.8% of incidents. With this much data, a Bayesian fit with weak
  priors gives almost the same probabilities as the current fit.
* **The decision uses one number.** An incident is escalated when its probability is
  above a cut-off. An interval around that probability does not change the action,
  except for incidents right at the cut-off. The uncertainty that matters for the
  conclusions is in the metrics, and the day-level bootstrap already measures it
  (Decision 4, Finding 2).
* **It would not rank incidents better.** Ranking quality is limited by what is known at
  report time, not by the model: gradient boosting beats logistic regression by only
  about 0.02 PR-AUC (Decision 4). A Bayesian model sees the same inputs.

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

## Finding 3: Delay minutes are a weak headline metric, and hard to predict

**Question.** The main model puts only about 15% of delay minutes in its top 10%. Is
delay minutes a good metric for this data, and can a model aimed at minutes catch more?

**Is it a good metric?** (train incidents 2019-2023, minutes at the bucket's lower bound)
* The shortest bucket (2 min and less) is 56% of incidents and counts **zero** minutes.
* The share of minutes from major incidents is **31.9%** if 30+ incidents count 30 min,
  and **45.1%** if they count their recorded duration. The headline moves by 13 points
  on an assumption the data cannot settle.
* **13.7%** of safety responses count zero minutes, and 23.1% count fewer than 5.

**Can a model aimed at minutes do better?** Rolling backtest, average of 2021-2023:

| Ranking | Top 10%: delay minutes | Top 10%: major | Top 10%: safety | Deviance explained |
|---|---|---|---|---|
| Major model (main) | 14.7% | 39.2% | 41.1% | |
| 5+ model | 18.6% | 14.8% | 14.5% | |
| Symptom mean minutes | 13.3% | 12.0% | 11.2% | 3.6% |
| Minutes model (Poisson gradient boosting) | 19.0% | 16.4% | 15.8% | 8.9% |

The minutes model gains about 4 points of minutes over the main model and loses about
23 points of major incidents. It explains under 9% of the deviance: what is known at
report time says little about how many minutes an incident will cost.

**Decision.** Safety responses and major incidents stay the target and the headline.
Delay minutes are reported as a secondary measure, always as a lower bound with the
30+ assumption stated. A classifier aimed at delays (5+ or 10+) can be added later as
a separate operations view, with this trade-off shown next to it.

**Reproduce.** `code/exploration/05_delay_minutes.ipynb`.

---

## Decision 7: Final holdout protocol (written before the holdout was run)

Fixed before any model saw 2024 or 2025. The holdout is run **once**; nothing below is
changed after seeing its result.

* **Data.** Train on train incidents of 2019-2023; test on 2024-01-01 to 2025-09-30
  (2025 ends in September, so it is a partial year).
* **Model.** Gradient boosting on the main inputs, target `y_major`, safety weight 2x
  (Decisions 2-4). Its setting is tuned exactly as in the backtest: trained on
  2019-2022, scored on 2023, best setting refitted on 2019-2023.
* **Calibration.** Platt (Decision 5): the tuned model trained on 2019-2022 scores 2023,
  the Platt curve is fitted on those scores, and applied to the holdout scores of the
  model trained on 2019-2023.
* **Benchmarks.** Base rate, symptom x line table and logistic regression, each tuned
  and trained the same way, with paired comparisons (day-level bootstrap) between each
  model and the one below it.
* **Reported.** For 2024-2025 together, and for 2024 and 2025 separately: PR-AUC with
  its 95% interval; the share of major incidents, safety responses and delay minutes
  (lower bound) in the top 5% and 10% of risk; and for the calibrated probabilities the
  Brier score (against a constant forecast of the 2019-2023 rate), the mean predicted
  vs observed rate, and the calibration error.
* **Known risks, stated in advance.** The share of rolling-stock symptoms confirmed as
  rolling-stock causes fell sharply in 2024 (notebook 04), and the major-incident rate
  has moved year to year (7.1%, 5.3%, 4.7% in 2021-2023), so calibration may be off in
  the holdout even if the ranking holds.

---

## Finding 4: Final holdout results (2024-2025)

Run once, following Decision 7. Train incidents 2019-2023 for training; 8,109 train
incidents from 2024-01-01 to 2025-09-30 for testing (major-incident rate 5.2%).

**Ranking** (95% intervals from the day-level bootstrap):

| Model | PR-AUC | Top 5%: major | Top 10%: major | Top 10%: safety | Top 10%: delay minutes |
|---|---|---|---|---|---|
| Base rate | 0.052 | 5.0% | 10.0% | 10.0% | 10.0% |
| Symptom x line table | 0.204 | 20.3% | 30.8% | 32.9% | 16.5% |
| Logistic regression | 0.290 | 33.9% | 41.4% | 43.4% | 17.1% |
| **Gradient boosting** | **0.313 [0.270, 0.364]** | **35.3%** | **42.9% [39.2, 47.3]** | **45.0%** | **17.7%** |

| Gradient boosting by year | 2024 | 2025 (Jan-Sep) |
|---|---|---|
| PR-AUC | 0.323 | 0.310 |
| Top 10%: major | 41.6% | 44.4% |
| Top 10%: safety | 43.7% | 46.5% |

**Paired differences in PR-AUC:** table - base rate +0.152 [0.122, 0.187]; logistic -
table +0.085 [0.057, 0.118]; gradient boosting - logistic **+0.023 [0.011, 0.033]**,
better in every resample. In the backtest that last gain was uncertain; on the
holdout it is clear.

**Calibration** (Platt): mean predicted probability 4.6% against an observed rate of
5.2%; calibration error 0.007; Brier score 0.042 against 0.050 for a constant forecast
of the 2019-2023 rate (about 15% better). The probabilities run slightly **low** this
time: the major-incident rate rose from 4.7% in 2023 (the year the curve was fitted
on) to 5.2%. The middle of the risk range is under-predicted most (groups 3-6:
about 2% predicted against 2-5% observed); the top 10% is accurate (22.7% predicted,
22.5% observed).

**Reading.**
* The holdout is **better** than the backtest average (PR-AUC 0.313 vs 0.237; 42.9% vs
  39.2% of major incidents in the top 10%). The likely reason is more training data
  (five years against two to four), and nothing was tuned on the holdout. The fair
  claim is the range: **about 40% of major incidents and 45% of safety responses in
  the top 10% of incidents, about 4x chance**, stable across 2021-2025.
* The risk stated in advance (the 2024 shift in how rolling-stock incidents are
  confirmed) did not hurt the ranking: 2024 is in line with the other years.
* Delay minutes stay secondary: 17.7% in the top 10% (Finding 3).

---

## Finding 5: What escalation would deliver (decision layer, holdout 2024-2025)

Calibrated probabilities of the chosen model on the holdout (8,109 train incidents in
1.75 years: about 243 major incidents and 225 safety responses a year). Computed by
`code/modeling/decision.py`.

**Alert budgets (main view, VP Operations).**

| Escalate | Probability cut-off | Escalations | Major caught | Safety caught | Delay minutes caught | Escalations per major found |
|---|---|---|---|---|---|---|
| Top 5% | 12.6% | about 230 a year (4-5 a week) | 35.3% (about 86 a year) | 37.1% | 14.4% | 2.7 |
| Top 10% | 9.2% | about 460 a year (9 a week) | 42.9% (about 104 a year) | 45.0% | 17.7% | 4.5 |

Wider budgets keep adding major incidents but more slowly: 15% catches 51%, 20%
catches 56%, 30% catches 67%. The top 1% (about 46 escalations a year) already catches
13% of major incidents, and 2 of every 3 escalations there are major.

**Cost grid (sensitivity view, CFO).** Escalate when P(major) > false-alarm cost /
(avoidable share x 14.6 min), where 14.6 is the average lower-bound delay of a major
incident in 2019-2023. Net delay minutes saved per year (lower bound):

| Avoidable share \ false-alarm cost | 0.5 min | 1 min | 2 min |
|---|---|---|---|
| 10% | 78 (2% escalated) | 49 (0.5%) | 0 (never pays) |
| 25% | 321 (5%) | 233 (3%) | 138 (0.6%) |
| 50% | 694 (14%) | 641 (5%) | 465 (3%) |

**Reading.**
* Escalation is **cheap to staff**: the 10% budget is about 9 escalations a week, and
  1 in 4.5 of them turns out major.
* **In delay minutes alone, the gain is small.** The holdout years average about 12,600
  delay minutes a year (lower bound), 4,100 of them from major incidents. Even the
  most generous assumptions save about 690 net minutes a year (5.5% of all minutes);
  the middle one (25% avoidable, 1-minute false alarm) saves about 230 (under 2%).
  With a high false-alarm cost and a low avoidable share, escalating never pays.
* So the case for the model rests on **safety**, not on minutes: escalating 1 incident
  in 10 puts about 45% of safety responses, about 100 a year, in front of the team
  early. This matches Findings 3 and 4, and should lead the recommendation to all
  three stakeholders, with minutes as the secondary, conservative figure.

---

## Outputs for the README

`python code/main.py` rebuilds everything from the raw files (about 4 minutes).
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
comparisons), `calibration_*.csv` (Decision 5), `sensitivity_*.csv` (Finding 2),
`holdout_calibration.csv` (Finding 4), `decision_budgets.csv` (Finding 5).
