"""The README figures, drawn from the saved results in outputs/backtest/.

Every figure is built by a `plot_*` function that takes a table and returns a matplotlib
Figure, so it can be checked on its own; make_figures() loads the saved tables and
writes each figure to outputs/figures/. The CSV files next to them are the table view
of every figure.

Figures:
    capture_curve       what escalating the riskiest X% of incidents catches (holdout)
    model_ladder        PR-AUC of each model: backtest folds and the final holdout
    calibration         predicted probability vs observed rate (holdout)
    cost_grid           net delay minutes saved per year under each cost assumption
    risk_by_symptom     share of major incidents by first symptom (2019-2023)
    risk_by_location    share of major incidents by kind of location (2019-2023)

Style: one light surface, recessive grid, thin marks, text in ink colours (never in a
series colour), a legend for two or more series plus direct labels. The three series
colours are the first three slots of the validated reference palette (blue, orange,
aqua); aqua is below 3:1 contrast on the surface, so its line is always direct-labelled.
"""

import matplotlib

matplotlib.use("Agg")   # draw to files only, never to a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from modeling.features import build_features
from modeling.target import add_targets, model_population

# ---------------------------------------------------------------------------
# Palette and style
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#a3a29c"                     # reference lines and de-emphasised marks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "font.family": "sans-serif",
}

# Stakeholder-facing names for the French symptom labels and the location kinds.
SYMPTOM_NAMES = {
    "Clientèle": "Customer",
    "Divers": "Other",
    "Exploitation stations": "Station operations",
    "Exploitation trains": "Train operations",
    "Feu, fumée, odeur, produit, etc...": "Fire, smoke, odour",
    "Matériel roulant": "Rolling stock",
    "Équipements fixes": "Fixed equipment",
    "unknown": "Unknown",
}
LOCATION_NAMES = {
    "station": "Station",
    "transfer_platform": "Transfer platform",
    "between_stations": "Between stations",
    "ancillary_structure": "Ancillary structure",
    "facility": "Yard / shop / facility",
    "unspecified": "Unspecified (incl. whole line)",
}
MODEL_NAMES = {
    "base_rate": "Base rate",
    "symptom_line_table": "Symptom x line\ntable",
    "logistic": "Logistic\nregression",
    "gradient_boosting": "Gradient\nboosting",
}

# Groups smaller than this are left out of the risk-by-category figures.
MIN_GROUP_SIZE = 50


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_capture_curve(curve: pd.DataFrame) -> plt.Figure:
    """Share of major incidents, safety responses and delay minutes caught by budget."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        budget = 100 * curve["budget"]
        # (column, legend label, short end-of-line label, vertical nudge of that label in
        # points: the safety and major lines end close together).
        series = [
            ("major_caught", "Major incidents", "Major", -7),
            ("safety_caught", "Safety responses", "Safety", 7),
            ("minutes_caught", "Delay minutes (lower bound)", "Minutes", 0),
        ]
        for (column, label, short_label, nudge), colour in zip(series, SERIES):
            values = 100 * curve[column]
            ax.plot(budget, values, color=colour, label=label)
            ax.annotate(short_label, (budget.iloc[-1], values.iloc[-1]), textcoords="offset points",
                        xytext=(6, nudge), va="center", fontsize=9, color=TEXT_SECONDARY)
        ax.plot(budget, budget, color=NEUTRAL, linestyle="--", linewidth=1, label="Random choice")

        # Mark the two alert budgets with what they catch.
        for budget_share in (0.05, 0.10):
            row = curve.loc[np.isclose(curve["budget"], budget_share)].iloc[0]
            ax.axvline(100 * budget_share, color=GRID, linewidth=1, zorder=0)
            ax.scatter(100 * budget_share, 100 * row["major_caught"], s=40, color=SERIES[0],
                       edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f"{100 * row['major_caught']:.0f}%", (100 * budget_share, 100 * row["major_caught"]),
                        textcoords="offset points", xytext=(-6, 8), ha="right", fontsize=9,
                        color=TEXT_PRIMARY)

        ax.set_xlim(0, 30)
        ax.set_ylim(0, 80)
        ax.set_xlabel("% of train incidents escalated (riskiest first)")
        ax.set_ylabel("% caught")
        ax.set_title("What escalating the riskiest incidents catches (holdout 2024-2025)", loc="left")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.subplots_adjust(right=0.88)
    return fig


def plot_model_ladder(backtest: pd.DataFrame, holdout: pd.DataFrame) -> plt.Figure:
    """PR-AUC for each model: the three backtest folds and the holdout with its interval."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        models = list(MODEL_NAMES)
        positions = np.arange(len(models))
        holdout = holdout[holdout["period"] == "2024-2025"].set_index("model").loc[models]

        for position, model in zip(positions, models):
            folds = backtest.loc[backtest["model"] == model, "pr_auc"]
            ax.scatter(np.full(len(folds), position - 0.12), folds, s=30, color=NEUTRAL, zorder=2,
                       label="Backtest year (2021, 2022, 2023)" if position == 0 else None)
        ax.errorbar(positions + 0.12, holdout["pr_auc"],
                    yerr=[holdout["pr_auc"] - holdout["pr_auc_low"], holdout["pr_auc_high"] - holdout["pr_auc"]],
                    fmt="o", markersize=7, color=SERIES[0], elinewidth=2, capsize=0, zorder=3,
                    label="Final holdout 2024-2025 (95% interval)")
        for position, value in zip(positions, holdout["pr_auc"]):
            ax.annotate(f"{value:.2f}", (position + 0.12, value), textcoords="offset points",
                        xytext=(10, 0), va="center", fontsize=9, color=TEXT_PRIMARY)

        ax.set_xticks(positions, [MODEL_NAMES[model] for model in models])
        ax.set_ylim(0, 0.45)
        ax.set_ylabel("PR-AUC for major incidents")
        ax.set_title("Each model beats the simpler one below it", loc="left")
        ax.legend(loc="upper left")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
    return fig


def plot_calibration(reliability: pd.DataFrame) -> plt.Figure:
    """Mean predicted probability vs observed rate in 10 equal-size groups."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        limit = 0.25
        ax.plot([0, limit], [0, limit], color=NEUTRAL, linestyle="--", linewidth=1,
                label="Perfect calibration")
        ax.plot(reliability["mean_predicted"], reliability["observed_rate"], color=SERIES[0],
                marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                label="Model, 10 equal-size groups of incidents")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Predicted probability of a major incident")
        ax.set_ylabel("Observed share of major incidents")
        ax.set_title("Predicted probabilities track observed rates\n(holdout 2024-2025)", loc="left")
        ax.legend(loc="upper left")
        fig.tight_layout()
    return fig


def plot_cost_grid(grid: pd.DataFrame) -> plt.Figure:
    """Heat map of net delay minutes saved per year, by cost assumption."""
    table = grid.pivot(index="avoidable_share", columns="false_alarm_cost_minutes",
                       values="net_minutes_saved_per_year")
    escalated = grid.pivot(index="avoidable_share", columns="false_alarm_cost_minutes",
                           values="share_escalated")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        colour_map = matplotlib.colors.LinearSegmentedColormap.from_list("blue", SEQUENTIAL_BLUE)
        image = ax.imshow(table.to_numpy(), cmap=colour_map, vmin=0, vmax=table.to_numpy().max(),
                          origin="lower", aspect="auto")
        for row, share in enumerate(table.index):
            for column, cost in enumerate(table.columns):
                value = table.loc[share, cost]
                level = value / table.to_numpy().max()
                label = (f"{value:,.0f} min\n{100 * escalated.loc[share, cost]:.1f}% escalated"
                         if escalated.loc[share, cost] > 0 else "never pays\n(nothing escalated)")
                ax.text(column, row, label, ha="center", va="center", fontsize=9,
                        color="white" if level > 0.55 else TEXT_PRIMARY)
        ax.set_xticks(range(len(table.columns)), [f"{cost:g} min" for cost in table.columns])
        ax.set_yticks(range(len(table.index)), [f"{100 * share:.0f}%" for share in table.index])
        ax.set_xlabel("Cost of a false alarm (delay-minute equivalent)")
        ax.set_ylabel("Share of a major incident's\ndelay that escalation avoids")
        ax.set_title("Net delay minutes saved per year (lower bound)", loc="left")
        ax.grid(False)
        fig.colorbar(image, ax=ax, label="net minutes saved per year")
        fig.tight_layout()
    return fig


def plot_risk_by_group(rates: pd.DataFrame, title: str) -> plt.Figure:
    """Share of major incidents per group, with 95% intervals and group sizes."""
    rates = rates.sort_values("rate")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(rates) + 1.5))
        positions = np.arange(len(rates))
        ax.errorbar(rates["rate"], positions,
                    xerr=[rates["rate"] - rates["low"], rates["high"] - rates["rate"]],
                    fmt="o", markersize=7, color=SERIES[0], elinewidth=2, capsize=0)
        overall = rates.attrs["overall_rate"]
        ax.axvline(overall, color=NEUTRAL, linestyle="--", linewidth=1,
                   label=f"All train incidents ({100 * overall:.1f}%)")
        for position, rate, high, incidents in zip(positions, rates["rate"], rates["high"], rates["incidents"]):
            ax.annotate(f"{100 * rate:.1f}%  (n = {int(incidents):,})", (high, position),
                        textcoords="offset points", xytext=(6, 0), va="center", fontsize=8,
                        color=TEXT_SECONDARY)
        ax.legend(loc="lower right")
        ax.set_yticks(positions, rates.index)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlim(0, max(0.3, rates["high"].max() + 0.08))
        ax.set_xlabel("Share of train incidents that became major (2019-2023)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tables behind the risk figures
# ---------------------------------------------------------------------------
def major_rate_by(labelled: pd.DataFrame, groups: pd.Series, names: dict) -> pd.DataFrame:
    """Share of major incidents per group, with Wilson 95% intervals.

    Groups with fewer than MIN_GROUP_SIZE incidents are left out (their interval is
    too wide to read). The overall rate is kept in the table's attrs for the figure.
    """
    table = labelled.groupby(groups.map(names), observed=True)["y_major"].agg(["sum", "size"])
    table = table[table["size"] >= MIN_GROUP_SIZE]
    low, high = wilson_interval(table["sum"], table["size"])
    rates = pd.DataFrame({"incidents": table["size"], "rate": table["sum"] / table["size"],
                          "low": low, "high": high})
    rates.attrs["overall_rate"] = labelled["y_major"].mean()
    return rates


def wilson_interval(successes: pd.Series, trials: pd.Series) -> tuple[pd.Series, pd.Series]:
    """95% Wilson interval for a proportion (stays inside [0, 1] for small groups)."""
    z = 1.96
    p = successes / trials
    centre = (p + z**2 / (2 * trials)) / (1 + z**2 / trials)
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / (1 + z**2 / trials)
    return centre - half, centre + half


# ---------------------------------------------------------------------------
# Writing every figure
# ---------------------------------------------------------------------------
def make_figures(incidents: pd.DataFrame) -> list:
    """Draw every figure from the saved results and write it to outputs/figures/.

    `incidents` are the cleaned incidents; the risk figures use train incidents of
    2019-2023 only (before the holdout). Returns the paths written.
    """
    results = config.BACKTEST_DIR
    labelled = add_targets(model_population(incidents))
    training = labelled[labelled["service_date"].dt.year <= 2023]
    features = build_features(training)

    figures = {
        "capture_curve": plot_capture_curve(pd.read_csv(results / "decision_curve.csv")),
        "model_ladder": plot_model_ladder(pd.read_csv(results / "main_metrics.csv"),
                                          pd.read_csv(results / "holdout_metrics.csv")),
        "calibration": plot_calibration(pd.read_csv(results / "holdout_reliability.csv")),
        "cost_grid": plot_cost_grid(pd.read_csv(results / "decision_cost_grid.csv")),
        "risk_by_symptom": plot_risk_by_group(
            major_rate_by(training, features["symptom"].astype(str), SYMPTOM_NAMES),
            "Fire and rolling-stock problems are the riskiest symptoms"),
        "risk_by_location": plot_risk_by_group(
            major_rate_by(training, features["location_kind"].astype(str), LOCATION_NAMES),
            "Incidents between stations or on transfer\nplatforms are riskier"),
    }

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, figure in figures.items():
        path = config.FIGURES_DIR / f"{name}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)
    return paths
