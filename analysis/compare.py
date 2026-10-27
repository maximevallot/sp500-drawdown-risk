import os

import numpy as np
import pandas as pd

from backtest import THRESHOLD, monthly_returns, portfolio_stats, score, trading_rule
from results_table import empirical, normal

# One comparison of every method, like for like: the Efficacy slide's table and chart.
# Written by Claude Code at Maxime's request (2026-10-18).
#
# Every column is scored on the same 420 month-ends, 1991-01 to 2025-12, the same total-return
# target and the same trading rule, and every model that trains has the team's thirteen predictors
# (Kritzman, 2026-10-19; README.md, "Which RBP to use"). Pointed at the thirteen at
# Maxime's request (2026-10-21):
#   - empirical frequency and normal distribution: analysis/results_table.py's, as in the deck;
#     the empirical frequency is also the reference for the Brier skill;
#   - RBP, the logistic regression and their dates and outcomes: data/rbp/rbp13_compare_forecasts.csv
#     (analysis/rbp13_compare.py);
#   - the net, boosting and the penalized logit: data/ml/ml_forecasts_13.csv (analysis/ml.py thirteen);
#   - Timestral: data/timestral/forecasts.csv. It uses no predictors, only the index's own history.
# GARCH, EWMA and the VIX are no longer columns: Kritzman made them predictors (2026-10-19), and
# GJR-GARCH's volatility forecast is one of the thirteen. The nine- and eleven-predictor columns are
# dropped (README.md).
#
# Writes data/compare/forecasts_all.csv, data/compare/scores.csv,
# deck/figures/probabilities_by_method.png and deck/figures/efficacy_table.png.
#
# Run from the repository root, after analysis/rbp13_compare.py and analysis/ml.py thirteen:
#   .venv/bin/python analysis/compare.py

THRESHOLDS = [0.10, 0.20]  # warning thresholds: no method gets near Kritzman's 50% (Flynn, 2026-10-18)
FIT_WINDOW, MIN_PAST_FITS = 60, 24  # RBP's fit cut-off, as in analysis/rbp.py
COMMON = "data/rbp/rbp13_compare_forecasts.csv"  # RBP, the logit, the dates and the outcomes
ML_13 = "data/ml/ml_forecasts_13.csv"
TIMESTRAL = "data/timestral/forecasts.csv"
DATES = 420  # 1991-01 to 2025-12

# Column order of the table, by family, RBP furthest right (Kritzman, 2026-10-19)
FAMILIES = {
    "Unconditional": ["Empirical frequency", "Normal distribution"],
    "Regression": ["Logistic regression"],
    "Machine learning": ["Neural net", "Boosting", "Logit, 13"],
    # Mistral AI's time-series foundation model, zero-shot (analysis/timestral_forecast.py)
    "Foundation model": ["Timestral, daily", "Timestral, monthly"],
    "RBP": ["RBP"],
}


def high_fit(fits, share):
    """RBP's rule (analysis/rbp.py): fit in the top `share` of the last FIT_WINDOW forecasts."""
    cutoff = fits.rolling(FIT_WINDOW, min_periods=MIN_PAST_FITS).quantile(1 - share).shift(1)
    return fits >= cutoff


def saved(path):
    """Forecasts saved by another script, by date."""
    return pd.read_csv(path, parse_dates=["Date"]).set_index("Date")


def runs(common):
    """Every method's forecasts on the common dates, as back-test results; none may be missing."""
    events = pd.read_csv("data/equity-index/sp500_tr_events.csv", parse_dates=["Date"]).set_index("Date")
    probs = {"Empirical frequency": empirical(events, common.index), "Normal distribution": normal(events, common.index),
             "Logistic regression": common["Logistic regression"], "RBP": common["RBP"]}
    ml = saved(ML_13)
    for name in FAMILIES["Machine learning"]:
        probs[name] = ml[name]
    if os.path.exists(TIMESTRAL):
        for name in FAMILIES["Foundation model"]:
            probs[name] = saved(TIMESTRAL)[name]
    out = {}
    for name, p in probs.items():
        p = p.reindex(common.index)
        assert p.notna().all(), f"{name} has no forecast on some of the {len(common)} dates"
        out[name] = pd.DataFrame({"Probability": p, "Event_A": common["Event_A"]})
    return out


def warned_months(results, warn, panel):
    """Kritzman's efficacy rows (2026-10-13) for the months a method warns in."""
    y = results["Event_A"]
    ret = panel["Return_3m"].reindex(results.index)
    return {
        "Months warned": int(warn.sum()),
        "Loss months warned (of all loss months)": int((warn & (y == 1)).sum()),
        "Share of warned months with a 10% loss": y[warn].mean() if warn.any() else np.nan,
        "False positives (warned, no loss)": int((warn & (y == 0)).sum()),
        "Average 3-month return when warned": ret[warn].mean() if warn.any() else np.nan,
        "Average 3-month return otherwise": ret[~warn].mean(),
    }


def table(runs_, reference, returns, panel, fit=None):
    """The Efficacy table: one column per method, the same rows for each.

    Rows: Brier skill and AUC; Kritzman's rows at each threshold (months warned, the share
    of them that lost 10%, false positives, average return when warned); and the trading
    rule, both the switch to 50/50 at the threshold and exposure scaled by the probability.
    fit: RBP's fit by date. RBP's top-fit variants warn only when the fit is high, and
    otherwise stay in stocks; they have no Brier score, AUC or scaled rule of their own.
    """
    returns = returns.reindex(reference.index).dropna()
    columns = {}
    variants = {}
    if fit is not None:
        for share in (0.50, 0.25):
            variants[f"RBP, top {share:.0%} fit"] = ("RBP", high_fit(fit, share).reindex(reference.index, fill_value=False))
    order = []
    for name in runs_:
        order.append(name)
        if name == "RBP":
            order += list(variants)
    for name in order:
        base, gate = variants.get(name, (name, None))
        r = runs_[base]
        p = r["Probability"]
        col = {}
        if gate is None:
            s = score(r, reference)
            col["Brier skill"], col["AUC"] = s["Brier skill"], s["AUC"]
        for thr in THRESHOLDS:
            warn = p >= thr
            if gate is not None:
                warn &= gate
            for row, value in warned_months(r, warn, panel).items():
                col[f"{row}, at {thr:.0%}"] = value
        warn = (p >= THRESHOLD) if gate is None else (p >= THRESHOLD) & gate
        switch = portfolio_stats(pd.Series(np.where(warn, 0.5, 1.0), index=p.index).loc[returns.index], returns)
        col["Return, switch at 10%"] = switch["Return"]
        col["Sharpe, switch"] = switch["Sharpe"]
        col["Max drawdown, switch"] = switch["Max drawdown"]
        if gate is None:
            scaled = trading_rule(r, returns)["Scaled"]
            col["Return, scaled"] = scaled["Return"]
            col["Sharpe, scaled"] = scaled["Sharpe"]
            col["Max drawdown, scaled"] = scaled["Max drawdown"]
        columns[name] = col
    static = trading_rule(reference, returns)["Static 75/25"]
    columns["Static 75/25"] = {"Return, scaled": static["Return"], "Sharpe, scaled": static["Sharpe"],
                               "Max drawdown, scaled": static["Max drawdown"]}
    stocks = portfolio_stats(pd.Series(1.0, index=returns.index), returns)
    columns["100% stocks"] = {"Return, switch at 10%": stocks["Return"], "Sharpe, switch": stocks["Sharpe"],
                              "Max drawdown, switch": stocks["Max drawdown"]}
    return pd.DataFrame(columns)


INK, MUTED, LINE, SHADE, SURFACE = "#0b0b0b", "#52514e", "#2a78d6", "#e4e2dc", "#fcfcfb"


def chart(forecasts, path):
    """Small multiples: each method's probability over time, loss months shaded, the 10% warning line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Logistic regression", "Logit, 13", "Neural net", "Boosting", "Timestral, daily", "RBP"]
    titles = {"Logit, 13": "Penalized logit", "Boosting": "Gradient boosting"}
    fig, axes = plt.subplots(3, 2, figsize=(13, 7.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    loss = forecasts.index[forecasts["Event_A"] == 1]
    for ax, name in zip(axes.T.flat, names):
        ax.set_facecolor(SURFACE)
        for d in loss:  # the month-end whose next three months lost 10% or more
            ax.axvspan(d - pd.Timedelta(days=15), d + pd.Timedelta(days=15), color=SHADE, lw=0)
        ax.axhline(THRESHOLD, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.plot(forecasts.index, forecasts[name], color=LINE, lw=1.5)
        ax.set_title(titles.get(name, name), loc="left", fontsize=12, color=INK, pad=4)
        ax.set_ylim(0, 0.7)
        ax.set_yticks([0, 0.25, 0.5], ["0%", "25%", "50%"])
        ax.tick_params(colors=MUTED, labelsize=10, length=0)
        ax.grid(axis="y", color=SHADE, lw=0.6)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(SHADE)
    fig.suptitle("Probability of a 10% loss over the next three months, by method, 1991–2025",
                 x=0.012, y=0.985, ha="left", fontsize=15, color=INK)
    fig.text(0.012, 0.925, "Grey bands: month-ends followed by a 10% or larger loss (total return).   "
             "Dashed line: the 10% warning threshold.", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=1.2, w_pad=2.5)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200)


# The efficacy figure: Kritzman's rows (2026-10-13), the methods as columns
FIGURE_COLUMNS = {
    "Normal distribution": "Normal\ndistribution", "Logistic regression": "Logistic\nregression",
    "Neural net": "Neural\nnet", "Boosting": "Gradient\nboosting", "Logit, 13": "Penalized\nlogit",
    "Timestral, daily": "Timestral\n(daily)", "RBP": "RBP\n(all)", "RBP, top 25% fit": "RBP\n(top 25% fit)",
}
# (section or None, label, scores row, format, which is best: "high", "low" or None)
FIGURE_ROWS = [
    ("When the model warns (probability of 10% or more)", None, None, None, None),
    (None, f"Months warned, of {DATES}", "Months warned, at 10%", "int", None),
    (None, "Share of them that lost 10% or more", "Share of warned months with a 10% loss, at 10%", "pct", "high"),
    (None, "False positives (warned, no loss)", "False positives (warned, no loss), at 10%", "int", None),
    (None, "Average 3-month return when warned", "Average 3-month return when warned, at 10%", "pct", "low"),
    (None, "Average 3-month return otherwise", "Average 3-month return otherwise, at 10%", "pct", None),
    ("Trading rule: 50/50 when warned, otherwise 100% stocks", None, None, None, None),
    (None, "Annual return", "Return, switch at 10%", "pct", "high"),
    (None, "Sharpe ratio", "Sharpe, switch", "num", "high"),
    (None, "Maximum drawdown", "Max drawdown, switch", "pct0", "high"),
    ("Stocks scaled down as the probability rises, 75% on average", None, None, None, None),
    (None, "Sharpe ratio", "Sharpe, scaled", "num", "high"),
    (None, "Maximum drawdown", "Max drawdown, scaled", "pct0", "high"),
    ("Forecast accuracy", None, None, None, None),
    (None, "AUC (0.5 = no skill)", "AUC", "num", "high"),
    (None, "Brier skill against the base rate", "Brier skill", "num3", "high"),
]


def efficacy_figure(scores, forecasts, panel, path):
    """The Efficacy table as an image, with a benchmark column and the best value in each row in bold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ret = panel["Return_3m"].reindex(forecasts.index)
    benchmark = {
        "Share of warned months with a 10% loss, at 10%": (forecasts["Event_A"].mean(), "all months"),
        "Average 3-month return when warned, at 10%": (ret.mean(), "all months"),
        "Return, switch at 10%": (scores.loc["Return, switch at 10%", "100% stocks"], "100% stocks"),
        "Sharpe, switch": (scores.loc["Sharpe, switch", "100% stocks"], "100% stocks"),
        "Max drawdown, switch": (scores.loc["Max drawdown, switch", "100% stocks"], "100% stocks"),
        "Sharpe, scaled": (scores.loc["Sharpe, scaled", "Static 75/25"], "static 75/25"),
        "Max drawdown, scaled": (scores.loc["Max drawdown, scaled", "Static 75/25"], "static 75/25"),
        "AUC": (0.5, "coin flip"),
        "Brier skill": (0.0, "base rate"),
    }
    fmt = {"int": lambda v: f"{v:.0f}", "pct": lambda v: f"{v:.1%}", "pct0": lambda v: f"{v:.0%}",
           "num": lambda v: f"{v:.2f}", "num3": lambda v: f"{v:.3f}"}

    methods = list(FIGURE_COLUMNS)
    label_w, col_w, bench_w, row_h, head_h = 3.3, 1.05, 1.75, 0.34, 0.75
    width = label_w + col_w * len(methods) + bench_w
    height = head_h + row_h * len(FIGURE_ROWS) + 0.25
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    xs = [label_w + col_w * (i + 0.5) for i in range(len(methods))]
    bench_x = label_w + col_w * len(methods) + bench_w / 2
    for x, m in zip(xs, methods):
        ax.text(x, head_h - 0.08, FIGURE_COLUMNS[m], ha="center", va="bottom", fontsize=10.5, color=INK,
                fontweight="bold", linespacing=1.15)
    ax.text(bench_x, head_h - 0.08, "Benchmark", ha="center", va="bottom", fontsize=10.5, color=MUTED,
            fontweight="bold")
    ax.plot([0.1, width - 0.1], [head_h, head_h], color=INK, lw=0.8)

    y = head_h
    for section, label, row, kind, best in FIGURE_ROWS:
        y += row_h
        if section:
            ax.text(0.12, y - row_h / 2, section, va="center", fontsize=10, color=MUTED, style="italic")
            continue
        ax.text(0.25, y - row_h / 2, label, va="center", fontsize=10.5, color=INK)
        values = scores.loc[row, methods].astype(float)
        target = None
        if best is not None and values.notna().any():
            target = values.max() if best == "high" else values.min()
        for x, v in zip(xs, values):
            text = "–" if pd.isna(v) else fmt[kind](v)
            bold = target is not None and not pd.isna(v) and np.isclose(v, target)
            ax.text(x, y - row_h / 2, text, ha="center", va="center", fontsize=10.5, color=INK,
                    fontweight="bold" if bold else "normal")
        if row in benchmark:
            v, what = benchmark[row]
            ax.text(bench_x, y - row_h / 2, f"{fmt[kind](v)}  {what}", ha="center", va="center",
                    fontsize=9.5, color=MUTED)
        ax.plot([0.1, width - 0.1], [y, y], color=SHADE, lw=0.6)
    fig.text(0.006, 0.004, f"{DATES} monthly forecasts, 1991–2025, 21 loss months in 12 episodes. S&P 500 total return. "
             "Bold: best in the row.", fontsize=9, color=MUTED, va="bottom")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    panel = saved(COMMON)  # the outcomes and the 3-month returns on the common dates
    assert len(panel) == DATES, len(panel)
    common = list(panel.index)
    all_runs = runs(panel)
    reference = all_runs["Empirical frequency"]
    returns = monthly_returns()

    order = [name for family in FAMILIES.values() for name in family if name in all_runs]
    rbp_fit = panel["Fit"]
    scores = table({name: all_runs[name] for name in order}, reference, returns, panel, fit=rbp_fit)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(f"{len(common)} forecasts, {common[0]:%Y-%m} to {common[-1]:%Y-%m}, "
          f"{int(reference['Event_A'].sum())} loss months")
    print(scores.to_string(float_format=lambda x: f"{x:.3f}"))

    # RBP's high-fit dates: are they easier for every method, or is RBP's fit informative?
    for share in (0.50, 0.25):
        dates = [d for d in common if high_fit(rbp_fit, share).get(d, False)]
        sub = {name: all_runs[name].loc[dates] for name in ["Empirical frequency", "Logistic regression", "Neural net", "Boosting", "RBP"]}
        rows = {}
        for name, r in sub.items():
            s = score(r, sub["Empirical frequency"])
            rows[name] = {"Brier skill": s["Brier skill"], "AUC": s["AUC"], "Hit rate at 10%": s["Hit rate"],
                          "False alarms at 10%": s["False alarms"]}
        print(f"\nOn RBP's top {share:.0%} by fit: {len(dates)} forecasts, "
              f"{int(sub['Empirical frequency']['Event_A'].sum())} loss months")
        print(pd.DataFrame(rows).to_string(float_format=lambda x: f"{x:.3f}"))

    os.makedirs("data/compare", exist_ok=True)
    forecasts = pd.DataFrame({"Event_A": reference["Event_A"],
                              **{name: all_runs[name]["Probability"] for name in order}})
    forecasts.to_csv("data/compare/forecasts_all.csv", float_format="%.6f")
    scores.to_csv("data/compare/scores.csv", float_format="%.6f")
    chart(forecasts, "deck/figures/probabilities_by_method.png")
    efficacy_figure(scores, forecasts, panel, "deck/figures/efficacy_table.png")
