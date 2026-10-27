import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import rbp

# The RBP evaluation for Kritzman: the efficacy table (without the trading rule) and an
# analysis of fit and of variable importance, from the forecasts analysis/rbp.py saved.
# Written by Claude Code at Flynn's request (2026-10-18).
#
# Reads data/rbp/rbp11_forecasts.csv and data/rbp/rbp11_importance.csv; writes
# data/rbp/rbp_evaluation.xlsx and prints the main tables.
#
# Run from the repository root, after analysis/rbp.py: python analysis/rbp_report.py

FORECASTS = "data/rbp/rbp11_forecasts.csv"
IMPORTANCE = "data/rbp/rbp11_importance.csv"
OUT = "data/rbp/rbp_evaluation.xlsx"
# Dates for per-forecast importance: the 2008 crash, the eve of COVID, its trough, and the 2022
# inflation sell-off (Kritzman's example of a variable that only mattered after COVID)
DATES = ["2008-09-30", "2008-11-30", "2020-02-29", "2020-03-31", "2021-12-31", "2022-03-31"]


def auc(group):
    return roc_auc_score(group["Event_A"], group["Probability"]) if group["Event_A"].nunique() == 2 else np.nan


def summary(group):
    """Forecast against outcome for one group of forecasts."""
    return pd.Series({
        "Forecasts": len(group),
        "Mean fit": group["Fit"].mean(),
        "Mean probability": group["Probability"].mean(),
        "Loss frequency": group["Event_A"].mean(),
        "Loss months": int(group["Event_A"].sum()),
        "Avg 3-month return": group["Return_3m"].mean(),
        "AUC": auc(group),
    })


if __name__ == "__main__":
    f = pd.read_csv(FORECASTS, parse_dates=["Date"]).set_index("Date")
    imp = pd.read_csv(IMPORTANCE, parse_dates=["Date"]).set_index(["Date", "Measure"])

    variants = {
        "RBP, all": f,
        "RBP, top 50% by fit": f[rbp.high_fit(f["Fit"], 0.50)],
        "RBP, top 25% by fit": f[rbp.high_fit(f["Fit"], 0.25)],
    }

    # 1. Efficacy, Kritzman's rows without the trading rule
    efficacy = pd.concat({name: rbp.efficacy(v) for name, v in variants.items()})
    unconditional = pd.DataFrame({name: summary(v) for name, v in variants.items()}).T

    # 2. Scores of each variant against the base rate on the same dates
    # Base rate as in analysis/backtest.py: share of loss months among all months since 1928
    # whose outcome was known at the forecast date
    events = pd.read_csv("data/equity-index/sp500_tr_events.csv", parse_dates=["Date"]).set_index("Date")
    base = events["Event_A_TR"].expanding().mean().shift(3)
    reference = f.assign(Probability=base.reindex(f.index))
    scores = pd.DataFrame({
        name: {
            "Brier": ((v["Probability"] - v["Event_A"]) ** 2).mean(),
            "Brier, base rate": ((reference.loc[v.index, "Probability"] - v["Event_A"]) ** 2).mean(),
            "AUC": auc(v),
        } for name, v in variants.items()
    }).T
    scores["Brier skill"] = 1 - scores["Brier"] / scores["Brier, base rate"]

    # 3. Fit: quartiles over the whole sample (descriptive, not a trading rule), by period,
    #    and calibration of the probability itself
    quartile = pd.qcut(f["Fit"], 4, labels=["Q1 lowest fit", "Q2", "Q3", "Q4 highest fit"])
    by_fit = f.groupby(quartile, observed=True).apply(summary)
    by_period = f.groupby((f.index.year // 5) * 5).apply(summary)
    by_period.index = [f"{y}-{y + 4}" for y in by_period.index]
    bucket = pd.cut(f["Probability"], [-0.01, 0.05, 0.10, 0.25, 1.0],
                    labels=["0-5%", "5-10%", "10-25%", "25% and above"])
    calibration = f.groupby(bucket, observed=True).apply(summary)
    fit_corr = pd.Series({
        "Correlation of fit with probability": f["Fit"].corr(f["Probability"]),
        "Correlation of fit with trailing loss months": f["Fit"].corr(f["Event_A"].rolling(12).sum()),
        "Forecasts whose 3 engine calls differed by more than 0.05": int((f["Spread of repeats"] > 0.05).sum()),
    })

    # 4. Variable importance: average over all forecasts, and on the chosen dates
    iof = imp.xs("Impact on fit", level="Measure")
    iop = imp.xs("Impact on prediction", level="Measure")
    average = pd.DataFrame({"Mean Impact on Fit": iof.mean(), "Mean Impact on Prediction": iop.mean(),
                            "Share of forecasts with positive Impact on Fit": (iof > 0).mean()})
    average = average.sort_values("Mean Impact on Fit", ascending=False)
    dates = [pd.Timestamp(d) for d in DATES if pd.Timestamp(d) in iof.index]
    on_dates = iof.loc[dates].T
    on_dates.columns = [f"{d:%Y-%m}" for d in dates]
    rank_on_dates = on_dates.rank(ascending=False).astype(int)

    notes = pd.DataFrame({"Note": [
        "Target: 1 if the S&P 500 total return over the next three months is -10% or worse, measured at the end of the three months (data/equity-index/sp500_tr_events.csv).",
        f"Forecasts: monthly, {f.index[0]:%Y-%m} to {f.index[-1]:%Y-%m}, {len(f)} forecasts, {int(f['Event_A'].sum())} loss months. Three-month windows overlap, so the loss months belong to fewer episodes.",
        "Engine: CSA rbp-engine 2.4.0, Grid Prediction on default settings. Each forecast is the median of three calls, because the engine occasionally returns an outlier in long runs.",
        "Training: every month from 1930-10 whose outcome was known at the forecast date (three months earlier or more); the window grows by one month each month. Predictors that start later are left to the engine as missing.",
        "Predictors (14), each as known at the forecast date: past 3-month total return, 63-day volatility, industrial production growth, CPI inflation, unemployment change, 10y-3m slope, real M2 growth (macro series as first published, ALFRED), turbulence, absorption-ratio shift, Baa-Aaa spread, implied volatility (VIX, VXO before 1990), CAPE (earnings lagged six months), Michigan sentiment, S&P 500 HHI (CRSP).",
        "Probability: RBP's prediction for a 0/1 outcome, a relevance-weighted share of past loss months, clipped to [0, 1].",
        "Top 50% / 25% by fit: a forecast counts if its fit is at or above that quantile of the previous 60 forecasts' fits, so no later information is used.",
        "Efficacy: a warning is a forecast at or above the threshold. Hit rate: share of warnings followed by a 10% loss. False positives: warnings not followed by one. Losses caught: loss months warned of.",
        "Fit quartiles and calibration buckets use the whole sample to describe the results; they are not a rule that could have been applied at the time.",
    ]})

    with pd.ExcelWriter(OUT) as xl:
        notes.to_excel(xl, sheet_name="Notes", index=False)
        efficacy.to_excel(xl, sheet_name="Efficacy")
        unconditional.to_excel(xl, sheet_name="Efficacy", startrow=len(efficacy) + 3)
        scores.to_excel(xl, sheet_name="Scores")
        by_fit.to_excel(xl, sheet_name="Fit")
        fit_corr.to_frame("Value").to_excel(xl, sheet_name="Fit", startrow=len(by_fit) + 3)
        by_period.to_excel(xl, sheet_name="Fit over time")
        calibration.to_excel(xl, sheet_name="Calibration")
        average.to_excel(xl, sheet_name="Importance")
        on_dates.to_excel(xl, sheet_name="Importance by date")
        rank_on_dates.to_excel(xl, sheet_name="Importance by date", startrow=len(on_dates) + 3)
        f.to_excel(xl, sheet_name="Forecasts")
        imp.to_excel(xl, sheet_name="Importance, every date")

    pd.set_option("display.width", 250)
    fmt = lambda x: f"{x:.3f}"
    for title, table in [("EFFICACY", efficacy), ("ALL FORECASTS IN EACH VARIANT", unconditional),
                         ("SCORES", scores), ("BY FIT QUARTILE", by_fit), ("FIT", fit_corr.to_frame("Value")),
                         ("BY PERIOD", by_period), ("CALIBRATION", calibration), ("IMPORTANCE", average),
                         ("IMPACT ON FIT BY DATE", on_dates), ("RANK BY DATE", rank_on_dates)]:
        print(f"\n{title}\n{table.to_string(float_format=fmt)}")
    print(f"\nSaved {OUT}")
