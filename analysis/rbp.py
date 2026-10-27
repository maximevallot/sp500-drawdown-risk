import os
import sys

import numpy as np
import pandas as pd
import rbp_engine as rbp

from backtest import load_panel, predictors, run_backtest, score, base_rate

# Relevance-based prediction with the CSA engine's Grid Prediction, on default settings
# (README.md), as a fit_predict for the shared back-test (analysis/backtest.py).
# Written by Claude Code at Kan's request (2026-10-18).
#
# At each month-end t: X = the past forecast dates whose outcome is known by t, by predictor;
# y = their outcome (1 if the S&P 500 lost 10% or more over the next three months);
# theta = the predictors at t. The prediction is a relevance-weighted average of past
# outcomes, so it reads as a probability; it is clipped to [0, 1].
#
# Kritzman's three variants (2026-10-13): every forecast, the 50% with the highest fit, the
# 25% with the highest fit. A forecast counts as high-fit if its fit is at or above that
# quantile of the fits of the last FIT_WINDOW forecasts before it, so the cut-off uses no
# future data. A rolling window rather than every past forecast, because fits are highest in
# the early years, when the training set is small.
#
# Predictors: all eleven, trained on every month from HISTORY_START, gaps included
# (rbp_grid_full_history). Real M2 starts only in 1980, so keeping only complete rows would
# lose every loss before 1980; with complete rows only, RBP did no better than the base rate
# (2026-10-18).
#
# Efficacy (Kritzman, 2026-10-13): for each variant and each warning threshold, the forecasts
# whose probability is at or above the threshold, their average 3-month total return, the share
# of them followed by a 10% loss (hit rate), and those not followed by one (false positives).
#
# Writes data/rbp/rbp13_forecasts.csv (probability, fit and outcome per date) and
# data/rbp/rbp13_importance.csv (Impact on Fit and Impact on Prediction per date and predictor),
# named after the number of predictors so that earlier runs (rbp11_*, Kan's rbp_forecasts.csv)
# stay as they were. The rbp13 files are the current RBP that every analysis should use
# (Flynn, 2026-10-20; README.md, "Which RBP to use").
#
# Run from the repository root with the licensed engine: .venv/bin/python analysis/rbp.py
#
# Credit spread, implied volatility, CAPE, sentiment, concentration, total-return momentum and volatility,
# and the efficacy table added by Claude Code at Flynn's request (2026-10-18).

# Kritzman's predictors (2026-10-09 and 2026-10-13): Pir's seven, with past return and trailing
# volatility on total return; Olivia's fragility signals; the credit spread; implied volatility,
# valuation and sentiment (Flynn, 2026-10-18)
# Implied volatility (from 1986) and real M2 growth (from 1980) left out, because their short
# histories cut the training of every grid cell that uses them (Flynn, 2026-10-18).
# Thirteen since 2026-10-20 (Flynn): the eleven plus the concentration of the S&P 500 (hhi, from
# 1957, so forecasts end in 2025-12 with CRSP) and GJR-skew-t GARCH's forecast volatility over the
# next three months (gjr_skewt_vol_q, from 1938), kept alongside trailing volatility.
PREDICTORS = [
    "logret_past_3m_tr", "vol_trailing63_tr",
    "ip_growth_12m", "cpi_infl_12m", "unrate_chg_12m", "yc_long",
    "turb17", "ar17_shift", "credit_baa_aaa",
    "cape", "sentiment_umich",
    "hhi", "gjr_skewt_vol_q",
]
NAME = f"rbp{len(PREDICTORS)}"  # output files: rbp11_* for the eleven, rbp13_* for the thirteen
HISTORY_START = "1930-10-31"   # turbulence, the last of the long-history predictors, starts
# First forecast: slide 11's window for the logistic regression, or another month-end given as
# the first argument (python analysis/rbp.py 1970-01-31), whose outputs get it as a suffix
FIRST_FORECAST = sys.argv[1] if len(sys.argv) > 1 else "1991-01-31"
SUFFIX = "" if len(sys.argv) <= 1 else f"_from{FIRST_FORECAST[:4]}"
FIT_WINDOW = 60                # past forecasts the fit cut-off is taken from
MIN_PAST_FITS = 24             # forecasts needed before the fit cut-off is used
THRESHOLDS = [0.10, 0.15, 0.20]  # warning thresholds for the efficacy table (Flynn, 2026-10-18)
REPEATS = 3  # engine calls per forecast, keeping the median (Flynn, 2026-10-18)

RECORDS = []  # what each call of the engine returned, beyond the probability


def rbp_grid(X_train, y_train, X_now):
    """Grid Prediction on default settings; keeps the fit and the impacts of each forecast.

    In long back-tests the engine now and then returns a forecast far from the one it gives
    for the same inputs on its own (for example 45% instead of 4%), two to four months a run,
    though its seed is fixed; its documentation says it computes in parallel on large training
    sets. So each forecast is computed REPEATS times and the call with the median prediction
    is kept, its fit and impacts with it: a one-off outlier is outvoted by the other two."""
    if y_train.nunique() < 2:
        return float(y_train.mean())
    outs = [
        rbp.predict_grid(y_train.to_numpy(float), X_train.to_numpy(float), X_now.to_numpy(float))
        for _ in range(REPEATS)
    ]
    yhats = [float(o.yhat[0]) for o in outs]
    out = outs[int(np.argsort(yhats)[len(yhats) // 2])]
    insights = out.grid_insights
    RECORDS.append({
        "Date": X_now.index[0],
        "Fit": float(out.fit[0]),
        "Adjusted fit": float(out.adjusted_fit[0]),
        "Raw": float(out.yhat[0]),
        "Spread of repeats": max(yhats) - min(yhats),
        "Impact on fit": dict(zip(X_train.columns, np.asarray(insights.impact_on_fit, float))),
        "Impact on prediction": dict(zip(X_train.columns, np.asarray(insights.impact_on_prediction, float))),
    })
    return float(np.clip(out.yhat[0], 0, 1))


def rbp_grid_full_history(X_all):
    """Grid Prediction on every past month, gaps included: the engine gives a month zero
    relevance in the grid cells that use a predictor it lacks (CSA technical overview,
    missing data), so predictors that start late do not cut the history of the others.
    The back-test must be run with an X that is never missing, such as X_all's first column,
    so that it drops no training month."""
    def fit_predict(X_train, y_train, X_now):
        return rbp_grid(X_all.loc[y_train.index], y_train, X_all.loc[X_now.index])
    return fit_predict


def total_return_target(panel):
    """The panel with Michal's total-return target in place of the price-only one."""
    tr = pd.read_csv("data/equity-index/sp500_tr_events.csv", parse_dates=["Date"]).set_index("Date")
    panel = panel.copy()
    panel["Return_3m"] = tr["Return_3m_TR"].reindex(panel.index)
    panel["Event_A"] = tr["Event_A_TR"].reindex(panel.index)
    return panel.dropna(subset=["Event_A"])


def fragility(panel):
    """Olivia's absorption ratio shift and turbulence (data/fragility), joined by month-end."""
    ar = pd.read_csv("data/fragility/absorption_ratio.csv", parse_dates=["Date"]).set_index("Date")
    turb = pd.read_csv("data/fragility/turbulence.csv", parse_dates=["Date"]).set_index("Date")
    return panel.join(ar[["ar17_shift"]]).join(turb[["turb17"]])


def total_return_inputs(panel):
    """Past 3-month return and trailing volatility on total return, in place of the panel's
    price-only ones. Return_3m dated t - 3 is the return from t - 3 to t, known at t; Michal's
    volatility is taken on the month's second-to-last trading day."""
    garch = pd.read_csv("data/volatility/garch_features_michal.csv", parse_dates=["Date"]).set_index("Date")
    panel = panel.copy()
    panel["logret_past_3m_tr"] = np.log1p(panel["Return_3m"]).shift(3)
    panel["vol_trailing63_tr"] = garch["vol_trailing63_tr"].reindex(panel.index)
    # GJR-skew-t forecast of the next 63 days' volatility, annualized, from the same file and day
    panel["gjr_skewt_vol_q"] = garch["gjr_skewt_vol_q"].reindex(panel.index)
    return panel


def credit_spread(panel):
    """Moody's Baa minus Aaa yield, percentage points (data/credit/BAA_AAA_raw.csv, from 1919;
    never revised). FRED's monthly value is the month's average, known only once the month has
    ended, so month m's average is used at the end of month m + 1."""
    raw = pd.read_csv("data/credit/BAA_AAA_raw.csv", parse_dates=["observation_date"])
    spread = raw.set_index("observation_date")["BAA_AAA"]
    spread.index = spread.index + pd.offsets.MonthEnd(2)
    return panel.join(spread.rename("credit_baa_aaa"))


def second_to_last_trading_day(daily):
    """A daily series' value on each month's second-to-last trading day, by month-end."""
    daily = daily.dropna()
    monthly = daily.groupby(daily.index.to_period("M")).apply(lambda x: x.iloc[-2] if len(x) >= 2 else np.nan)
    monthly.index = monthly.index.to_timestamp("M")
    return monthly


def implied_volatility(panel):
    """VIX divided by 100 on the month's second-to-last trading day (Olivia's vol_vix, from
    1990-01), with the VXO, its predecessor on S&P 100 options, taken the same way for 1986-1989."""
    raw = pd.read_csv("data/volatility/VXOCLS_raw.csv", parse_dates=["observation_date"])
    vxo = pd.to_numeric(raw.set_index("observation_date")["VXOCLS"], errors="coerce")
    vxo = second_to_last_trading_day(vxo).loc[:"1989-12-31"] / 100
    panel = panel.copy()
    panel["implied_vol"] = panel["vol_vix"].combine_first(vxo.reindex(panel.index))
    return panel


def shiller():
    """Robert Shiller's monthly price, earnings, CPI and long rate (data/valuation/ie_data_raw.xls)."""
    raw = pd.read_excel("data/valuation/ie_data_raw.xls", sheet_name="Data", header=None, skiprows=8)
    raw = raw.iloc[:, [0, 1, 3, 4, 6]].set_axis(["date", "P", "E", "CPI", "GS10"], axis=1)
    raw = raw[pd.to_numeric(raw["date"], errors="coerce").notna()].astype(float)
    year = raw["date"].astype(int)
    month = ((raw["date"] - year) * 100).round().astype(int)  # 1871.1 is October
    raw.index = pd.to_datetime(dict(year=year, month=month, day=1)) + pd.offsets.MonthEnd(0)
    return raw[["P", "E", "CPI", "GS10"]]


def cape(panel):
    """Shiller's cyclically adjusted P/E, rebuilt with what was known at each month-end t:
    the real price of month t - 1 (Shiller's P is the month's average of daily closes, so it is
    complete only once the month ends) over the average real earnings of the 120 months ending
    at t - 6. S&P's quarterly earnings come out up to about five months after the months they
    cover, so Shiller's published CAPE, which uses earnings up to month t, was not known at t.
    CPI is not seasonally adjusted, so never revised; the CPI of t - 1 is out by mid-month t."""
    s = shiller()
    real_price = s["P"] / s["CPI"]
    real_earnings = (s["E"] / s["CPI"]).rolling(120, min_periods=120).mean()
    value = real_price.shift(1) / real_earnings.shift(6)
    return panel.join(value.rename("cape"))


def yield_curve_long(panel):
    """The slope of the yield curve from 1934: Shiller's long rate (FRED's 10-year GS10 from 1953,
    a long government bond yield before) minus the 3-month bill (FRED TB3MS), percentage points.
    Both are averages of the month's daily rates, known only once the month ends, so month m's
    slope is used at the end of month m + 1. Market rates, never revised. Replaces the panel's
    yc_10y_3m (daily, from 1962) so the whole history is measured the same way (Flynn, 2026-10-18)."""
    raw = pd.read_csv("data/rates/TB3MS_raw.csv", parse_dates=["observation_date"])
    bill = raw.set_index("observation_date")["TB3MS"]
    bill.index = bill.index + pd.offsets.MonthEnd(0)
    slope = (shiller()["GS10"] - bill).dropna()
    slope.index = slope.index + pd.offsets.MonthEnd(1)
    return panel.join(slope.rename("yc_long"))


def sentiment(panel):
    """University of Michigan consumer sentiment (data/sentiment/UMCSENT_raw.csv). The final
    figure for month m comes out at the end of month m, so it is used at the end of month m + 1.
    Quarterly before 1978: the last figure published is carried for the two months in between."""
    raw = pd.read_csv("data/sentiment/UMCSENT_raw.csv", parse_dates=["observation_date"])
    value = pd.to_numeric(raw.set_index("observation_date")["UMCSENT"], errors="coerce")
    value.index = value.index + pd.offsets.MonthEnd(2)
    return panel.join(value.ffill(limit=2).rename("sentiment_umich"))


def concentration(panel):
    """HHI of the S&P 500 members' market caps at the close of each month's last trading day
    (data/concentration/sp500_hhi.csv, built by analysis/hhi.py from CRSP), used for the forecast
    dated that month-end (Flynn, 2026-10-18). Left out, with a notice, until the file exists."""
    path = "data/concentration/sp500_hhi.csv"
    if not os.path.exists(path):
        print(f"{path} is missing, so RBP runs without hhi: get the CRSP extract "
              "(analysis/pull_crsp_sp500.py), then run analysis/hhi.py.\n")
        return panel
    hhi = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return panel.join(hhi["hhi"])


def efficacy(results):
    """Kritzman's efficacy rows for one set of forecasts, at each warning threshold."""
    rows = {}
    for threshold in THRESHOLDS:
        warned = results[results["Probability"] >= threshold]
        losses = int(warned["Event_A"].sum())
        rows[f"p >= {threshold:.0%}"] = {
            "Forecasts": len(results),
            "Warnings": len(warned),
            "Avg return, warned": warned["Return_3m"].mean(),
            "Avg return, not warned": results.loc[results["Probability"] < threshold, "Return_3m"].mean(),
            "Hit rate": losses / len(warned) if len(warned) else np.nan,
            "False positives": len(warned) - losses,
            "Losses caught": f"{losses} of {int(results['Event_A'].sum())}",
        }
    return pd.DataFrame(rows).T


def high_fit(fits, share):
    """True where a forecast's fit is in the top `share` of the fits of the last FIT_WINDOW forecasts."""
    cutoff = fits.rolling(FIT_WINDOW, min_periods=MIN_PAST_FITS).quantile(1 - share).shift(1)
    return fits >= cutoff


def run(name, fit_predict, X, panel, last_forecast=None):
    """Back-test one RBP set-up; returns its forecasts with the fit, its three variants'
    scores, and what the engine returned for each forecast."""
    RECORDS.clear()
    results = run_backtest(fit_predict, X, panel, FIRST_FORECAST, last_forecast)
    reference = run_backtest(base_rate, X, panel, FIRST_FORECAST, last_forecast)
    info = pd.DataFrame(RECORDS).set_index("Date")
    results = results.join(info[["Fit", "Adjusted fit", "Raw", "Spread of repeats"]]).join(panel["Return_3m"])

    variants = {f"{name}, all": results}
    for share in (0.50, 0.25):
        variants[f"{name}, top {share:.0%} by fit"] = results[high_fit(results["Fit"], share)]
    table = pd.DataFrame({label: score(r, reference) for label, r in variants.items()}).T
    efficacy_table = pd.concat({label: efficacy(r) for label, r in variants.items()})
    return results, table, efficacy_table, info


if __name__ == "__main__":
    panel = total_return_inputs(total_return_target(load_panel()))
    panel = concentration(sentiment(cape(yield_curve_long(credit_spread(fragility(panel))))))
    X_all = predictors(panel, [c for c in PREDICTORS if c in panel]).loc[HISTORY_START:]
    # Forecasts end at the last month-end with every predictor (HHI's CRSP data can end earlier)
    last_forecast = X_all.dropna().index[-1]
    missing = X_all.loc[FIRST_FORECAST:last_forecast].isna().sum()
    assert not missing.any(), f"predictors missing on forecast dates: {missing[missing > 0].to_dict()}"
    # The back-test runs on a column that is never missing; the predictors come from X_all
    results, table, efficacy_table, info = run(
        "RBP", rbp_grid_full_history(X_all), X_all[["turb17"]], panel, last_forecast
    )

    pd.set_option("display.width", 250)
    fmt = lambda x: f"{x:.3f}"
    print(f"Forecasts {results.index[0]:%Y-%m} to {results.index[-1]:%Y-%m}; "
          f"average 3-month total return over all of them: {results['Return_3m'].mean():.2%}")
    print("\nEfficacy (Kritzman): forecasts at or above each threshold")
    print(efficacy_table.to_string(float_format=fmt))
    print("\nScores (analysis/backtest.py; its hit rate is the share of loss months warned of)")
    print(table.to_string(float_format=fmt))
    print(f"\nRaw predictions below 0, clipped: {int((results['Raw'] < 0).sum())} of {len(results)}")
    outliers = results[results["Spread of repeats"] > 0.05]
    print(f"Forecasts whose {REPEATS} engine calls disagreed by more than 0.05 (median kept): {len(outliers)}"
          + (f", on {', '.join(f'{d:%Y-%m}' for d in outliers.index)}" if len(outliers) else ""))
    print("\nMean Impact on Fit by predictor:")
    iof = pd.DataFrame(info["Impact on fit"].tolist(), index=info.index).mean()
    print(iof.sort_values(ascending=False).to_string(float_format=fmt))

    importance = pd.concat({
        kind: pd.DataFrame(info[kind].tolist(), index=info.index)
        for kind in ("Impact on fit", "Impact on prediction")
    }, names=["Measure"]).reorder_levels([1, 0]).sort_index()
    os.makedirs("data/rbp", exist_ok=True)
    results.to_csv(f"data/rbp/{NAME}_forecasts{SUFFIX}.csv")
    importance.to_csv(f"data/rbp/{NAME}_importance{SUFFIX}.csv")
