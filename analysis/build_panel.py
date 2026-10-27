import numpy as np
import pandas as pd

# Builds the one monthly dataset every model reads: data/panel/monthly.csv.
# Written by Claude Code at Maxime's request (2026-10-13).
#
# One row per calendar month-end. Two kinds of column:
#   - the target, Return_3m and Event_A: what happens over the NEXT three months.
#     Never use them as predictors; backtest.py keeps them apart (TARGET_COLUMNS).
#   - predictors: each value is what was known at that month-end. Macro series must be
#     lagged by their release delay before they go in. A predictor is blank before it starts.
#
# The target is the S&P 500 total return (Kritzman, 2026-10-13), Michal's series built from
# Yahoo's ^GSPC and ^SP500TR and Shiller's dividends.
# To add a predictor: load it, align it to month-ends, lag it if needed, and join it below.
#
# Run from the repository root: python3 analysis/build_panel.py

HORIZON = 3
T = HORIZON / 12


def target():
    """Return_3m and Event_A on total return, from Michal's event series (analysis/volatility_michal.ipynb)."""
    events = pd.read_csv(
        "data/equity-index/sp500_tr_events.csv", parse_dates=["Date"]
    ).set_index("Date")
    events = events.rename(columns={"Return_3m_TR": "Return_3m", "Event_A_TR": "Event_A"})
    return events[["Return_3m", "Event_A"]]


def volatility():
    """Olivia's volatility series and their probabilities of a 10% loss (analysis/volatility.py)."""
    vol = pd.read_csv(
        "data/volatility/vol_probabilities.csv", parse_dates=["Date"]
    ).set_index("Date")
    return vol


def garch():
    """Michal's probabilities of a 10% loss on total return (analysis/volatility_michal.ipynb):
    EWMA volatility from 1929 and GJR-GARCH with skewed-t shocks from 1938.

    Already taken on the month's second-to-last trading day, so they go in without a lag.
    """
    g = pd.read_csv(
        "data/volatility/garch_features_michal.csv", parse_dates=["Date"]
    ).set_index("Date")
    return g[["p_ewma94_tr", "gjr_skewt_p_sim_end"]]


def industrial_production():
    """Samantha's industrial production growth, as known at each month-end (analysis/indpro_pit.py).

    Already point in time, built from the ALFRED vintage valid the day before each month-end,
    so it goes in without a further lag.
    """
    ip = pd.read_csv("data/macro/indpro_pit.csv", parse_dates=["Date"]).set_index("Date")
    return ip[["ip_growth_1m", "ip_growth_3m", "ip_growth_12m"]]


def unemployment():
    """Samantha's unemployment rate, as known at each month-end (analysis/unrate_pit.py), from 1960.

    Already point in time, like industrial production, so it goes in without a further lag.
    """
    un = pd.read_csv("data/macro/unrate_pit.csv", parse_dates=["Date"]).set_index("Date")
    return un[["unrate", "unrate_chg_3m", "unrate_chg_12m"]]


def inflation():
    """Samantha's CPI inflation, as known at each month-end (analysis/cpi_pit.py), from 1949.

    Already point in time, like industrial production, so it goes in without a further lag.
    """
    cpi = pd.read_csv("data/macro/cpi_pit.csv", parse_dates=["Date"]).set_index("Date")
    return cpi[["cpi_infl_12m", "cpi_infl_chg_12m"]]


def money_supply(inflation):
    """Samantha's M2 growth, as known at each month-end (analysis/m2_pit.py), from 1980.

    Already point in time, so it goes in without a further lag. Real growth subtracts the
    12-month CPI inflation known at the same month-end.
    """
    m2 = pd.read_csv("data/macro/m2_pit.csv", parse_dates=["Date"]).set_index("Date")
    m2 = m2[["m2_growth_3m", "m2_growth_12m"]]
    m2["m2_real_growth_12m"] = m2["m2_growth_12m"] - inflation["cpi_infl_12m"]
    return m2


def yield_curve():
    """The slope of the yield curve: 10-year Treasury minus 3-month bill, percentage points, from 1962.

    Daily market rates (Olivia's data/rates), never revised. Each month-end takes the last day
    with both rates before it, the same day-before rule as the macro series.
    """
    rates = [
        pd.read_csv(f"data/rates/{name}_raw.csv", parse_dates=["observation_date"])
        .set_index("observation_date")[name]
        for name in ["DGS10", "DTB3"]
    ]
    daily = pd.concat(rates, axis=1).dropna()  # blank: a holiday
    spread = (daily["DGS10"] - daily["DTB3"]).rename("yc_10y_3m")
    # a day counts for the first month-end at least one day after it
    month_end = (spread.index + pd.Timedelta(days=1)) + pd.offsets.MonthEnd(0)
    return spread.groupby(month_end).last().to_frame()


def credit_spread():
    """Moody's Baa minus Aaa yield, percentage points, monthly from 1919 (data/credit/BAA_AAA_raw.csv).

    Each value is the month's average of daily yields, so the whole of month m is known only on
    its last day: month-end t takes month t - 1's value (Kritzman's day-before rule).
    credit_spread_chg_3m: its change over the three months to that value.
    """
    raw = pd.read_csv("data/credit/BAA_AAA_raw.csv", parse_dates=["observation_date"])
    spread = raw.set_index(raw["observation_date"] + pd.offsets.MonthEnd(0))["BAA_AAA"]
    spread = spread.shift(1, freq="ME")  # month t - 1's average, dated t
    return pd.DataFrame({"credit_spread": spread, "credit_spread_chg_3m": spread.diff(HORIZON)})


def cape():
    """Log CAPE as known at each month-end (analysis/cape_pit.py): last month's price over ten
    years of real earnings lagged six months. Already point in time."""
    c = pd.read_csv("data/valuation/cape_pit.csv", parse_dates=["Date"]).set_index("Date")
    return np.log(c["cape_pit"]).rename("log_cape").to_frame()


def trend(events):
    """The S&P 500 total-return index against its own past, at each month-end close, as for
    logret_past_3m: trend_10m, the index over its 10-month average, minus 1; dd_12m_high, the
    index over its highest month-end in the last 12 months, minus 1 (0 at a new high).

    Return_1m_TR dated m is the return of month m, so the index at m is known at m.
    """
    tr = pd.read_csv("data/equity-index/sp500_tr_events.csv", parse_dates=["Date"]).set_index("Date")
    index = (1 + tr["Return_1m_TR"]).cumprod()
    return pd.DataFrame({
        "trend_10m": index / index.rolling(10).mean() - 1,
        "dd_12m_high": index / index.rolling(12).max() - 1,
    }).reindex(events.index)


def past_return(events):
    """The S&P 500 return over the three months ending at each month-end, known at that date.

    Return_3m dated t - 3 is the return from t - 3 to t, so shifting by 3 rows moves it to t.
    """
    return np.log1p(events["Return_3m"]).shift(HORIZON).rename("logret_past_3m")


def normal_hist(logret_past_3m):
    """Normal model with the historical mean and volatility of past 3-month log returns.

    Expanding, from at least 60 months, so each value uses only returns known at that date.
    """
    from scipy.stats import norm

    mu = logret_past_3m.expanding(min_periods=60).mean()
    sigma = logret_past_3m.expanding(min_periods=60).std()
    return pd.Series(
        norm.cdf((np.log(0.9) - mu) / sigma), index=mu.index, name="p_normal_hist"
    )


def build():
    events = target()
    past = past_return(events)
    cpi = inflation()
    panel = events.join(
        [past, normal_hist(past), volatility(), garch(), industrial_production(), unemployment(), cpi,
         money_supply(cpi), yield_curve(), credit_spread(), cape(), trend(events)],
        how="left",
    )
    panel.index.name = "Date"
    return panel


if __name__ == "__main__":
    panel = build()
    panel.to_csv("data/panel/monthly.csv", float_format="%.6f")
    print(f"{len(panel)} month-ends, {panel.index.min():%Y-%m} to {panel.index.max():%Y-%m}")
    print("First month-end with a value, by column:")
    print(panel.apply(lambda c: c.first_valid_index()).dt.strftime("%Y-%m").to_string())
