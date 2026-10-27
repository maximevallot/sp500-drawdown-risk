"""Absorption ratio from the French daily industry portfolios.

Follows Kritzman, Li, Page and Rigobon (2011): at each day, the share of the
variance of industry returns explained by the top principal components, from a
500-day covariance matrix weighted with a 250-day half-life; then the
standardized shift, (15-day mean - 252-day mean) / 252-day standard deviation.

Two universes:
  - 17 industries, complete from 1926, top 3 components;
  - 49 industries, complete from 1971-03-12, top 10 components (closest to the
    paper's 51 MSCI industries and 10 eigenvectors).

Each month is sampled on its second-to-last trading day, so the value is known
before the month ends (Kritzman, 2026-10-13).

Output: data/fragility/absorption_ratio.csv, one row per month-end. The
diagnostics against the 10% loss events are printed, not saved.

Run from anywhere: python3 analysis/absorption_ratio.py
"""

from io import StringIO
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

WINDOW = 500
HALF_LIFE = 250
SHORT_WINDOW = 15
ANNUAL_WINDOW = 252
SPIKE_THRESHOLD = 1.0

UNIVERSES = {
    "17": {
        "path": "data/industry-returns/17_Industry_Portfolios_Daily_raw.csv",
        "n_industries": 17,
        "n_components": 3,
        "start": None,
    },
    "49": {
        "path": "data/industry-returns/49_Industry_Portfolios_Daily_raw.csv",
        "n_industries": 49,
        "n_components": 10,
        "start": "1971-03-12",  # first day every industry has data
    },
}

OUTPUT = ROOT / "data/fragility/absorption_ratio.csv"


# 1. Load only the value-weighted daily returns block
def load_industry_returns(path, n_industries, start=None):
    lines = Path(path).read_text().splitlines()

    first = next(
        i for i, line in enumerate(lines)
        if "Average Value Weighted Returns -- Daily" in line
    )

    header = first + 1
    while not lines[header].strip():
        header += 1

    rows = []
    for line in lines[header + 1:]:
        if re.match(r"^\s*\d{8}\s*,", line):
            rows.append(line)
        elif rows:
            break  # Stop before the equal-weighted block.

    returns = pd.read_csv(
        StringIO("\n".join([lines[header]] + rows)),
        index_col=0,
        na_values=[-99.99, -999],
    )

    returns.index = pd.to_datetime(
        returns.index.astype(str), format="%Y%m%d"
    )
    returns.index.name = "Date"
    returns.columns = returns.columns.str.strip()

    returns = (
        returns.apply(pd.to_numeric, errors="raise")
        .sort_index()
        / 100.0
    )

    if returns.index.has_duplicates:
        raise ValueError("Duplicate industry-return dates.")
    if returns.shape[1] != n_industries:
        raise ValueError(
            f"Expected {n_industries} industries; found {returns.shape[1]}."
        )

    if start is not None:
        returns = returns.loc[start:].copy()
    if returns.empty:
        raise ValueError("No returns available after the start date.")

    # Missing rows are kept: windows containing them stay unavailable.
    return returns


# 2. Absorption ratio for one 500-day window
def absorption_ratio(window_returns, n):
    x = np.asarray(window_returns, dtype=float)

    if x.ndim != 2 or len(x) != WINDOW:
        raise ValueError("Expected a 500-day returns matrix.")
    if not 1 <= n <= x.shape[1]:
        raise ValueError("Invalid number of principal components.")
    if not np.isfinite(x).all():
        return np.nan

    # Oldest observation has age 499; today's observation has age 0.
    ages = np.arange(len(x) - 1, -1, -1)
    weights = 0.5 ** (ages / HALF_LIFE)
    weights /= weights.sum()

    centered = x - weights @ x
    covariance = centered.T @ (weights[:, None] * centered)

    # eigvalsh returns eigenvalues in ascending order.
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0)
    total_variance = eigenvalues.sum()
    if total_variance <= 0:
        return np.nan

    return eigenvalues[-n:].sum() / total_variance


# 3. Daily ratio and continuous standardized shift
def calculate_daily(returns, n_components):
    values = returns.to_numpy()
    ratios = np.full(len(returns), np.nan)

    for end in range(WINDOW - 1, len(returns)):
        window = values[end - WINDOW + 1:end + 1]
        ratios[end] = absorption_ratio(window, n_components)

    ar = pd.Series(ratios, index=returns.index)

    short_mean = ar.rolling(SHORT_WINDOW, min_periods=SHORT_WINDOW).mean()
    annual_mean = ar.rolling(ANNUAL_WINDOW, min_periods=ANNUAL_WINDOW).mean()
    annual_std = ar.rolling(ANNUAL_WINDOW, min_periods=ANNUAL_WINDOW).std(ddof=1)

    shift = (short_mean - annual_mean) / annual_std.where(annual_std > 0)

    return pd.DataFrame({"ar": ar, "shift": shift})


# 4. Sample the second-to-last trading day of each month
def sample_monthly(daily):
    rows = []

    for month, group in daily.groupby(daily.index.to_period("M")):
        if len(group) < 2:
            continue

        row = group.iloc[-2].to_dict()
        row["signal_date"] = group.index[-2]
        row["Date"] = month.to_timestamp("M")
        rows.append(row)

    return pd.DataFrame(rows).set_index("Date")


# 5a. The 10% loss events, on total return since 2026-10-14 (Michal)
def load_events():
    """Monthly target from data/equity-index/sp500_tr_events.csv.

    Renamed to Return_3m and Event_A so the diagnostics read the same as
    with the earlier price-only file. Added by Claude (2026-10-15).
    """
    events = pd.read_csv(
        ROOT / "data/equity-index/sp500_tr_events.csv", parse_dates=["Date"]
    ).set_index("Date").sort_index()
    return events.rename(
        columns={"Return_3m_TR": "Return_3m", "Event_A_TR": "Event_A"}
    )[["Return_3m", "Event_A"]]


# 5b. Reproduce the calm-market event dates from the volatility analysis
def get_calm_events(events):
    px = pd.read_csv(ROOT / "data/licensed/sp500_yahoo_raw.csv")
    px["Date"] = pd.to_datetime(px["Date"], format="%m/%d/%Y")

    closes = px.set_index("Date").sort_index()["Close"]
    r = np.log(closes / closes.shift(1)).dropna()

    monthly_vol = (r.rolling(63).std() * np.sqrt(252)).resample("ME").last()

    baseline = events.join(
        monthly_vol.rename("Volatility"), how="inner"
    ).dropna(subset=["Event_A", "Volatility"])

    # Same retrospective threshold as in analysis/volatility.py.
    threshold = baseline.loc[baseline["Event_A"].eq(0), "Volatility"].median()

    calm_events = baseline.loc[
        baseline["Event_A"].eq(1) & baseline["Volatility"].lt(threshold)
    ]
    return threshold, calm_events


# 6. Diagnostics against the 10% loss events (printed only)
def report(name, monthly, events, calm_events):
    shift = f"ar{name}_shift"
    data = monthly[[shift]].join(events, how="inner").dropna(
        subset=[shift, "Event_A"]
    )

    event = data["Event_A"].eq(1)
    spike = data[shift].ge(SPIKE_THRESHOLD)

    hits = int((event & spike).sum())
    false_alarms = int((~event & spike).sum())

    print(f"\n=== {name} industries: {data.index.min():%Y-%m} "
          f"to {data.index.max():%Y-%m}, {len(data)} months ===")

    print("\nStandardized shift by Event_A:")
    print(
        data.groupby("Event_A")[shift]
        .agg(["count", "mean", "median", "std"])
        .to_string(float_format=lambda x: f"{x:.3f}")
    )

    print(f"\nSpike diagnostics (shift >= {SPIKE_THRESHOLD:g}):")
    print(f"Event months: {int(event.sum())}")
    print(f"Hits: {hits}  Misses: {int(event.sum()) - hits}  "
          f"False alarms: {false_alarms}")
    print(f"Hit rate: {hits / event.sum():.1%}")
    print(f"False positive rate: {false_alarms / (~event).sum():.1%}")
    if spike.sum():
        print(f"Share of spikes that were false alarms: "
              f"{false_alarms / spike.sum():.1%}")

    calm = calm_events[["Volatility"]].join(monthly[[shift]], how="left")
    available = calm.dropna(subset=[shift])

    print(f"\nCalm-market event dates: {len(calm)}, "
          f"with a signal: {len(available)}, "
          f"with a spike: {int(available[shift].ge(SPIKE_THRESHOLD).sum())}")
    print(available.to_string(float_format=lambda x: f"{x:.4f}"))


# 7. Run and save
def main():
    events = load_events()

    threshold, calm_events = get_calm_events(events)
    print(f"Calm-market volatility threshold: {threshold:.2%}")

    signal_dates = None
    columns = []

    for name, config in UNIVERSES.items():
        returns = load_industry_returns(
            ROOT / config["path"], config["n_industries"], config["start"]
        )
        print(f"\n{name} industries: {len(returns):,} days, "
              f"{returns.index.min():%Y-%m-%d} to "
              f"{returns.index.max():%Y-%m-%d}, "
              f"{int(returns.isna().any(axis=1).sum())} days with missing data")

        monthly = sample_monthly(
            calculate_daily(returns, config["n_components"])
        )

        dates = monthly.pop("signal_date")
        signal_dates = dates if signal_dates is None else signal_dates.combine_first(dates)

        monthly.columns = [f"ar{name}", f"ar{name}_shift"]
        columns.append(monthly)

        report(name, monthly, events, calm_events)

    output = pd.concat(columns, axis=1).sort_index()
    output.insert(0, "signal_date", signal_dates.reindex(output.index))
    output = output.dropna(how="all", subset=output.columns[1:])
    output.index.name = "Date"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, float_format="%.6f", date_format="%Y-%m-%d")
    print(f"\nSaved {OUTPUT.relative_to(ROOT)}: {len(output)} month-ends")


if __name__ == "__main__":
    main()
