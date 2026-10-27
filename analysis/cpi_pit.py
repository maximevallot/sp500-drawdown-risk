import numpy as np
import pandas as pd

# CPI inflation as it was known at each month-end, with no look-ahead.
# Written by Claude Code at Samantha's request (2026-10-15), the same way as analysis/indpro_pit.py.
#
# Input: data/macro/CPIAUCNS_raw.csv, every ALFRED vintage of the CPI-U, not seasonally adjusted,
# in the real-time-period layout. ALFRED's vintages start on 1949-03-24, so this series starts in 1949.
#
# For each month-end t, the series as published the day before t (Kritzman's timing rule):
# the rows valid on that day. The unadjusted CPI is not revised, but it was rebased several times
# (1935-39, 1957-59, 1967 and 1982-84 = 100), so inflation is computed inside one vintage only.
# Only 12-month changes: shorter changes in an unadjusted index are mostly seasonal.
#
# Output: data/macro/cpi_pit.csv, one row per calendar month-end.
# Run from the repository root: python3 analysis/cpi_pit.py

RAW = "data/macro/CPIAUCNS_raw.csv"
OUT = "data/macro/cpi_pit.csv"


def load_vintages():
    """Every published value, with the dates it was the current figure (end blank while current)."""
    raw = pd.read_csv(
        RAW, parse_dates=["period_start_date", "realtime_start_date", "realtime_end_date"]
    )
    raw = raw.dropna(subset=["CPIAUCNS"])  # blank: not in that vintage, or never published (October 2025)
    raw["realtime_end_date"] = raw["realtime_end_date"].fillna(pd.Timestamp.max)
    return raw


def as_known_on(raw, day):
    """The series as published on day: observation month -> value."""
    valid = (raw["realtime_start_date"] <= day) & (raw["realtime_end_date"] >= day)
    return raw.loc[valid].set_index("period_start_date")["CPIAUCNS"].sort_index()


def inflation(series, end):
    """12-month change to the month `end` within one vintage, as a fraction; NaN if not published."""
    earlier = end - pd.DateOffset(months=12)
    if end not in series.index or earlier not in series.index:
        return np.nan
    return series.loc[end] / series.loc[earlier] - 1


def point_in_time(raw):
    # from the first vintage to the end of the month of the last one downloaded
    first = raw["realtime_start_date"].min()
    last_vintage = raw["realtime_start_date"].max()
    month_ends = pd.date_range(first, last_vintage + pd.offsets.MonthEnd(0), freq="ME")
    month_ends = month_ends[month_ends <= pd.Timestamp.today()]  # not a month-end still to come
    rows = []
    for t in month_ends:
        series = as_known_on(raw, t - pd.Timedelta(days=1))
        if series.empty:
            continue
        last = series.index[-1]
        infl = inflation(series, last)
        infl_year_ago = inflation(series, last - pd.DateOffset(months=12))
        rows.append({
            "Date": t,
            "cpi_obs_month": last.strftime("%Y-%m"),
            # how stale the latest figure is: 1 means last month's figure is out
            "cpi_months_old": (t.year - last.year) * 12 + t.month - last.month,
            "cpi_infl_12m": infl,
            # change in 12-month inflation from a year earlier, as a fraction
            "cpi_infl_chg_12m": infl - infl_year_ago,
        })
    return pd.DataFrame(rows).set_index("Date")


if __name__ == "__main__":
    pit = point_in_time(load_vintages())
    pit.to_csv(OUT, float_format="%.6f")
    print(f"{OUT}: {len(pit)} month-ends, {pit.index[0].date()} to {pit.index[-1].date()}")
    print("How old the latest figure is, in months:")
    print(pit["cpi_months_old"].value_counts().sort_index().to_string())
