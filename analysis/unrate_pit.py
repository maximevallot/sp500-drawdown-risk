import numpy as np
import pandas as pd

# The unemployment rate as it was known at each month-end, with no look-ahead.
# Written by Claude Code at Samantha's request (2026-10-15), the same way as analysis/indpro_pit.py.
#
# Input: data/macro/UNRATE_raw.csv, every ALFRED vintage in the real-time-period layout
# (one row per observation month and value, valid from realtime_start_date to realtime_end_date).
# ALFRED's vintages start on 1960-03-15, so this series starts in 1960.
#
# For each month-end t, the series as published the day before t (Kritzman's timing rule):
# the rows valid on that day. From that one vintage, the latest month published, its level
# and its change. The rate is already a percentage, so changes are in percentage points,
# computed inside the vintage so that seasonal revisions do not mix in.
#
# Output: data/macro/unrate_pit.csv, one row per calendar month-end.
# Run from the repository root: python3 analysis/unrate_pit.py

RAW = "data/macro/UNRATE_raw.csv"
OUT = "data/macro/unrate_pit.csv"


def load_vintages():
    """Every published value, with the dates it was the current figure (end blank while current)."""
    raw = pd.read_csv(
        RAW, parse_dates=["period_start_date", "realtime_start_date", "realtime_end_date"]
    )
    raw = raw.dropna(subset=["UNRATE"])  # blank: never published (October 2025, the shutdown)
    raw["realtime_end_date"] = raw["realtime_end_date"].fillna(pd.Timestamp.max)
    return raw


def as_known_on(raw, day):
    """The series as published on day: observation month -> value."""
    valid = (raw["realtime_start_date"] <= day) & (raw["realtime_end_date"] >= day)
    return raw.loc[valid].set_index("period_start_date")["UNRATE"].sort_index()


def change(series, months):
    """Change over the last `months` months of one vintage, in percentage points; NaN if not published."""
    last = series.index[-1]
    earlier = last - pd.DateOffset(months=months)
    if earlier not in series.index:
        return np.nan
    return series.loc[last] - series.loc[earlier]


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
        rows.append({
            "Date": t,
            "unrate_obs_month": last.strftime("%Y-%m"),
            # how stale the latest figure is: 1 means last month's figure is out
            "unrate_months_old": (t.year - last.year) * 12 + t.month - last.month,
            "unrate": series.loc[last],
            "unrate_chg_3m": change(series, 3),
            "unrate_chg_12m": change(series, 12),
        })
    return pd.DataFrame(rows).set_index("Date")


if __name__ == "__main__":
    pit = point_in_time(load_vintages())
    pit.to_csv(OUT, float_format="%.6f")
    print(f"{OUT}: {len(pit)} month-ends, {pit.index[0].date()} to {pit.index[-1].date()}")
    print("How old the latest figure is, in months:")
    print(pit["unrate_months_old"].value_counts().sort_index().to_string())
