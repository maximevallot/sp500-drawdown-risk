import numpy as np
import pandas as pd

# Industrial production as it was known at each month-end, with no look-ahead.
# Written by Claude Code at Samantha's request (2026-10-15), reviewed with Samantha.
#
# Input: data/macro/INDPRO_raw.csv, every ALFRED vintage in the real-time-period layout
# (one row per observation month and value, valid from realtime_start_date to realtime_end_date).
#
# For each month-end t, the series as published the day before t (Kritzman's timing rule):
# the rows valid on that day. From that one vintage, the latest month published and its growth.
# Growth is computed inside the vintage only, because the index is rebased between vintages.
#
# Output: data/macro/indpro_pit.csv, one row per calendar month-end.
# Run from the repository root: python3 analysis/indpro_pit.py

RAW = "data/macro/INDPRO_raw.csv"
OUT = "data/macro/indpro_pit.csv"


def load_vintages():
    """Every published value, with the dates it was the current figure (end blank while current)."""
    raw = pd.read_csv(
        RAW, parse_dates=["period_start_date", "realtime_start_date", "realtime_end_date"]
    )
    raw = raw.dropna(subset=["INDPRO"])  # blank: that vintage did not include the month
    raw["realtime_end_date"] = raw["realtime_end_date"].fillna(pd.Timestamp.max)
    return raw


def as_known_on(raw, day):
    """The series as published on day: observation month -> value."""
    valid = (raw["realtime_start_date"] <= day) & (raw["realtime_end_date"] >= day)
    return raw.loc[valid].set_index("period_start_date")["INDPRO"].sort_index()


def growth(series, months):
    """Change over the last `months` months of one vintage, as a fraction; NaN if not published."""
    last = series.index[-1]
    earlier = last - pd.DateOffset(months=months)
    if earlier not in series.index:
        return np.nan
    return series.loc[last] / series.loc[earlier] - 1


def point_in_time(raw):
    # from the first vintage to the end of the month of the last one downloaded
    first = raw["realtime_start_date"].min()
    last_vintage = raw["realtime_start_date"].max()
    month_ends = pd.date_range(first, last_vintage + pd.offsets.MonthEnd(0), freq="ME")
    rows = []
    for t in month_ends:
        series = as_known_on(raw, t - pd.Timedelta(days=1))
        if series.empty:
            continue
        last = series.index[-1]
        rows.append({
            "Date": t,
            "ip_obs_month": last.strftime("%Y-%m"),
            # how stale the latest figure is: 1 means last month's figure is out
            "ip_months_old": (t.year - last.year) * 12 + t.month - last.month,
            "ip_growth_1m": growth(series, 1),
            "ip_growth_3m": growth(series, 3),
            "ip_growth_12m": growth(series, 12),
        })
    return pd.DataFrame(rows).set_index("Date")


if __name__ == "__main__":
    pit = point_in_time(load_vintages())
    pit.to_csv(OUT, float_format="%.6f")
    print(f"{OUT}: {len(pit)} month-ends, {pit.index[0].date()} to {pit.index[-1].date()}")
    print("How old the latest figure is, in months:")
    print(pit["ip_months_old"].value_counts().sort_index().to_string())
