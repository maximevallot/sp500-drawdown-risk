import os

import numpy as np
import pandas as pd

# Concentration of the S&P 500: the Herfindahl-Hirschman index (HHI) of its members' market caps.
# Written by Claude Code at Flynn's request (2026-10-18).
#
# Input: data/licensed/crsp_sp500_monthly_raw.csv, from analysis/pull_crsp_sp500.py: one row per
# member and month, with CRSP's month-end price (prc) and shares outstanding (shrout, thousands).
# A negative price is CRSP's bid-ask midpoint on a day without trades, so its absolute value is used.
#
# HHI = sum of squared weights, each member's market cap over the members' total: 1/N if every
# stock had the same weight (0.002 for 500 stocks), 1 if one stock were the whole index.
# CRSP's market cap counts every share, while S&P has weighted by float since 2005, so the
# weights are close to the index's, not equal to them.
#
# Timing (Flynn, 2026-10-18): the value at the close of the last trading day of month t is used
# for the forecast dated month-end t, which covers months t + 1 to t + 3. For example, the close
# of 31 December goes into the forecast made for 1 January. The daily market series in the panel
# use the second-to-last trading day instead (Kritzman, 2026-10-13); concentration moves too
# slowly for the one day to matter.
#
# Output: data/concentration/sp500_hhi.csv, one row per calendar month-end, with the HHI, the
# effective number of stocks (1 / HHI), the share of the ten largest, and the number of members.
# Derived from licensed data: only these monthly aggregates are written, never a member's data.
#
# Run from the repository root: python analysis/hhi.py

RAW = "data/licensed/crsp_sp500_monthly_raw.csv"
OUT = "data/concentration/sp500_hhi.csv"
MIN_MEMBERS = 400  # fewer members with a price means a gap in the data, not a smaller index


def concentration(raw):
    """HHI and related measures by calendar month-end."""
    data = raw.assign(mcap=raw["prc"].abs() * raw["shrout"]).dropna(subset=["mcap"])
    data = data[data["mcap"] > 0]
    data["Date"] = data["date"] + pd.offsets.MonthEnd(0)  # CRSP's date is the last trading day

    def measures(month):
        w = month["mcap"] / month["mcap"].sum()
        return pd.Series({
            "hhi": (w ** 2).sum(),
            "effective_n": 1 / (w ** 2).sum(),
            "top10_share": w.nlargest(10).sum(),
            "n_members": len(w),
        })

    out = data.groupby("Date")[["mcap"]].apply(measures)
    out.loc[out["n_members"] < MIN_MEMBERS, ["hhi", "effective_n", "top10_share"]] = np.nan
    return out


if __name__ == "__main__":
    raw = pd.read_csv(RAW, parse_dates=["date"])
    out = concentration(raw)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, float_format="%.6f")
    print(f"{OUT}: {len(out)} month-ends, {out.index[0]:%Y-%m} to {out.index[-1]:%Y-%m}")
    print(f"Months with fewer than {MIN_MEMBERS} members, left blank: {int(out['hhi'].isna().sum())}")
    print(out.iloc[:: max(1, len(out) // 10)].to_string(float_format=lambda x: f"{x:.4f}"))
