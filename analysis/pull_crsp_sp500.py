import sys

import pandas as pd
import wrds

# Pulls every S&P 500 constituent's month-end price and shares outstanding from CRSP on WRDS,
# for the concentration (HHI) predictor built by analysis/hhi.py.
# Written by Claude Code at Flynn's request (2026-10-18).
#
# Licensed data: the output goes to data/licensed/, which is gitignored. Never commit it.
# Run it in your own terminal, not through an agent: wrds.Connection() asks for your WRDS
# username and password, and an agent session may be logged.
#
#   python -m pip install wrds
#   python analysis/pull_crsp_sp500.py
#
# CRSP moved to a new format (CIZ) in 2024; the old tables (SIZ) stopped updating at the end of
# 2024. The script tries the new tables first and falls back to the old ones.

OUT = "data/licensed/crsp_sp500_monthly_raw.csv"

# New format: membership spells in dsp500list_v2, monthly prices in msf_v2
CIZ = """
    select m.permno, m.mthcaldt as date, m.mthprc as prc, m.shrout
    from crsp.dsp500list_v2 as s
    join crsp.msf_v2 as m
      on m.permno = s.permno
     and m.mthcaldt between s.mbrstartdt and s.mbrenddt
    order by m.mthcaldt, m.permno
"""

# Old format: membership spells in dsp500list, monthly prices in msf
SIZ = """
    select m.permno, m.date, m.prc, m.shrout
    from crsp.dsp500list as s
    join crsp.msf as m
      on m.permno = s.permno
     and m.date between s.start and s.ending
    order by m.date, m.permno
"""

if __name__ == "__main__":
    db = wrds.Connection()
    try:
        data = db.raw_sql(CIZ, date_cols=["date"])
        source = "CIZ (dsp500list_v2, msf_v2)"
    except Exception as error:
        print(f"New-format tables failed ({error}); trying the old ones.", file=sys.stderr)
        data = db.raw_sql(SIZ, date_cols=["date"])
        source = "SIZ (dsp500list, msf)"
    db.close()

    data.to_csv(OUT, index=False)
    per_month = data.groupby(data["date"].dt.to_period("M")).size()
    print(f"{source}: {len(data):,} rows, {data['date'].min():%Y-%m} to {data['date'].max():%Y-%m}")
    print(f"Stocks per month: median {per_month.median():.0f}, min {per_month.min()}, max {per_month.max()}")
    print(f"Saved {OUT}")
