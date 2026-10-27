import pandas as pd
import numpy as np
from scipy.stats import norm

# Target on S&P 500 TOTAL return (Kritzman, 2026-10-13), built by Michal in
# analysis/volatility_michal.ipynb: one row per month-end, with the 3-month total return
# from that month-end and both events already scored from the daily total return series.
# (Switched from the price-only data/licensed/sp500_yahoo_raw.csv with Claude's help, 2026-10-14.)
events_tr = pd.read_csv(
    'data/equity-index/sp500_tr_events.csv',
    parse_dates = ['Date'],
    index_col = 'Date'
)

# We define Event A as the event where SP500 falls by 10% or more between the start and end
# of a three month period

### VERSION 1: QUARTERLY
# keep only the quarter-end months: the 3-month return from a quarter-end is the next quarter's return
quarterly = events_tr[events_tr.index.month % 3 == 0].copy()
quarterly = quarterly.rename(columns = {
    'Return_3m_TR': 'Return_3m',
    'Event_A_TR': 'Event_A',
    'Event_B_TR': 'Event_B'
})

quarterly = quarterly.dropna(subset = ['Return_3m'])

# Historical probability
historical_prob_A = quarterly['Event_A'].mean()
print(f"Quarter Historical probability A: {historical_prob_A:.2%}")
#print(quarterly.head())
#print(quarterly.tail())
#print(len(quarterly))

# Assume ANNUAL normal dist of 10% mean, 20% sd
# use log returns
mu_annual = 0.1
sigma_annual = 0.2

# convert to quarterly log-return parameters
T = 3/12

# 10% and 20% are the mean and sd of DISCRETE annual returns (prof's comment): convert both
# to the mean and sd of log returns before using the log-return normal
# sigma_log^2 = ln(1 + sigma^2/(1+mu)^2), mu_log = ln(1+mu) - sigma_log^2/2
sigma_annual_log = np.sqrt(np.log(1 + sigma_annual**2/(1 + mu_annual)**2))
mu_annual_log = np.log(1 + mu_annual) - 0.5 * sigma_annual_log**2

mu_q = mu_annual_log*T
sigma_q = sigma_annual_log * np.sqrt(T)

# probability of a 10% decline or more
normal_prob_A = norm.cdf(
    np.log(0.9),
    loc = mu_q,
    scale = sigma_q
)
print(f"Quarter Normal probability A, 10%/20%: {normal_prob_A:.2%}")

#print("Number of declines:", quarterly['Event_A'].sum())
#print("Total quarters:", len(quarterly))

# Normal distribution, historical mu and sigma
# the historical ARITHMETIC mean and sd of discrete quarterly returns (Kritzman, 2026-10-13),
# converted to log terms the same way as the 10%/20% above
def log_params(returns):
    m = returns.mean()
    sd = returns.std()
    sigma_log = np.sqrt(np.log(1 + sd**2/(1 + m)**2))
    mu_log = np.log(1 + m) - 0.5 * sigma_log**2
    return m, sd, mu_log, sigma_log

m_q, sd_q, mu_q_hist, sigma_q_hist = log_params(quarterly['Return_3m'])
print(f"Quarter hist mean {4*m_q:.1%} a year, sd {2*sd_q:.1%} a year (discrete)")

normal_prob_A_hist = norm.cdf(
    np.log(0.9),
    loc = mu_q_hist,
    scale = sigma_q_hist
)
print(f"Quarter Normal probability A, hist: {normal_prob_A_hist:.2%}")

# same since 1976, for the fine print on the slides
quarterly_76 = quarterly[quarterly.index >= '1976-01-01']
m_76, sd_76, mu_76, sigma_76 = log_params(quarterly_76['Return_3m'])
print(f"Since 1976: hist mean {4*m_76:.1%} a year, sd {2*sd_76:.1%} a year (discrete)")
print(f"Since 1976: Quarter Normal probability A, hist: {norm.cdf(np.log(0.9), loc = mu_76, scale = sigma_76):.2%}")
print(f"Since 1976: Quarter Historical probability A: {quarterly_76['Event_A'].mean():.2%} "
      f"({quarterly_76['Event_A'].sum()} of {len(quarterly_76)})")

## Intuition to Claude (consider adding to writeup)
# There’s one thing worth noting for your write-up: the normal model with the historical figures overstates the 10% falls (rolling windows: 9.0% against 6.9% on total return, 1928–2026; since 1976 it understates them, 3.1% against 4.5%).
# That’s because real quarterly returns pile up near the middle more than a normal curve does. Big falls are rarer than a normal curve with the
# same volatility predicts, but when they happen they’re more extreme (1929, 1987, 2008).

# Event B defined as prob of dropping 10% or more any time in the quarter
# (Event_B_TR: any daily close in the next three months 10% or more below the start)
historical_prob_B = quarterly['Event_B'].mean()

print(f"Quarter Historical probability B: {historical_prob_B:.2%}")

# formula code from Claude
b = np.log(0.9)
normal_prob_B = (
    # same as event A
    norm.cdf((b-mu_q)/sigma_q)
    + np.exp(2*mu_q * b /sigma_q**2) * norm.cdf((b+mu_q)/sigma_q)
    # adds the path that went down 10% but recovered by quarter end
)
print(f"Quarter Normal probability B, 10%/20%: {normal_prob_B:.2%}")

# formula code from Claude
normal_prob_B_hist = (
    # same as event A
    norm.cdf((b-mu_q_hist)/sigma_q_hist)
    + np.exp(2 * mu_q_hist * b /sigma_q_hist **2) * norm.cdf((b+mu_q_hist)/sigma_q_hist)
)
print(f"Quarter Normal probability B, hist: {normal_prob_B_hist:.2%}")


# VERSION 2: MONTHLY

# EVENT A: end of 3-month
# 2a: historical frequency
# every month-end, 3-month total return to three month-ends later
monthly = events_tr.rename(columns = {
    'Return_3m_TR': 'Return_3m',
    'Event_A_TR': 'Event_A',
    'Event_B_TR': 'Event_B'
})

monthly = monthly.dropna(subset = ['Return_3m'])

# Historical probability
historical_prob_A_m = monthly['Event_A'].mean()
print(f"Rolling Monthly Historical probability A: {historical_prob_A_m:.2%}")
print(f"Rolling Monthly windows that happened, per 50 years (600): {600*historical_prob_A_m:.0f} ({monthly['Event_A'].sum()} of {len(monthly)})")


# 2b: normal 10% mean, 20% std
# remains the same regardless of whether its rolling month / quarterly
print(f"Rolling Monthly Normal probability A, 10%/20%: {normal_prob_A:.2%}")

# 2c: normal, hist
# Normal distribution, historical mu and sigma
# historical discrete mean and sd, converted to log terms
m_m, sd_m, mu_m_hist, sigma_m_hist = log_params(monthly['Return_3m'])

normal_prob_A_hist_m = norm.cdf(
    np.log(0.9),
    loc = mu_m_hist,
    scale = sigma_m_hist
)
print(f"Rolling Monthly Normal probability A, hist: {normal_prob_A_hist_m:.2%}")
print(f"Rolling Monthly hist mean {4*m_m:.1%} a year, sd {2*sd_m:.1%} a year (discrete)")
print(f"Rolling Monthly windows expected in 50 years (600): {600*normal_prob_A_hist_m:.0f}")
print(f"Rolling Monthly windows expected over the whole sample ({len(monthly)}): {len(monthly)*normal_prob_A_hist_m:.0f}")

# same since 1976, for the fine print on the slides (the slides use these rolling windows)
monthly_76 = monthly[monthly.index >= '1976-01-01']
m_m76, sd_m76, mu_m76, sigma_m76 = log_params(monthly_76['Return_3m'])
print(f"Since 1976: Rolling Monthly hist mean {4*m_m76:.1%} a year, sd {2*sd_m76:.1%} a year (discrete)")
print(f"Since 1976: Rolling Monthly Normal probability A, hist: {norm.cdf(np.log(0.9), loc = mu_m76, scale = sigma_m76):.2%}")
print(f"Since 1976: Rolling Monthly Historical probability A: {monthly_76['Event_A'].mean():.2%} "
      f"({monthly_76['Event_A'].sum()} of {len(monthly_76)})")

# same over the 1991-2025 back-test window (420 month-ends), to match slide 5's 21 losses
# (added with Claude's help, 2026-10-21)
monthly_test = monthly[(monthly.index >= '1991-01-01') & (monthly.index <= '2025-12-31')]
m_mt, sd_mt, mu_mt, sigma_mt = log_params(monthly_test['Return_3m'])
normal_prob_A_test = norm.cdf(np.log(0.9), loc = mu_mt, scale = sigma_mt)
print(f"1991-2025: Rolling Monthly windows expected by the full-sample normal ({len(monthly_test)}): "
      f"{len(monthly_test)*normal_prob_A_hist_m:.0f}")
print(f"1991-2025: Rolling Monthly hist mean {4*m_mt:.1%} a year, sd {2*sd_mt:.1%} a year (discrete)")
print(f"1991-2025: Rolling Monthly Normal probability A, own hist: {normal_prob_A_test:.2%}, "
      f"windows expected: {len(monthly_test)*normal_prob_A_test:.0f}")
print(f"1991-2025: Rolling Monthly Historical probability A: {monthly_test['Event_A'].mean():.2%} "
      f"({monthly_test['Event_A'].sum()} of {len(monthly_test)})")

# EVENT B: if it dips anytime (daily close) within that rolling 3 month
# a historical prob
historical_prob_B_m = monthly['Event_B'].mean()

print(f"Rolling Monthly Historical probability B: {historical_prob_B_m:.2%}")

# once again, the same
print(f"Rolling Monthly Normal probability B, 10%/20%: {normal_prob_B:.2%}")

# formula code from Claude
normal_prob_B_hist_m = (
    # same as event A
    norm.cdf((b-mu_m_hist)/sigma_m_hist)
    + np.exp(2 * mu_m_hist * b /sigma_m_hist **2) * norm.cdf((b+mu_m_hist)/sigma_m_hist)
)
print(f"Rolling Monthly Normal probability B, hist: {normal_prob_B_hist_m:.2%}")

# The price-only target, data/equity-index/sp500_events.csv, was written by the
# 2026-10-12 version of this script and is left as it is: the total return target
# is data/equity-index/sp500_tr_events.csv.
