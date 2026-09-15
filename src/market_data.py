"""
Pulls the four target indicators plus the 3-month bill control:

    DXY (US Dollar Index)        Yahoo Finance   DX-Y.NYB
    10s2s spread                 FRED            T10Y2Y
    1-year Treasury yield        FRED            DGS1
    Growth minus Value           Yahoo Finance   IWF (Russell 1000 Growth)
                                                  minus IWN (Russell 2000 Value)
    3-month T-bill (control)     FRED            DGS3MO

Requires: pandas_datareader OR a direct FRED API key, and yfinance.
Run with normal internet access (not this sandboxed container).
"""

import re

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from fredapi import Fred
except ImportError:
    Fred = None


def get_fred_series(series_id: str, start: str, end: str, api_key: str | None = None) -> pd.Series:
    """Fetch a FRED series. Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html

    If you don't want to bother with an API key, you can instead download
    the CSV directly from https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}
    which needs no auth — see get_fred_csv() below.
    """
    if Fred is None or api_key is None:
        return get_fred_csv(series_id, start, end)
    fred = Fred(api_key=api_key)
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s.name = series_id
    return s


def get_fred_csv(series_id: str, start: str, end: str) -> pd.Series:
    """No-auth fallback: FRED's public CSV download endpoint."""
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        f"&cosd={start}&coed={end}"
    )
    df = pd.read_csv(url, parse_dates=["observation_date"], index_col="observation_date")
    s = df[series_id].replace(".", pd.NA).astype(float)
    return s


def get_yahoo_series(ticker: str, start: str, end: str) -> pd.Series:
    if yf is None:
        raise ImportError("pip install yfinance")

    # yfinance's `end` is EXCLUSIVE (returns data strictly before that date),
    # unlike FRED's, which is inclusive. Push it forward one day so a
    # requested end date actually shows up in the result instead of being
    # silently dropped — this matters most for the most recent day(s),
    # which is exactly what the Step 4 forecast needs.
    end_inclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start, end=end_inclusive, progress=False, auto_adjust=True)

    # Newer yfinance versions can return MultiIndex columns (field, ticker)
    # even for a single ticker, depending on version — handle both shapes.
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = df["Close"]

    return close.rename(ticker)


def build_indicator_panel(start: str, end: str, fred_api_key: str | None = None) -> pd.DataFrame:
    """Returns a daily DataFrame with columns:
    dxy, t10y2y, dgs1, growth_value, dgs3mo
    """
    dxy = get_yahoo_series("DX-Y.NYB", start, end)
    iwf = get_yahoo_series("IWF", start, end)
    iwn = get_yahoo_series("IWN", start, end)
    growth_value = (iwf.pct_change() - iwn.pct_change()).rename("growth_value_pct_spread")

    t10y2y = get_fred_series("T10Y2Y", start, end, fred_api_key).rename("t10y2y")
    dgs1 = get_fred_series("DGS1", start, end, fred_api_key).rename("dgs1")
    dgs3mo = get_fred_series("DGS3MO", start, end, fred_api_key).rename("dgs3mo")

    panel = pd.concat([dxy.rename("dxy"), t10y2y, dgs1, growth_value, dgs3mo], axis=1)
    panel = panel.sort_index()
    return panel


def _parse_release_hour(release_time: str | None) -> int | None:
    """Parse a string like '2:00 p.m.' or '10:00 a.m.' into a 24-hour int
    hour. Returns None if unparseable/missing."""
    if not release_time:
        return None
    m = re.search(r"(\d{1,2}):?(\d{2})?\s*([ap])\.?m\.?", release_time, re.IGNORECASE)
    if not m:
        return None
    hour = int(m.group(1))
    is_pm = m.group(3).lower() == "p"
    if is_pm and hour != 12:
        hour += 12
    if not is_pm and hour == 12:
        hour = 0
    return hour


def one_day_change(panel: pd.DataFrame, event_date: str, release_time: str | None = None,
                    after_hours_cutoff_hour: int = 16) -> dict:
    """Given the indicator panel and a release date, compute the change from
    the prior trading day's close to this trading day's close (or next
    available trading day if event_date isn't itself a trading day).

    If release_time is given and parses to an hour at or after
    after_hours_cutoff_hour (default 4pm, i.e. after market close),
    the event is treated as happening on the NEXT trading day instead —
    the market can't react same-day to something it hasn't heard yet.
    This matters most for speeches/testimony, which (unlike statements,
    minutes, and press conferences, all consistently 2:00-2:30pm ET) can
    happen at any hour, including evening dinner keynotes.
    """
    idx = panel.index
    event_ts = pd.Timestamp(event_date)

    hour = _parse_release_hour(release_time)
    if hour is not None and hour >= after_hours_cutoff_hour:
        event_ts = event_ts + pd.Timedelta(days=1)

    # trading day on/after the (possibly shifted) event
    after = idx[idx >= event_ts]
    if len(after) == 0:
        return {}
    t1 = after[0]
    # trading day strictly before t1
    before = idx[idx < t1]
    if len(before) == 0:
        return {}
    t0 = before[-1]

    changes = {}
    for col in ["dxy", "t10y2y", "dgs1", "dgs3mo"]:
        if col in panel.columns:
            changes[f"d_{col}"] = panel.loc[t1, col] - panel.loc[t0, col]
    if "growth_value_pct_spread" in panel.columns:
        changes["d_growth_value"] = panel.loc[t1, "growth_value_pct_spread"]
    return changes
