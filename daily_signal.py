"""Self-contained daily signal generator for XAUUSD strategies.

Fetches ~3.5y of 1m data from EODHD (12 chunked requests), rebuilds daily bars
on the exact 17:00 ET session boundary, and prints today's signals for:
  1. Grind Momentum (primary): long if past-5d>0 AND out of HVB regime, else flat
  2. GC-Composite v1 (reference): TOM + dip(rate-gated) + big-day continuation

Usage: EODHD_API_TOKEN=xxx python3 src/daily_signal.py [--capital 100000 --lev 1.5]
Run after 17:05 ET for a final session close. No local state required.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

API = "https://eodhd.com/api/intraday/XAUUSD.FOREX"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"


def fetch_1m(token: str, days: int = 1250) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    chunks, cursor = [], start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=110), end)
        for attempt in range(4):
            r = requests.get(API, params={
                "api_token": token, "interval": "1m", "fmt": "json",
                "from": int(cursor.timestamp()), "to": int(chunk_end.timestamp())},
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
                timeout=60)
            if r.status_code == 200:
                rows = r.json()
                if rows:
                    df = pd.DataFrame(rows)
                    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                    chunks.append(df.set_index("ts")[["open", "high", "low", "close"]].astype(float))
                break
            time.sleep(10 * (attempt + 1))
        cursor = chunk_end
        time.sleep(0.3)
    m1 = pd.concat(chunks)
    m1 = m1[~m1.index.duplicated(keep="first")].sort_index()
    m1 = m1[~((m1.index.dayofweek == 5) | ((m1.index.dayofweek == 6) & (m1.index.hour < 22)))]
    med = m1["close"].rolling(7, center=True, min_periods=3).median()
    dev = pd.concat([(m1[c] / med - 1).abs() for c in m1.columns], axis=1).max(axis=1)
    return m1[dev <= 0.015]


def daily_bars(m1: pd.DataFrame) -> pd.DataFrame:
    et = m1.tz_convert("America/New_York")
    et = et[et.index.dayofweek < 5]
    sh = et.copy()
    sh.index = sh.index + pd.Timedelta(hours=7)   # 17:00 ET -> midnight
    d = sh.resample("1D").agg({"open": "first", "high": "max",
                               "low": "min", "close": "last"}).dropna()
    # drop a partially-formed final session (e.g. run after the 18:00 ET reopen:
    # the new session has only a few minutes of bars and must not count as a day)
    counts = sh["close"].resample("1D").count()
    if counts.reindex(d.index).iloc[-1] < 300:
        d = d.iloc[:-1]
    return d


def real_rate_falling() -> bool | None:
    try:
        rr = pd.read_csv(FRED, parse_dates=["observation_date"]).set_index(
            "observation_date")["DFII10"].dropna()
        return bool(rr.diff(60).iloc[-1] < 0)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--lev", type=float, default=1.5)
    ap.add_argument("--force", action="store_true", help="bypass session-completeness gate")
    args = ap.parse_args()
    token = os.environ.get("EODHD_API_TOKEN") or sys.exit("set EODHD_API_TOKEN")

    d = daily_bars(fetch_1m(token))
    last = d.index[-1]
    now_et = pd.Timestamp.now(tz="America/New_York")
    if not args.force and (last.date() != now_et.date() or now_et.hour < 17):
        print("SESSION_INCOMPLETE_SKIP: latest complete session "
              f"{last.date()}, now {now_et:%H:%M} ET — today's session not closed yet")
        return
    ret = d["close"].pct_change() * 1e4

    # frozen Grind spec (series form): all state lagged one session
    vol60 = ret.rolling(60).std()
    vol_pct = vol60.rolling(756, min_periods=252).rank(pct=True).shift(1)
    above = (d["close"] > d["close"].rolling(60).mean()).shift(1)
    hvb = (vol_pct > 0.6) & above.fillna(False)
    mom5s = d["close"].pct_change(5) > 0
    sig = (mom5s & ~hvb).astype(int)
    grind, grind_prev = bool(sig.iloc[-1]), bool(sig.iloc[-2])
    in_hvb = bool(hvb.iloc[-1])
    mom5 = float(d["close"].iloc[-1] / d["close"].iloc[-6] - 1)
    ma60 = d["close"].rolling(60).mean()
    shadow_short = bool(d["close"].iloc[-2] < ma60.iloc[-2]) and (mom5 < 0)

    idx = d.index
    mon = pd.Series(idx.tz_localize(None).to_period("M"), index=idx)
    tom_rank = mon.groupby(mon).cumcount()
    next_is_tom = (tom_rank.iloc[-1] + 1 < 3) or (idx[-1].month != (idx[-1] + pd.tseries.offsets.BDay(1)).month)
    dip_raw = bool(((d["close"] <= d["low"].rolling(20).min().shift())
                    .rolling(20, min_periods=1).max()).iloc[-1])
    rr_falling = real_rate_falling()
    dip = dip_raw and (rr_falling is not False)
    thr = ret.abs().rolling(252, min_periods=100).quantile(0.9)
    big = int(np.sign(ret.iloc[-1])) if abs(ret.iloc[-1]) > thr.iloc[-1] else 0
    v1_pos = int(np.clip(int(next_is_tom) + int(dip) + big, -1, 2))

    px = float(d["close"].iloc[-1])
    lots = round(args.lev * args.capital / (px * 100), 2)

    # execution guidance (F1 study 2026-07-10: entries earlier=better, exits later=better)
    if grind and not grind_prev:
        action = f"BUY {lots} lots 市价，越早越好：18:00 ET 重开即买（拖到次日平均多付 ~18bps）"
    elif grind_prev and not grind:
        action = "SELL 全部持仓：等到 20:00 ET 市价卖出（或挂限价 昨收+10bps 至 20:00，未成交则 20:00 市价）——晚卖平均多赚 ~4bps"
    elif grind:
        action = "无操作（维持多头）"
    else:
        action = "无操作（维持空仓）"

    print(f"===== XAUUSD DAILY SIGNAL | session close {last.date()} 17:00 ET "
          f"(generated {now_et:%Y-%m-%d %H:%M ET}) =====")
    print(f"close={px:.2f}  past-5d={mom5*100:+.2f}%  regime={'HVB(blow-off)' if in_hvb else 'GRIND'}")
    print()
    print(f">>> GRIND MOMENTUM (primary): {'LONG' if grind else 'FLAT'}  (yesterday: {'LONG' if grind_prev else 'FLAT'})")
    print(f">>> ACTION: {action}")
    print()
    print(f">>> GC-COMPOSITE v1 (reference): pos={v1_pos:+d}  "
          f"[TOM={int(next_is_tom)} Dip={int(dip)} Big={big:+d}]")
    print()
    print(f">>> SHADOW SHORT (纸面追踪，勿交易): {'SHORT' if shadow_short else 'inactive'}"
          f"  [below 60dMA={bool(d['close'].iloc[-2] < ma60.iloc[-2])}, 5d-down={mom5 < 0}]")


if __name__ == "__main__":
    main()
