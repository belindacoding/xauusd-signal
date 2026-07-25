"""Reconcile the logged real-time previousClose against the backfilled 1m bars.

The signal is computed on a ~15h-stale price: EODHD delivers the intraday feed
in a daily batch (watermark ~06:50 UTC), so the "17:00 ET close" the pipeline
sees is really that morning's ~02:50 ET bar. The live endpoint does not quote
metals, but it does return a previousClose for XAUUSD that is fresher than /eod.
This script answers the two questions that decide whether previousClose can
replace the stale price:

  1. WHICH price is it -- the 17:00 ET session close, the 23:59 UTC close, or
     something else?
  2. WHEN does it update -- is it already the just-finished session by 18:00 ET,
     or a day behind?

Run it a few days after the rows were logged, once the bars have backfilled:
    EODHD_API_TOKEN=xxx python3 calibration/reconcile.py
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

CSV = Path(__file__).with_name("feed_cal.csv")
API = "https://eodhd.com/api/intraday/XAUUSD.FOREX"


def load_bars(token: str, days: int = 30) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    r = requests.get(API, params={"api_token": token, "interval": "1m", "fmt": "json",
                                  "from": int((now - timedelta(days=days)).timestamp()),
                                  "to": int(now.timestamp())},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=90)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.set_index("ts")[["close"]].astype(float).sort_index()


def main() -> None:
    token = os.environ.get("EODHD_API_TOKEN") or sys.exit("set EODHD_API_TOKEN")
    if not CSV.exists():
        sys.exit(f"no calibration rows yet at {CSV}")
    rows = pd.read_csv(CSV)
    m1 = load_bars(token)
    et = m1.tz_convert("America/New_York")
    frontier = m1.index[-1]

    def at(when_et: str):
        """Last close at or before a given ET timestamp, or NaN if the bars do
        not reach it yet -- otherwise a not-yet-backfilled reference point would
        silently collapse onto the frontier price and fake a match."""
        want = pd.Timestamp(when_et, tz="America/New_York")
        if want > frontier.tz_convert("America/New_York"):
            return float("nan")
        w = et[et.index <= want]
        if not len(w) or (want - w.index[-1]) > pd.Timedelta("4h"):
            return float("nan")     # market closed / gap around that point
        return float(w["close"].iloc[-1])

    print(f"backfilled through {frontier:%Y-%m-%d %H:%M} UTC\n")
    print(f"{'run (ET)':<18}{'prev_close':>11}{'best match':>22}{'err':>8}"
          f"{'vs same-day 17:00':>19}{'vs prev-day 17:00':>19}")
    for _, r in rows.iterrows():
        run = pd.Timestamp(r["run_et"])
        d = run.date()
        prev = float(r["prev_close"]) if str(r["prev_close"]) != "NA" else float("nan")
        if pd.isna(prev):
            continue
        # candidate reference points, all in ET
        cands = {
            f"{d} 17:00 ET": at(f"{d} 17:00"),
            f"{d} 19:59 ET (23:59Z)": at(f"{d} 19:59"),
            f"{d - timedelta(days=1)} 17:00 ET": at(f"{d - timedelta(days=1)} 17:00"),
            f"{d - timedelta(days=1)} 19:59 ET": at(f"{d - timedelta(days=1)} 19:59"),
        }
        cands = {k: v for k, v in cands.items() if pd.notna(v)}
        if not cands:
            print(f"{r['run_et']:<18}{prev:>11.2f}   (bars not backfilled yet)")
            continue
        best = min(cands, key=lambda k: abs(cands[k] - prev))
        same, prev_day = cands.get(f"{d} 17:00 ET"), cands.get(f"{d - timedelta(days=1)} 17:00 ET")
        f = lambda x: f"{(prev / x - 1) * 1e4:+.0f}bp" if x else "n/a"
        print(f"{r['run_et']:<18}{prev:>11.2f}{best:>22}"
              f"{(prev / cands[best] - 1) * 1e4:>+7.0f}bp{f(same):>19}{f(prev_day):>19}")

    print("\nread: if 'best match' is consistently the SAME-DAY 17:00 ET close and the "
          "error is a few bp, previousClose is the true session close and the signal "
          "should be computed from it instead of the stale watermark bar.")


if __name__ == "__main__":
    main()
