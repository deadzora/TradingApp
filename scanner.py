# scanner.py
# Builds/refreshes data/universe_dynamic.json by scanning a base universe
# for liquid, moving symbols (momentum + volume filters).
#
# Requires:
#   pip install alpaca-trade-api pyyaml pandas numpy
#
# Env:
#   APCA_API_KEY_ID, APCA_API_SECRET_KEY  (required)
#   APCA_API_BASE_URL (optional; used for trading endpoints, SDK still handles data)

import os
import sys
import json
import time
import math
import traceback
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv(override=True)

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

from alpaca_trade_api import REST

os.environ["APCA_API_DATA_URL"] = "https://data.alpaca.markets"  # host only
os.environ.pop("POLYGON_API_KEY", None)  # avoid polygon fallback

# ===================== CONFIG =====================

SLEEP_SECONDS = 300                 # how often to refresh the dynamic universe
DATA_DIR = Path("data")
DYN_PATH = DATA_DIR / "universe_dynamic.json"

TIMEFRAME = "1Day"                 # scan timeframe
LOOKBACK_BARS = 90                  # ~ 1 trading day of 15-min bars (6.5h*4=26; 2-4 days buffer)
MIN_ROWS_REQUIRED = 30              # require enough bars to compute features

TOP_N = 30                          # how many symbols to output
MIN_DOLLAR_VOL = 2_000_000          # min avg $ volume over lookback window
MIN_CLOSE_PRICE = 3.0               # skip penny-ish stocks
MAX_SPREAD_PCT = 0.02               # optional: skip if (high-low)/close too wild on last bar

# If config.yaml exists, we'll try to read a static universe from: universe.static
DEFAULT_BASE_UNIVERSE = [
    # Megacaps / ETFs
    "AAPL","MSFT","NVDA","META","AMZN","GOOGL","TSLA","AVGO","NFLX","SMCI",
    "SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","XLY","XLP",
    # High-liquidity tech/semis + movers
    "AMD","INTC","MU","TSM","ARM","PLTR","CRWD","PANW","MRVL","ADBE",
    "SHOP","UBER","ABNB","SNOW","NET","DDOG","MDB","CELH","ENPH","SOFI",
    # Others
    "BA","CAT","F","GM","NKE","PFE","DIS","COIN","BKNG","LULU"
]

# Scoring weights (tweak to taste)
W_MOMENTUM = 0.55   # recent % change & slope
W_VOLUME   = 0.30   # dollar volume (liquidity)
W_RANGE    = 0.15   # intraday range normalization (prefer controlled ranges)

# ==================================================


def now_utc_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")


def load_base_universe():
    # Try config.yaml: universe.static
    cfg_file = Path("config.yaml")
    if yaml and cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            uni = (cfg.get("universe") or {}).get("static")
            if isinstance(uni, list) and len(uni) > 0:
                print(f"[scanner] loaded {len(uni)} static symbols from config.yaml", flush=True)
                return list(dict.fromkeys([s.strip().upper() for s in uni if isinstance(s, str)]))
        except Exception as e:
            print(f"[scanner] warning: failed reading config.yaml universe: {e}", flush=True)
    # Fallback
    return DEFAULT_BASE_UNIVERSE


def get_env():
    key = os.getenv("APCA_API_KEY_ID")
    sec = os.getenv("APCA_API_SECRET_KEY")
    base = os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets/v2"
    if not key or not sec:
        print("[scanner] Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in env.", flush=True)
        sys.exit(1)
    print(f"[scanner] using key: {key[:4]}******  secret: {sec[:4]}******", flush=True)
    return key, sec, base


def pct_change(a, b):
    try:
        if b == 0 or np.isnan(a) or np.isnan(b):
            return np.nan
        return (a - b) / b
    except Exception:
        return np.nan


def compute_features(df: pd.DataFrame):
    """
    Expects a df with columns: open high low close volume
    Index is timestamps (tz-aware). Returns dict of features.
    """
    if df is None or len(df) < MIN_ROWS_REQUIRED:
        return None

    # basic sanity
    closes = df["close"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    vols = df["volume"].astype(float)
    opens = df["open"].astype(float)

    last_close = float(closes.iloc[-1])
    last_high = float(highs.iloc[-1])
    last_low = float(lows.iloc[-1])

    # Liquidity: dollar volume averaged
    dollar_vol = (closes * vols).rolling(32).mean().iloc[-1]
    if pd.isna(dollar_vol):
        dollar_vol = float((closes * vols).mean())

    # Momentum: recent % change  (last vs median of prior window)
    window = min(32, len(closes)-1)
    baseline = float(closes.iloc[-(window+1):-1].median()) if window > 4 else float(closes.iloc[0])
    mom_pct = pct_change(last_close, baseline)

    # Simple slope: OLS-like (normalized)
    x = np.arange(len(closes[-window:]))
    y = closes[-window:].values
    if len(x) >= 5:
        slope = float(np.polyfit(x, y, 1)[0]) / (np.mean(y) + 1e-8)
    else:
        slope = 0.0

    # Range control: prefer not-too-wild (normalize last bar range)
    last_range = (last_high - last_low) / max(last_close, 1e-6)
    range_score = 1.0 - np.clip(last_range / 0.02, 0.0, 1.0)  # 0–1; 1 is tighter

    # Volume score: log scale to compress extremes
    vol_score = np.clip((math.log10(max(dollar_vol, 1.0)) - 5.5) / 1.5, 0.0, 1.0)  # ~5.5→$3.2M as 0
    # Momentum score: combine mom & slope
    mom_score = np.clip(0.5 * (mom_pct * 10) + 0.5 * (slope * 500), -1.0, 1.0)  # heuristic scaling
    mom_score = np.clip((mom_score + 1.0) / 2.0, 0.0, 1.0)  # map to 0..1

    score = W_MOMENTUM * mom_score + W_VOLUME * vol_score + W_RANGE * range_score

    return {
        "last_close": last_close,
        "dollar_vol": float(dollar_vol),
        "mom_pct": float(mom_pct) if mom_pct is not None else np.nan,
        "slope_n": float(slope),
        "range_last": float(last_range),
        "score": float(score),
    }


def fetch_bars(api: REST, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
    try:
        bars = api.get_bars(symbol, timeframe, limit=limit)
        if hasattr(bars, "df"):
            df = bars.df
        else:
            # newer SDKs return a dataframe directly or a list; handle both
            df = pd.DataFrame(bars).set_index("timestamp") if bars else None
        if df is None or df.empty:
            return None
        # standardize columns just in case
        cols = {c.lower(): c for c in df.columns}
        needed = ["open", "high", "low", "close", "volume"]
        for n in needed:
            if n not in [k.lower() for k in df.columns]:
                # some SDK versions title-case; try to map
                if n in cols:
                    pass
        # ensure tz-aware index
        if df.index.tz is None:
            df.index = df.index.tz_localize(timezone.utc)
        return df
    except Exception as e:
        print(f"[scanner] {symbol} fetch error: {e}", flush=True)
        return None


def main_loop():
    key, sec, base = get_env()
    api = REST(key, sec, base_url=base, api_version='v2')  # auth matches your diag

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    base_universe = load_base_universe()
    print(f"[scanner] base universe size: {len(base_universe)}", flush=True)

    while True:
        start_ts = now_utc_iso()
        print(f"[{start_ts}] scanner: starting pass...", flush=True)

        rows = []
        for sym in base_universe:
            df = fetch_bars(api, sym, TIMEFRAME, LOOKBACK_BARS)
            if df is None or len(df) < MIN_ROWS_REQUIRED:
                continue

            # basic filters
            last_close = float(df["close"].iloc[-1])
            if last_close < MIN_CLOSE_PRICE:
                continue

            feats = compute_features(df)
            if not feats:
                continue

            # extra guardrails
            if feats["dollar_vol"] < MIN_DOLLAR_VOL:
                continue
            if feats["range_last"] > MAX_SPREAD_PCT:
                continue

            rows.append({"symbol": sym, **feats})

        if not rows:
            print("[scanner] no candidates this pass (data/filters too strict).", flush=True)
            # still write an empty-but-valid file so the loop sees updated_at
            payload = {"updated_at": now_utc_iso(), "symbols": []}
            with open(DYN_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"[scanner] wrote {DYN_PATH} (0 symbols). sleeping {SLEEP_SECONDS}s", flush=True)
            time.sleep(SLEEP_SECONDS)
            continue

        df = pd.DataFrame(rows).sort_values("score", ascending=False)
        selected = df.head(TOP_N)["symbol"].tolist()

        payload = {
            "updated_at": now_utc_iso(),
            "symbols": selected,
            "tickers": selected,
        }
        with open(DYN_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[scanner] selected {len(selected)} symbols: {selected[:10]}{'...' if len(selected)>10 else ''}", flush=True)
        print(f"[scanner] wrote {DYN_PATH}. sleeping {SLEEP_SECONDS}s", flush=True)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[scanner] stopped by user.", flush=True)
    except Exception:
        print("[scanner] fatal error:\n" + traceback.format_exc(), flush=True)
        sys.exit(1)
