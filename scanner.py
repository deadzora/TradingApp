import os, json, time, math, datetime as dt
from pathlib import Path
import pandas as pd
from alpaca_trade_api import REST

DATA_DIR = Path("data")
DYN_PATH = DATA_DIR / "universe_dynamic.json"

# ---- knobs (tweak as you like) ----
SCAN_INTERVAL_SEC = 300            # scan every 5 minutes
ASSET_UNIVERSE_MAX = 800           # how many assets to consider per pass
CANDIDATES_MAX = 100               # how many to publish to file
MIN_PRICE = 1.0                    # include penny stocks but exclude sub-$1 noise by default
MAX_PRICE = 5000.0
MIN_DOLLAR_VOL = 1_000_000         # average(Close*Volume) over 20d
REQUIRE_POS_MOM = True             # only keep positive momentum if True
INCLUDE_ETFS = True                # keep ETFs (SPY/QQQ/etc.)
# -----------------------------------

def _now():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def atomic_write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)

def get_assets(api: REST):
    """Return a filtered list of tradable assets."""
    assets = api.list_assets(status="active")
    out = []
    for a in assets:
        # include only US equities/ETFs that are tradable
        if not getattr(a, "tradable", False):
            continue
        cl = getattr(a, "class_", None)
        if cl not in ("us_equity", "us_etf", "us_equity/etf"):
            continue
        if (not INCLUDE_ETFS) and getattr(a, "symbol", "").endswith((".U", ".W")):
            continue
        out.append(a.symbol)
    # simple cap to avoid rate limits
    return out[:ASSET_UNIVERSE_MAX]

def fetch_daily(api: REST, symbol: str, limit: int = 60):
    bars = api.get_bars(symbol, "1Day", limit=limit).df
    if bars is None or len(bars) == 0:
        return None
    df = bars[["open","high","low","close","volume"]].copy()
    return df

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(high, low, close, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def score_daily(df: pd.DataFrame):
    # indicators
    ma20 = df["close"].rolling(20).mean()
    ma50 = df["close"].rolling(50).mean()
    rsi14 = rsi(df["close"], 14)
    atr14 = atr(df["high"], df["low"], df["close"], 14)
    vol_z = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)
    dollar_vol = (df["close"] * df["volume"]).rolling(20).mean()

    last = df.index[-1]
    row = {
        "price": float(df["close"].iloc[-1]),
        "rsi14": float(rsi14.iloc[-1]) if not math.isnan(rsi14.iloc[-1]) else 50.0,
        "atr_pct": float((atr14.iloc[-1] / df["close"].iloc[-1])) if df["close"].iloc[-1] else 0.0,
        "vol_z": float(vol_z.iloc[-1]) if not math.isnan(vol_z.iloc[-1]) else 0.0,
        "dv_20": float(dollar_vol.iloc[-1]) if not math.isnan(dollar_vol.iloc[-1]) else 0.0,
        "mom_pos": bool(df["close"].iloc[-1] > (ma20.iloc[-1] if not math.isnan(ma20.iloc[-1]) else df["close"].iloc[-1])),
        "nh_20": bool(df["close"].iloc[-1] >= df["close"].rolling(20).max().iloc[-1]),
        "nl_20": bool(df["close"].iloc[-1] <= df["close"].rolling(20).min().iloc[-1]),
    }

    # simple weighted score (0..100)
    score = 0.0
    # momentum (+)
    score += 20.0 if row["mom_pos"] else 0.0
    score += 20.0 if row["nh_20"] else 0.0
    # liquidity (+)
    score += 20.0 * min(1.0, row["dv_20"] / max(MIN_DOLLAR_VOL, 1))
    # unusual volume (+ if vol_z>0)
    score += 10.0 * max(0.0, min(1.0, (row["vol_z"] + 3) / 6))  # maps roughly -3..+3 to 0..1
    # volatility sweet spot (ATR% between ~1% and 8%)
    atrp = row["atr_pct"]
    if 0.01 <= atrp <= 0.08:
        score += 20.0
    elif 0.005 <= atrp <= 0.12:
        score += 10.0
    # RSI comfort zone
    if 40 <= row["rsi14"] <= 70:
        score += 10.0

    return score, row

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = REST(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"),
               base_url="https://paper-api.alpaca.markets")

    while True:
        start = dt.datetime.utcnow()
        print(f"[{_now()}] scanner: starting pass...")
        try:
            symbols = get_assets(api)
            results = []
            for i, sym in enumerate(symbols, 1):
                try:
                    df = fetch_daily(api, sym, limit=60)
                    if df is None or len(df) < 40:
                        continue
                    price = float(df["close"].iloc[-1])
                    if not (MIN_PRICE <= price <= MAX_PRICE):
                        continue
                    s, feats = score_daily(df)
                    if REQUIRE_POS_MOM and not feats["mom_pos"]:
                        continue
                    if feats["dv_20"] < MIN_DOLLAR_VOL:
                        continue
                    results.append({
                        "symbol": sym, "score": round(s, 2), "price": price,
                        "atr_pct": round(feats["atr_pct"], 4),
                        "vol_z": round(feats["vol_z"], 2),
                        "dv_20": round(feats["dv_20"], 2),
                        "nh_20": feats["nh_20"],
                        "ts": _now()
                    })
                except Exception:
                    # skip bad symbols quietly to keep scanner moving
                    continue

                # light throttling to be nice to API
                if i % 50 == 0:
                    time.sleep(0.5)

            # sort & keep top N
            results.sort(key=lambda x: x["score"], reverse=True)
            top = results[:CANDIDATES_MAX]
            payload = {"updated": _now(), "count": len(top), "candidates": top}

            atomic_write_json(DYN_PATH, payload)
            print(f"[{_now()}] scanner: wrote {len(top)} candidates → {DYN_PATH}")

        except Exception as e:
            print(f"[{_now()}] scanner: error: {e}")

        took = int((dt.datetime.utcnow() - start).total_seconds())
        sleep_for = max(0, SCAN_INTERVAL_SEC - took)
        print(f"[{_now()}] scanner: sleeping {sleep_for}s")
        time.sleep(sleep_for)

if __name__ == "__main__":
    main()
