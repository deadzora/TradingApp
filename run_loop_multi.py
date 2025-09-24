# run_loop_multi.py — providers + direct Alpaca(iex) HTTP fallback with explicit start; verbose logs; discovery; lower thresholds
import time
import numpy as np
import pandas as pd
from pathlib import Path
from bot.providers.registry import make_brokers, make_data_feeds
from bot.features import compute_indicators
from bot.scoring import shortlist
from bot.signals import mean_reversion_signal, momentum_signal
from bot.sizing import atr_position_size
from bot.bandit import ContextualBandit
import yaml
from datetime import datetime, timezone, timedelta
import json, sys, os, pathlib

# ----- stdout safety -----
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----- load .env (same as scanner) -----
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

# ----- force canonical data host & avoid polygon fallback -----
os.environ["APCA_API_DATA_URL"] = "https://data.alpaca.markets"  # host only
os.environ.pop("POLYGON_API_KEY", None)

def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def iso_utc(dt_: datetime) -> str:
    if dt_.tzinfo is None:
        dt_ = dt_.replace(tzinfo=timezone.utc)
    return dt_.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def load_config(p='config.yaml'):
    with open(p, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(cfg):
    for k in ["data_dir", "logs_dir", "model_dir"]:
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)

# ---------- Direct Alpaca fallback (HTTP, iex feed, with explicit start) ----------
def fetch_alpaca_http(symbol: str, timeframe: str, lookback_days: int, limit: int, feed: str = "iex", adjustment: str = "raw"):
    """
    Fetch bars directly from Alpaca Data v2 over HTTP with explicit 'start'.
    timeframe: '15Min' or '1Day'
    Returns pandas DataFrame with columns open,high,low,close,volume indexed by UTC timestamp, or None.
    """
    try:
        import requests
        key = os.getenv("APCA_API_KEY_ID")
        sec = os.getenv("APCA_API_SECRET_KEY")
        if not key or not sec:
            print(f"  · {symbol}: alpaca_http missing credentials", flush=True)
            return None

        start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        params = {
            "timeframe": timeframe,
            "start": iso_utc(start),
            "limit": limit,
            "adjustment": adjustment,
            "feed": feed,
        }
        headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec, "Accept": "application/json"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            print(f"  · {symbol}: alpaca_http({timeframe}) {r.status_code} {r.text[:160]}", flush=True)
            return None
        data = r.json()
        bars = data.get("bars") or []
        if not bars:
            print(f"  · {symbol}: alpaca_http({timeframe}) returned 0 bars", flush=True)
            return None
        # Normalize to expected columns
        rows = []
        for b in bars:
            rows.append({
                "timestamp": b.get("t"),
                "open": float(b.get("o", 0) or 0),
                "high": float(b.get("h", 0) or 0),
                "low": float(b.get("l", 0) or 0),
                "close": float(b.get("c", 0) or 0),
                "volume": float(b.get("v", 0) or 0),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            print(f"  · {symbol}: alpaca_http({timeframe}) normalized empty df", flush=True)
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        print(f"  · {symbol}: alpaca_http({timeframe}) error: {e}", flush=True)
        return None

# ---- very verbose fetch helper (providers -> SDK iex -> HTTP iex (with start)) ----
def fetch_with_routing(cfg, feeds, symbol):
    """
    Try configured providers first (15m then 1d).
    If empty, try direct Alpaca SDK with feed='iex'.
    If still empty, try direct Alpaca HTTP with feed='iex' and explicit 'start'.
    """
    route_default = cfg["routing"]["default"]["data"]
    tries = [route_default] + [n for n in cfg["providers"]["data"] if n != route_default]

    # Providers
    for name in tries:
        feed = feeds.get(name)
        if not feed:
            print(f"  · {symbol}: provider '{name}' not found in feeds", flush=True)
            continue
        # 15m first
        try:
            d = feed.fetch_history(symbol, period="60d", interval="15m")
            if d is not None and len(d) > 0:
                return d, f"{name} (15m)"
            else:
                print(f"  · {symbol}: {name} 15m returned empty/None", flush=True)
        except Exception as e:
            print(f"  · {symbol}: {name} 15m error: {e}", flush=True)
        # 1d fallback
        try:
            d = feed.fetch_history(symbol, period="6mo", interval="1d")
            if d is not None and len(d) > 0:
                return d, f"{name} (1d)"
            else:
                print(f"  · {symbol}: {name} 1d returned empty/None", flush=True)
        except Exception as e:
            print(f"  · {symbol}: {name} 1d error: {e}", flush=True)

    # Direct Alpaca SDK (iex) — try with explicit start/end
    try:
        from alpaca_trade_api import REST
        api = REST(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"),
                   base_url=os.getenv("APCA_API_BASE_URL"), api_version='v2')
        start15 = datetime.now(timezone.utc) - timedelta(days=60)
        start1d = datetime.now(timezone.utc) - timedelta(days=180)
        bars = api.get_bars(symbol, "15Min", start=iso_utc(start15), limit=200, feed="iex")
        df = bars.df if hasattr(bars, "df") else None
        if df is not None and len(df) > 0:
            return df, "alpaca_sdk (15m, iex, start)"
        bars = api.get_bars(symbol, "1Day", start=iso_utc(start1d), limit=180, feed="iex")
        df = bars.df if hasattr(bars, "df") else None
        if df is not None and len(df) > 0:
            return df, "alpaca_sdk (1d, iex, start)"
        else:
            print(f"  · {symbol}: alpaca_sdk iex returned empty with start", flush=True)
    except Exception as e:
        print(f"  · {symbol}: alpaca_sdk iex error: {e}", flush=True)

    # Direct HTTP (iex) with explicit start
    df = fetch_alpaca_http(symbol, "15Min", lookback_days=60, limit=200, feed="iex")
    if df is not None and len(df) > 0:
        return df, "alpaca_http (15m, iex, start)"
    df = fetch_alpaca_http(symbol, "1Day", lookback_days=180, limit=180, feed="iex")
    if df is not None and len(df) > 0:
        return df, "alpaca_http (1d, iex, start)"

    return None, None

def build_feature_row(df: pd.DataFrame):
    feat = compute_indicators(df)
    latest = pd.concat([df.tail(1), feat.tail(1)], axis=1)
    latest["avg_dollar_vol"] = (df["close"] * df["volume"]).rolling(20).mean().tail(1).values
    return latest

def decide_action(df: pd.DataFrame):
    mr = mean_reversion_signal(df)
    mom = momentum_signal(df)
    if mr and not mom:
        return "MR"
    if mom and not mr:
        return "MOM"
    if mr and mom:
        return "MOM"
    return "HOLD"

def log_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def append_equity_row(data_dir, equity, savings=0.0):
    out = Path(data_dir) / "equity.csv"
    if not out.exists():
        out.write_text("ts,equity,savings\n", encoding="utf-8")
    ts = now_iso()
    with out.open("a", encoding="utf-8") as f:
        f.write(f"{ts},{equity:.2f},{savings:.2f}\n")

def load_discovery_symbols(data_dir: str, max_feed: int) -> list[str]:
    p = pathlib.Path(data_dir) / "universe_dynamic.json"
    if not p.exists():
        print(f"[disc] {p} not found; using 0 discovery symbols", flush=True)
        return []
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        syms = obj.get("symbols") or obj.get("tickers") or []
        out = []
        for s in syms:
            if isinstance(s, str):
                s = s.strip().upper()
                if s and s not in out:
                    out.append(s)
        if max_feed and max_feed > 0:
            out = out[:max_feed]
        print(f"[disc] loaded {len(out)} discovery symbols from {p}", flush=True)
        return out
    except Exception as e:
        print(f"[disc] read error {p}: {e}", flush=True)
        return []

def main():
    print("[boot] loading config...", flush=True)
    cfg = load_config()
    ensure_dirs(cfg)
    data_dir = cfg["paths"]["data_dir"]
    equity_scale = float(cfg.get("equity_scale", 1.0))

    # endpoints by mode (paper/live)
    mode = str(cfg.get("mode", "paper")).lower()
    base_url = cfg.get("alpaca", {}).get("paper_base_url" if mode == "paper" else "live_base_url")
    if base_url:
        os.environ["APCA_API_BASE_URL"] = base_url
    print(f"[env] mode={mode}  trading_base={os.getenv('APCA_API_BASE_URL')}  data_host={os.getenv('APCA_API_DATA_URL')}", flush=True)

    # configurable row thresholds (defaults tuned for your current data)
    data_cfg = cfg.get("data", {}) or {}
    MIN_ROWS_15M = int(data_cfg.get("min_rows_15m", 60))   # was 120
    MIN_ROWS_1D  = int(data_cfg.get("min_rows_1d", 120))   # was 220

    print("[boot] making brokers/data feeds...", flush=True)
    brokers = make_brokers(cfg)
    feeds = make_data_feeds(cfg)

    # quick SDK sanity (iex) with explicit start. It's OK if 0 here; HTTP fallback may still work.
    try:
        from alpaca_trade_api import REST
        api = REST(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"),
                   base_url=os.getenv("APCA_API_BASE_URL"), api_version='v2')
        start15 = datetime.now(timezone.utc) - timedelta(days=60)
        spy = api.get_bars("SPY", "15Min", start=iso_utc(start15), limit=10, feed="iex").df
        print(f"[sanity] SPY bars via SDK iex (start): {len(spy)}", flush=True)
    except Exception as e:
        print(f"[sanity] SDK iex failed: {e}", flush=True)

    # show feeds
    print(f"[boot] feeds available: {list(feeds.keys())}", flush=True)
    print(f"[boot] default route: {cfg['routing']['default']}", flush=True)

    model = ContextualBandit(
        cfg["paths"]["model_dir"],
        cfg["bandit"]["epsilon_start"],
        cfg["bandit"]["epsilon_floor"],
    )

    base = (cfg["universe"].get("equities") or []) + (cfg["universe"].get("crypto") or [])
    disc_max = int(cfg.get("discovery", {}).get("max_feed", 50))
    dyn = load_discovery_symbols(cfg["paths"]["data_dir"], disc_max)
    symbols = list(dict.fromkeys([*(s.strip().upper() for s in base), *dyn]))

    broker = brokers.get("alpaca")

    print(f"[boot] universe(base={len(base)}, dyn={len(dyn)}) → {len(symbols)} symbols", flush=True)
    print(f"[boot] loop_sleep_seconds={cfg['loop_sleep_seconds']}", flush=True)
    print(f"[boot] equity_scale={equity_scale}", flush=True)

    while True:
        try:
            t0 = datetime.now(timezone.utc)
            print(f"[{now_iso()}] loop start — fetching data...", flush=True)

            dyn = load_discovery_symbols(cfg["paths"]["data_dir"], disc_max)
            symbols = list(dict.fromkeys([*(s.strip().upper() for s in base), *dyn]))
            print(f"[info] symbols this pass: {len(symbols)} (dyn={len(dyn)})", flush=True)

            rows, feat_map, data_used = [], {}, {}
            for sym in symbols:
                df, used = fetch_with_routing(cfg, feeds, sym)
                if df is None:
                    continue

                used_lc = str(used).lower()
                min_rows = MIN_ROWS_15M if "(15m)" in used_lc else MIN_ROWS_1D
                if len(df) < min_rows:
                    print(f"  - {sym}: {used}, rows={len(df)} < {min_rows} (skip)", flush=True)
                    continue

                latest = build_feature_row(df)
                latest.index = [sym]
                rows.append(latest)
                feat_map[sym] = df
                data_used[sym] = used
                print(f"  + {sym}: {used}, rows={len(df)} (ok)", flush=True)

            if not rows:
                try:
                    raw_eq = float(broker.get_equity()) if broker else 0.0
                except Exception:
                    raw_eq = 0.0
                append_equity_row(data_dir, raw_eq * equity_scale, savings=0.0)
                print("[warn] no symbols — wrote equity heartbeat, sleeping...", flush=True)
                time.sleep(cfg["loop_sleep_seconds"])
                continue

            feats = pd.concat(rows)
            sl = shortlist(feats, cfg["scoring"]["weights"], cfg["scoring"]["min_score_to_trade"])
            print(f"[info] shortlist size={len(sl)}", flush=True)

            top_k = cfg.get("portfolio", {}).get("top_k", 3)
            max_positions = cfg.get("portfolio", {}).get("max_positions", 6)
            weighting = cfg.get("portfolio", {}).get("weighting", "equal")
            selected = sl.head(top_k)["symbol"].tolist()
            print(f"[info] selected: {selected}", flush=True)

            if weighting == "risk" and selected:
                w_raw = []
                for s in selected:
                    f = feats.loc[s]
                    price = float(f["close"])
                    atr_pct = (float(f.get("atr_14", 0)) / price) if price else 0.0
                    w_raw.append((s, 1.0 / max(atr_pct, 1e-6)))
                tot = sum(w for _, w in w_raw) or 1.0
                weights = {s: w / tot for s, w in w_raw}
            else:
                weights = {s: 1.0 / max(1, len(selected)) for s in selected}

            opened = 0
            for sym in selected:
                if opened >= max_positions:
                    print(f"[skip] reached max_positions={max_positions}", flush=True)
                    break

                if not broker:
                    print(f"[debug] broker missing; cannot trade {sym}", flush=True)
                    break

                if broker.has_open_position(sym):
                    print(f"[skip] {sym}: already in position", flush=True)
                    continue

                df = feat_map[sym]
                rule_action = decide_action(df)
                print(f"[debug] {sym}: rule_action={rule_action}", flush=True)
                if rule_action == "HOLD":
                    print(f"[skip]  {sym}: rule says HOLD", flush=True)

                # Features for bandit
                f = feats.loc[sym]
                x_vec = np.nan_to_num(
                    np.array(
                        [
                            f.get("ret_5d", 0),
                            f.get("ret_20d", 0),
                            f.get("rsi_14", 50),
                            f.get("atr_14", 0),
                            float(f.get("ma_50", 0) or 0),
                            float(f.get("ma_200", 0) or 0),
                            f.get("vol_z", 0),
                        ]
                    ),
                    nan=0.0,
                )

                chosen, _ = model.select(x_vec)
                final_action = chosen if chosen != "HOLD" else rule_action
                print(f"[debug] {sym}: bandit={chosen} -> final_action={final_action}", flush=True)

                if final_action == "HOLD":
                    print(f"[skip]  {sym}: final_action=HOLD (no trade)", flush=True)
                    continue

                # Equity & sizing
                try:
                    raw_eq = float(broker.get_equity()) if broker else 0.0
                except Exception as e:
                    print(f"[debug] get_equity error: {e}; assuming 0", flush=True)
                    raw_eq = 0.0
                equity = raw_eq * equity_scale

                price = float(f["close"])
                atr_pct = (float(f.get("atr_14", 0)) / price) if price else 0.0
                per_trade_risk = (
                    cfg["risk"]["per_trade_risk_fraction"] * max(1e-9, float(len(selected))) ** -1
                    if cfg.get("portfolio", {}).get("weighting", "equal") != "risk"
                    else cfg["risk"]["per_trade_risk_fraction"]
                )
                qty, stop_frac = atr_position_size(
                    equity,
                    per_trade_risk,
                    price,
                    atr_pct,
                    cfg["stops_targets"]["stop_fraction"],
                )

                min_qty = int(cfg.get("risk", {}).get("min_qty", 0))
                if qty <= 0 and min_qty > 0:
                    print(f"[debug] {sym}: qty computed {qty} -> bumping to min_qty={min_qty}", flush=True)
                    qty = min_qty

                if qty <= 0:
                    print(f"[skip]  {sym}: qty<=0 after sizing (equity_scaled={equity:.2f}, price={price:.2f}, atr_pct={atr_pct:.4f})", flush=True)
                    continue

                stop_price = price * (1 - stop_frac)
                take_price = price * (1 + cfg["stops_targets"]["take_fraction"])
                print(
                    f"[debug] {sym}: placing bracket BUY x{qty} @~{price:.2f} "
                    f"stop={stop_price:.2f} take={take_price:.2f}",
                    flush=True,
                )

                try:
                    broker.place_bracket_buy(sym, qty, stop_price, take_price, entry_price=price)
                    opened += 1
                    print(
                        f"[trade] BUY {sym} x{qty} @~{price:.2f} | "
                        f"stop {stop_price:.2f} take {take_price:.2f} "
                        f"(equity_scaled={equity:.2f})",
                        flush=True,
                    )
                    log_json(
                        Path(cfg["paths"]["data_dir"]) / "decisions.jsonl",
                        {
                            "ts": now_iso(),
                            "symbol": sym,
                            "final_action": final_action,
                            "qty": int(qty),
                            "stop_price": float(stop_price),
                            "take_price": float(take_price),
                            "equity_scaled": float(equity),
                            "data_provider": "alpaca",  # or override if you want to pass 'used'
                        },
                    )
                except Exception as e:
                    print(f"[error] order {sym} failed: {e}", flush=True)

            try:
                raw_eq = float(broker.get_equity()) if broker else 0.0
            except Exception:
                raw_eq = 0.0
            append_equity_row(data_dir, raw_eq * equity_scale, savings=0.0)

            model.decay_epsilon(cfg["bandit"]["epsilon_decay_per_day"])

            dt_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
            print(
                f"[{now_iso()}] loop end — took {dt_ms} ms. sleeping {cfg['loop_sleep_seconds']}",
                flush=True,
            )
            time.sleep(cfg["loop_sleep_seconds"])

        except Exception as e:
            print(f"[fatal] unhandled exception: {e}", flush=True)
            try:
                log_json(
                    Path(cfg["paths"]["logs_dir"]) / "fatal.jsonl",
                    {"ts": now_iso(), "err": str(e)},
                )
            except Exception:
                pass
            time.sleep(cfg["loop_sleep_seconds"])

if __name__ == "__main__":
    main()
