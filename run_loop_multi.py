import time, numpy as np, pandas as pd
from pathlib import Path
from bot.providers.registry import make_brokers, make_data_feeds
from bot.features import compute_indicators
from bot.scoring import shortlist
from bot.signals import mean_reversion_signal, momentum_signal
from bot.sizing import atr_position_size
from bot.bandit import ContextualBandit
import yaml, datetime as dt, json
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------- helpers ----------
def now_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def load_config(p='config.yaml'):
    with open(p, 'r') as f:
        return yaml.safe_load(f)

def ensure_dirs(cfg):
    for k in ["data_dir", "logs_dir", "model_dir"]:
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)

def fetch_with_routing(cfg, feeds, symbol):
    """Try 15m first, then 1d fallback to avoid 'no data' nights/weekends."""
    tries = [cfg["routing"]["default"]["data"]] + [
        n for n in cfg["providers"]["data"] if n != cfg["routing"]["default"]["data"]
    ]
    for name in tries:
        feed = feeds.get(name)
        if not feed:
            continue
        # 15m first
        try:
            d = feed.fetch_history(symbol, period="60d", interval="15m")
            if d is not None and len(d) > 0:
                return d, f"{name} (15m)"
        except Exception:
            pass
        # 1d fallback
        try:
            d = feed.fetch_history(symbol, period="6mo", interval="1d")
            if d is not None and len(d) > 0:
                return d, f"{name} (1d)"
        except Exception:
            pass
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
    ts = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    with out.open("a", encoding="utf-8") as f:
        f.write(f"{ts},{equity:.2f},{savings:.2f}\n")

def load_dynamic_universe(path="data/universe_dynamic.json", max_symbols=50):
    p = Path(path)
    if not p.exists():
        return []
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        symbols = [c.get("symbol") for c in obj.get("candidates", []) if c.get("symbol")]
        return symbols[:max_symbols]
    except Exception:
        return []

# ---------- main ----------
def main():
    print("[boot] loading config...", flush=True)
    cfg = load_config()
    ensure_dirs(cfg)
    data_dir = cfg["paths"]["data_dir"]
    equity_scale = float(cfg.get("equity_scale", 1.0))  # e.g., 0.001 to make 100k -> 100

    print("[boot] making brokers/data feeds...", flush=True)
    brokers = make_brokers(cfg)
    feeds = make_data_feeds(cfg)
    model = ContextualBandit(
        cfg["paths"]["model_dir"],
        cfg["bandit"]["epsilon_start"],
        cfg["bandit"]["epsilon_floor"],
    )

    # base + dynamic universe (initial)
    base = cfg["universe"]["equities"] + cfg["universe"].get("crypto", [])
    dyn = load_dynamic_universe("data/universe_dynamic.json",
                                max_symbols=cfg.get("discovery", {}).get("max_feed", 50))
    seen = set()
    symbols = []
    for s in base + dyn:
        if s not in seen:
            seen.add(s)
            symbols.append(s)

    broker = brokers.get("alpaca")

    print(f"[boot] universe(base={len(base)}, dyn={len(dyn)}) → {len(symbols)} symbols", flush=True)
    print(f"[boot] loop_sleep_seconds={cfg['loop_sleep_seconds']}", flush=True)
    print(f"[boot] equity_scale={equity_scale}", flush=True)

    while True:
        try:
            t0 = dt.datetime.utcnow()
            print(f"[{now_iso()}] loop start — fetching data...", flush=True)

            # hot-reload dynamic symbols each pass
            dyn = load_dynamic_universe("data/universe_dynamic.json",
                                        max_symbols=cfg.get("discovery", {}).get("max_feed", 50))
            seen = set()
            symbols = []
            for s in base + dyn:
                if s not in seen:
                    seen.add(s)
                    symbols.append(s)
            print(f"[info] symbols this pass: {len(symbols)} (dyn={len(dyn)})", flush=True)

            rows, feat_map, data_used = [], {}, {}
            for sym in symbols:
                df, used = fetch_with_routing(cfg, feeds, sym)
                if df is None:
                    # print(f"  - {sym}: no data", flush=True)
                    continue

                # smaller requirement for 15m bars than for 1d bars
                min_rows = 120 if "(15m)" in str(used) else 220
                if len(df) < min_rows:
                    # print(f"  - {sym}: {used}, rows={len(df)} < {min_rows} (skip)", flush=True)
                    continue

                latest = build_feature_row(df)
                latest.index = [sym]
                rows.append(latest)
                feat_map[sym] = df
                data_used[sym] = used
                # print(f"  + {sym}: {used}, rows={len(df)}", flush=True)

            # If nothing passed, still log scaled equity and sleep
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

            # portfolio selection
            top_k = cfg.get("portfolio", {}).get("top_k", 3)
            max_positions = cfg.get("portfolio", {}).get("max_positions", 6)
            weighting = cfg.get("portfolio", {}).get("weighting", "equal")
            selected = sl.head(top_k)["symbol"].tolist()
            print(f"[info] selected: {selected}", flush=True)

            # weights
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
                    break
                if not broker or broker.has_open_position(sym):
                    continue

                df = feat_map[sym]
                rule_action = decide_action(df)
                if rule_action == "HOLD":
                    continue

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
                if final_action == "HOLD":
                    continue

                # scaled equity for sizing
                try:
                    raw_eq = float(broker.get_equity()) if broker else 0.0
                except Exception:
                    raw_eq = 0.0
                equity = raw_eq * equity_scale

                price = float(f["close"])
                atr_pct = (float(f.get("atr_14", 0)) / price) if price else 0.0
                per_trade_risk = (
                    cfg["risk"]["per_trade_risk_fraction"]
                    * weights.get(sym, 1.0 / max(1, len(selected)))
                )
                qty, stop_frac = atr_position_size(
                    equity,
                    per_trade_risk,
                    price,
                    atr_pct,
                    cfg["stops_targets"]["stop_fraction"],
                )
                if qty <= 0:
                    continue

                stop_price = price * (1 - stop_frac)
                take_price = price * (1 + cfg["stops_targets"]["take_fraction"])

                try:
                    broker.place_bracket_buy(sym, qty, stop_price, take_price)
                    opened += 1
                    print(
                        f"[trade] BUY {sym} x{qty} @~{price:.2f} | "
                        f"stop {stop_price:.2f} take {take_price:.2f} "
                        f"(w={weights.get(sym):.2f}; equity_scaled={equity:.2f})",
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
                            "weight": float(weights.get(sym, 0.0)),
                            "data_provider": data_used.get(sym, "?"),
                        },
                    )
                except Exception as e:
                    print(f"[error] order {sym} failed: {e}", flush=True)

            # equity snapshot (scaled) each loop
            try:
                raw_eq = float(broker.get_equity()) if broker else 0.0
            except Exception:
                raw_eq = 0.0
            append_equity_row(data_dir, raw_eq * equity_scale, savings=0.0)

            # decay exploration
            model.decay_epsilon(cfg["bandit"]["epsilon_decay_per_day"])

            dt_ms = int((dt.datetime.utcnow() - t0).total_seconds() * 1000)
            print(
                f"[{now_iso()}] loop end — took {dt_ms} ms. sleeping {cfg['loop_sleep_seconds']}s",
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
