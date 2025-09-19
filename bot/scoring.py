
import pandas as pd
def score_symbol(row, weights):
    mom = 0
    if row.get("ma_50") and row.get("ma_200"):
        mom += 5 if row["close"] > row["ma_50"] else 0
        mom += 5 if row["close"] > row["ma_200"] else 0
    mom += 5 if (row.get("ret_20d") or 0) > 0 else 0
    rsi = row.get("rsi_14", 50) or 50
    mom += 5 if 40 <= rsi <= 70 else 0
    liq = 10 if (row.get("vol_z") or 0) > 0 else 0
    liq += 10
    theme = 10; sent = 10
    score = (weights["momentum"]*(mom/20.0)) + (weights["liquidity"]*(liq/20.0)) + (weights["theme"]*0.5) + (weights["sentiment"]*0.5)
    return float(score)
def shortlist(latest_feats: pd.DataFrame, weights: dict, min_score: int):
    rows = []
    for sym, r in latest_feats.iterrows():
        s = score_symbol({**r.to_dict(), "close": r["close"]}, weights)
        rows.append((sym, s))
    df = pd.DataFrame(rows, columns=["symbol","score"]).sort_values("score", ascending=False)
    return df[df["score"] >= min_score]
