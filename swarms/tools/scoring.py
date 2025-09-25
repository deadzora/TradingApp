# swarms/tools/scoring.py
import numpy as np
import pandas as pd

def zscore(s: pd.Series): return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def score_attention(reddit_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Reddit mentions into attention metrics."""
    if reddit_df.empty:
        return pd.DataFrame(columns=["ticker","attn_score","attn_reason"])
    g = (reddit_df
         .assign(weight=lambda d: 1 + 0.5*np.log1p(d["score"].clip(lower=0)) + 0.25*np.log1p(d["num_comments"].clip(lower=0)))
         .groupby("ticker")
         .agg(mentions=("ticker","count"), weight_sum=("weight","sum")))
    g["attn_score"] = 50 + 10*zscore(g["weight_sum"]) + 5*zscore(g["mentions"])
    g["attn_reason"] = "reddit: m={} wsum={}".format(g["mentions"], g["weight_sum"])
    g = g.reset_index()[["ticker","attn_score","attn_reason"]]
    return g

def score_news(news_df: pd.DataFrame) -> pd.DataFrame:
    """Simple: more fresh articles -> higher score (you can add LLM sentiment later)."""
    if news_df.empty: return pd.DataFrame(columns=["ticker","news_score","news_reason"])
    # explode tickers if present
    if "symbols" in news_df.columns:
        s = (news_df.explode("symbols")
             .rename(columns={"symbols":"ticker"})
             .groupby("ticker").size().rename("n").reset_index())
    elif "symbols.0" in news_df.columns:
        cols = [c for c in news_df.columns if c.startswith("symbols.")]
        s = news_df[cols].stack().reset_index(drop=True).rename("ticker").to_frame()
        s["n"] = 1
        s = s.groupby("ticker")["n"].sum().reset_index()
    else:
        return pd.DataFrame(columns=["ticker","news_score","news_reason"])
    s["news_score"] = 50 + 10*zscore(s["n"])
    s["news_reason"] = "news_count={}".format(s["n"])
    return s[["ticker","news_score","news_reason"]]

def score_liquidity(bars: pd.DataFrame) -> pd.DataFrame:
    """Expect bars with columns: ticker, close, volume; compute $vol 20d."""
    if bars.empty: return pd.DataFrame(columns=["ticker","liq_score","liq_reason"])
    g = (bars.assign(dollar=lambda d: d["close"]*d["volume"])
              .groupby("ticker").dollar.rolling(20).mean().groupby(level=0).last().reset_index(name="adv20"))
    g["liq_score"] = 40 + 15*zscore(np.log1p(g["adv20"]))
    g["liq_reason"] = "adv20=${:,.0f}".format(g["adv20"])
    return g[["ticker","liq_score","liq_reason"]]

def fuse_scores(attn=None, news=None, liq=None, weights=None) -> pd.DataFrame:
    weights = weights or {"attn":0.4,"news":0.3,"liq":0.3}
    df = None
    for piece, key in [(attn,"attn"), (news,"news"), (liq,"liq")]:
        if piece is None or piece.empty: continue
        df = piece if df is None else pd.merge(df, piece, on="ticker", how="outer")
    if df is None: return pd.DataFrame(columns=["ticker","score","why"])
    for k in ["attn","news","liq"]:
        if f"{k}_score" not in df.columns: df[f"{k}_score"] = 50.0
        if f"{k}_reason" not in df.columns: df[f"{k}_reason"] = ""
    df["score"] = (weights["attn"]*df["attn_score"] + weights["news"]*df["news_score"] + weights["liq"]*df["liq_score"])
    df["why"] = df.apply(lambda r: f"attn:{r.attn_score:.0f} {r.attn_reason} | news:{r.news_score:.0f} {r.news_reason} | liq:{r.liq_score:.0f} {r.liq_reason}", axis=1)
    return df.sort_values("score", ascending=False).reset_index(drop=True)
