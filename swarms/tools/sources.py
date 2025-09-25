# swarms/tools/sources.py
import os, re, time, json, math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests

UTC = timezone.utc
DATA = Path(__file__).resolve().parents[2] / "data"
DATA.mkdir(exist_ok=True, parents=True)

# --- tiny cache to avoid hammering APIs during 1-min scheduler ticks ---
def _cache_path(name): return DATA / f"cache_{name}.json"
def _read_cache(name, ttl_s):
    p = _cache_path(name)
    if not p.exists(): return None
    try:
        obj = json.loads(p.read_text())
        if time.time() - obj.get("_t", 0) <= ttl_s: return obj["data"]
    except Exception: pass
    return None
def _write_cache(name, data): _cache_path(name).write_text(json.dumps({"_t": time.time(), "data": data}))

# ---------------- Alpaca News (official) ----------------
# Docs: https://data.alpaca.markets/v1beta1/news
def fetch_alpaca_news(tickers=None, since_minutes=240, limit=100) -> pd.DataFrame:
    tickers = tickers or []
    key = os.getenv("APCA_API_KEY_ID"); sec = os.getenv("APCA_API_SECRET_KEY")
    if not (key and sec): return pd.DataFrame()
    cache_key = f"alp_news_{','.join(sorted(tickers))}_{since_minutes}_{limit}"
    cached = _read_cache(cache_key, ttl_s=60)  # 1 min cache
    if cached is not None: return pd.DataFrame(cached)

    base = "https://data.alpaca.markets/v1beta1/news"
    start = (datetime.now(UTC) - timedelta(minutes=since_minutes)).isoformat().replace("+00:00","Z")
    params = {"limit": limit, "start": start}
    if tickers: params["symbols"] = ",".join(tickers)
    r = requests.get(base, headers={"APCA-API-KEY-ID":key,"APCA-API-SECRET-KEY":sec}, params=params, timeout=15)
    if r.status_code != 200:
        # Some accounts don’t include news access; return empty gracefully.
        return pd.DataFrame()
    items = r.json().get("news") or []
    df = pd.json_normalize(items)
    _write_cache(cache_key, df.to_dict(orient="records"))
    return df

# ---------------- Reddit (official API via PRAW) ----------------
# You need: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT in .env
def fetch_reddit_mentions(subs=("wallstreetbets","stocks","options","pennystocks"),
                          window_h=6, limit_per_sub=200) -> pd.DataFrame:
    try:
        import praw  # pip install praw
    except Exception:
        return pd.DataFrame()

    cid = os.getenv("REDDIT_CLIENT_ID"); csec = os.getenv("REDDIT_CLIENT_SECRET"); ua = os.getenv("REDDIT_USER_AGENT","tradingapp/1.0 by you")
    if not (cid and csec): return pd.DataFrame()

    cache_key = f"reddit_{'-'.join(subs)}_{window_h}"
    cached = _read_cache(cache_key, ttl_s=180)  # 3 min cache
    if cached is not None: return pd.DataFrame(cached)

    reddit = praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua)
    since_ts = time.time() - window_h*3600

    rows, cashtag = [], re.compile(r'\$[A-Za-z]{1,5}\b')
    for s in subs:
        try:
            for post in reddit.subreddit(s).new(limit=limit_per_sub):
                if getattr(post, "created_utc", 0) < since_ts: break
                text = f"{post.title}\n{getattr(post,'selftext','') or ''}"
                for tag in set(m.group(0)[1:].upper() for m in cashtag.finditer(text)):
                    rows.append({
                        "subreddit": s, "ticker": tag, "score": int(post.score or 0),
                        "num_comments": int(post.num_comments or 0),
                        "created_utc": float(post.created_utc or since_ts),
                        "permalink": f"https://reddit.com{post.permalink}",
                        "title": post.title
                    })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    _write_cache(cache_key, df.to_dict(orient="records"))
    return df

# ---------------- SEC EDGAR (official) ----------------
# Docs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
# For alerts, we use the "company submissions" JSON once we map tickers->CIK.
def fetch_sec_recent(cik: str, max_items=20) -> pd.DataFrame:
    if os.getenv("SEC_ENABLED", "0") != "1":
        return pd.DataFrame()  # soft-disable until you have a key/time window
    ua = os.getenv("SEC_USER_AGENT", "TradingApp/1.0 (contact: you@example.com)")
    api_key = os.getenv("SEC_API_KEY")  # optional if not required yet
    headers = {"User-Agent": ua}
    if api_key:
        headers["X-Api-Key"] = api_key
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        filings = data.get("filings", {}).get("recent", {})
        return pd.DataFrame(filings).head(max_items)
    except Exception:
        return pd.DataFrame()

# ---------------- (Optional) Finviz snapshots (ToS-sensitive) ----------------
def fetch_finviz_top(kind="ta_topgainers", view=340) -> pd.DataFrame:
    # Heads-up: site has terms; prefer Elite export. Keep disabled by default.
    if os.getenv("ENABLE_FINVIZ_SCRAPE","0") != "1":
        return pd.DataFrame()
    url = f"https://finviz.com/screener.ashx?s={kind}&v={view}"
    try:
        tables = pd.read_html(url)  # brittle but simple
        # find the wide table (view=340) if present
        df = max(tables, key=lambda t: t.shape[1])
        return df
    except Exception:
        return pd.DataFrame()
