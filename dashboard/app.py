from flask import Flask, render_template, jsonify
import os, json, csv
from pathlib import Path
from collections import deque

DATA_DIR = Path(os.getenv("RB_DATA_DIR", "../data"))
LIMIT = int(os.getenv("RB_DASH_LIMIT", "300"))  # cap items returned

app = Flask(__name__)

def tail_jsonl(path: Path, limit: int):
    out = deque(maxlen=limit)
    if not path.exists():
        return list(out)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return list(out)

def tail_csv_equity(path: Path, limit: int):
    if not path.exists():
        return []
    rows = deque(maxlen=limit)
    with path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    "ts": row.get("ts", ""),
                    "equity": float(row.get("equity", 0) or 0),
                    "savings": float(row.get("savings", 0) or 0),
                })
            except Exception:
                continue
    return list(rows)

def maybe_rotate(path: Path, max_mb: int = 20):
    """If a file exceeds max_mb, rotate once to .1 and truncate current."""
    try:
        if not path.exists():
            return
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_mb:
            backup = path.with_suffix(path.suffix + ".1")
            if backup.exists():
                backup.unlink(missing_ok=True)
            path.replace(backup)
            # touch a new empty file
            path.write_text("", encoding="utf-8")
    except Exception:
        pass

@app.route("/")
def index():
    equity = tail_csv_equity(DATA_DIR / "equity.csv", 300)
    decisions = tail_jsonl(DATA_DIR / "decisions.jsonl", 150)[::-1]  # newest first for initial render
    latest_equity = equity[-1]["equity"] if equity else None
    latest_savings = equity[-1]["savings"] if equity else 0.0
    return render_template("index.html",
                           equity=equity,
                           decisions=decisions,
                           latest_equity=latest_equity,
                           latest_savings=latest_savings)

@app.route("/api/equity")
def api_equity():
    rows = tail_csv_equity(DATA_DIR / "equity.csv", LIMIT)
    latest = rows[-1] if rows else {"equity": 0.0, "savings": 0.0}
    return jsonify({
        "rows": rows,
        "latest": {"equity": latest.get("equity", 0.0), "savings": latest.get("savings", 0.0)}
    })

@app.route("/api/decisions")
def api_decisions():
    maybe_rotate(DATA_DIR / "decisions.jsonl", max_mb=50)
    # newest first
    return jsonify(tail_jsonl(DATA_DIR / "decisions.jsonl", LIMIT)[::-1])

@app.route("/api/health")
def api_health():
    return {"ok": True}

if __name__ == "__main__":
    # No debug reloader (prevents duplicate workers), threaded for responsiveness
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8080")), debug=False, use_reloader=False, threaded=True)
