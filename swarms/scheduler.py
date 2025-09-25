# swarms/scheduler.py
import os, time, yaml
from datetime import datetime, timedelta
from pathlib import Path

# ---- load .env from repo root ----
ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
except Exception:
    pass

# ---- timezone (ET) ----
try:
    from zoneinfo import ZoneInfo  # py>=3.9; on Windows: pip install tzdata
except Exception:
    raise SystemExit("Please `pip install tzdata` for Windows zoneinfo support")

ET = ZoneInfo("America/New_York")

# ---- helpers ----
def now_et():
    return datetime.now(ET)

def parse_hhmm(s: str):
    h, m = map(int, s.split(":"))
    return h, m

def within_window(now, start_hm, end_hm):
    sh, sm = parse_hhmm(start_hm); eh, em = parse_hhmm(end_hm)
    start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end   = now.replace(hour=eh, minute=em, second=59, microsecond=0)
    return start <= now <= end

def minutes_since_midnight(dt_):
    return dt_.hour * 60 + dt_.minute

def every_trigger(now, spec_every_minutes: int, window=None):
    """
    True at the start minute of each cadence; if window=["HH:MM","HH:MM"], also require within it.
    """
    if window and not within_window(now, window[0], window[1]):
        return False
    return (minutes_since_midnight(now) % spec_every_minutes) == 0

def should_run_fixed(now, hhmm_list):
    mm = f"{now.hour:02d}:{now.minute:02d}"
    return mm in set(hhmm_list)

def market_rth(now):
    # 09:30–16:00 ET
    return within_window(now, "09:30", "16:00")

def load_schedule():
    with open(Path(__file__).with_name("schedule.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def call_agents(only_agents):
    # Use env to filter inside swarms.run_agents
    os.environ["SWARM_ONLY"] = ",".join(only_agents)
    # DRY_RUN toggle stays inside run_agents.py
    import importlib
    mod = importlib.import_module("swarms.run_agents")
    # run once (async inside)
    mod.main_entry_once()

def cooled(last_ts, agent, cooldown_min):
    t = last_ts.get(agent)
    return (t is None) or ((now_et() - t) >= timedelta(minutes=cooldown_min))

def mark_run(last_ts, agent):
    last_ts[agent] = now_et()

def decide_agents_to_run(cfg, last_ts):
    now = now_et()
    run = []

    # DAYTRADE
    d = cfg.get("daytrade", {})
    if d:
        gate = (market_rth(now) if d.get("rth_only", True) else True)
        if gate and should_run_fixed(now, d.get("run_at", [])) and cooled(last_ts, "daytrade", d.get("cooldown_min", 20)):
            run.append("daytrade")

    # OPTIONS
    o = cfg.get("options", {})
    if o:
        gate = (market_rth(now) if o.get("rth_only", True) else True)
        if gate and should_run_fixed(now, o.get("run_at", [])) and cooled(last_ts, "options", o.get("cooldown_min", 60)):
            run.append("options")

    # PENNY
    p = cfg.get("penny", {})
    if p:
        gate = (market_rth(now) if p.get("rth_only", True) else True)
        if gate:
            trig = False
            for win in (p.get("windows") or []):
                ev = win.get("every", "60m")
                if ev.endswith("m"):
                    mins = int(ev[:-1])
                else:
                    continue
                if every_trigger(now, mins, win.get("between")):
                    trig = True; break
            if trig and cooled(last_ts, "penny", p.get("cooldown_min", 30)):
                run.append("penny")

    # CRYPTO (24/7)
    c = cfg.get("crypto", {})
    if c:
        trig = False
        evs = c.get("every")
        if isinstance(evs, str) and evs.endswith("m"):
            mins = int(evs[:-1])
            if every_trigger(now, mins):
                trig = True
        if now.strftime("%H:%M") in set(c.get("extra_at", [])):
            trig = True
        if trig and cooled(last_ts, "crypto", c.get("cooldown_min", 30)):
            run.append("crypto")

    return run

def main():
    print(f"[sched] up | {now_et().isoformat()} ET")
    cfg = load_schedule()
    last_ts = {}
    tick = 60  # seconds
    while True:
        try:
            to_run = decide_agents_to_run(cfg, last_ts)
            if to_run:
                print(f"[sched] {now_et().strftime('%Y-%m-%d %H:%M')} ET -> run {to_run}")
                call_agents(to_run)
                for a in to_run:
                    mark_run(last_ts, a)
            time.sleep(tick)
        except KeyboardInterrupt:
            print("\n[sched] bye")
            break
        except Exception as e:
            print(f"[sched] error: {e}")
            time.sleep(tick)

if __name__ == "__main__":
    main()
