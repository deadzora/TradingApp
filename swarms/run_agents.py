# swarms/run_agents.py
from __future__ import annotations
import time, yaml
from datetime import datetime, timezone
from pathlib import Path
from .data import DataAPI
from .orchestrator import fuse_proposals
from .risk import AccountState, can_open_more, position_size
from .schemas import OrderIntent
from .agents.penny import PennyAgent
from .agents.daytrade import DaytradeAgent
from .agents.options import OptionsAgent
from .agents.crypto import CryptoAgent

def load_cfg(): return yaml.safe_load(Path("swarms/config.yaml").read_text(encoding="utf-8"))

def main():
    cfg = load_cfg()
    data = DataAPI()

    agents = [
        PennyAgent(data, cfg),
        DaytradeAgent(data, cfg),
        OptionsAgent(data, cfg),
        CryptoAgent(data, cfg),
    ]

    # TODO: wire to your actual account/broker for real equity/positions
    state = AccountState(equity=100_000.0, open_positions={}, todays_pnl_fraction=0.0)

    while True:
        all_props = []
        for a in agents:
            try:
                props = a.scan()
                all_props.extend(props)
            except Exception as e:
                print(f"[{a.name}] scan error: {e}")

        intents = fuse_proposals(all_props, cfg)
        print(f"[{datetime.now(timezone.utc).isoformat()}] intents: {len(intents)}")

        # apply risk + sizing + order translation
        new_trades = 0
        for i in intents:
            if new_trades >= cfg["orchestrator"]["max_new_trades_per_cycle"]:
                break

            # per-agent open count (stub: 0). You can track by reading broker positions.
            agent_open_count = 0
            if not can_open_more(state, cfg, agent_open_count):
                continue
            if i.symbol in cfg["risk"]["ban_list"]:
                continue

            # price stub (replace with last/close from DataAPI)
            px = i.entry_limit if i.entry_limit > 0 else 100.0

            # entry slop to make limit marketable (demo)
            px = px * (1.0 + cfg["order_defaults"]["entry_slop_fraction"])

            qty = position_size(
                equity=state.equity * cfg["risk"]["equity_scale"],
                px=px,
                per_trade_risk_fraction=cfg["risk"]["per_trade_risk_fraction"],
                stop_fraction=cfg["order_defaults"]["stop_fraction"],
                min_qty=cfg["risk"]["min_qty"],
            )
            if qty <= 0: continue

            i.qty = qty
            i.entry_limit = px
            i.stop_price = px * (1 - cfg["order_defaults"]["stop_fraction"])
            i.take_limit = px * (1 + cfg["order_defaults"]["take_fraction"])

            # >>> replace this print with your Alpaca LIMIT bracket submit <<<
            print(f"[order] {i.symbol} {i.qty} @~{i.entry_limit:.2f} "
                  f"SL {i.stop_price:.2f} TP {i.take_limit:.2f} | {i.meta}")

            new_trades += 1

        time.sleep(cfg["runtime"]["cadence_seconds"])

if __name__ == "__main__":
    main()
