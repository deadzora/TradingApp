
def atr_position_size(equity, per_trade_risk_frac, price, atr_pct, stop_frac_default=0.01):
    stop_frac = max(atr_pct, stop_frac_default)
    risk_dollars = equity * per_trade_risk_frac
    dps = price * stop_frac
    if dps <= 0: return 0, stop_frac
    qty = max(0, int(risk_dollars / dps))
    return qty, stop_frac
