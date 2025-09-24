from dotenv import load_dotenv
load_dotenv(override=True)

import os, requests
from alpaca_trade_api import REST

def show(var):
    v = os.getenv(var)
    return f"{var}={repr(v)} len={len(v) if v else 0}"

print(show("APCA_API_BASE_URL"))
print(show("APCA_API_DATA_URL"))
print(show("APCA_API_KEY_ID"))
print(show("APCA_API_SECRET_KEY"))

base = os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"
key  = os.getenv("APCA_API_KEY_ID")
sec  = os.getenv("APCA_API_SECRET_KEY")

# Raw HTTP to /v2/account (bypasses SDK, shows exact response)
r = requests.get(
    f"{base}/v2/account",
    headers={"APCA-API-KEY-ID": key or "", "APCA-API-SECRET-KEY": sec or ""},
    timeout=15
)
print("HTTP status:", r.status_code)
print("HTTP body  :", r.text[:300])

# SDK call too
try:
    api = REST(key, sec, base_url=base, api_version="v2")
    acct = api.get_account()
    print("SDK equity :", acct.equity)
    print("SDK status :", acct.status)
except Exception as e:
    print("SDK ERROR  :", e)
