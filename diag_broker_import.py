from bot.providers.registry import make_brokers
import os

print("APCA_API_BASE_URL =", os.getenv("APCA_API_BASE_URL"))
print("APCA_API_KEY_ID set?", bool(os.getenv("APCA_API_KEY_ID")))
print("APCA_API_SECRET_KEY set?", bool(os.getenv("APCA_API_SECRET_KEY")))

brokers = make_brokers({"alpaca": {}}) if False else make_brokers  # just avoid linter noise
brokers = make_brokers({"providers":{"broker":["alpaca"]}, "alpaca":{}})  # minimal cfg shape
b = brokers.get("alpaca")
print("Broker type:", type(b).__name__)
print("Equity:", b.get_equity())
