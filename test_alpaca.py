"""Quick test for Alpaca connection"""
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

def test_alpaca():
    from alpaca.trading.client import TradingClient

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    print(f"API Key: {api_key[:10]}...")

    client = TradingClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=True
    )

    # Get account info
    account = client.get_account()

    print("\n✅ Conexión exitosa!")
    print(f"Account: {account.account_number}")
    print(f"Status: {account.status}")
    print(f"Cash: ${float(account.cash):,.2f}")
    print(f"Equity: ${float(account.equity):,.2f}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")

    # Check market status
    clock = client.get_clock()
    print(f"\nMarket Open: {clock.is_open}")
    print(f"Next Open: {clock.next_open}")
    print(f"Next Close: {clock.next_close}")

if __name__ == "__main__":
    test_alpaca()
