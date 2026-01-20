"""
Live test - Run scanner and monitoring immediately
"""
import asyncio
from dotenv import load_dotenv
from loguru import logger
import sys

load_dotenv()

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}", level="INFO")

async def run_live_test():
    from telegram import Bot
    import os
    from data.market_data import market_data
    from scanner.premarket import premarket_scanner, SCAN_UNIVERSE
    from strategy.orb import orb_strategy
    from execution.orders import order_executor

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    bot = Bot(token=token)

    # Check market status
    logger.info("Checking market status...")
    clock_info = order_executor.get_next_market_times()
    is_open = clock_info.get('is_open', False)

    if is_open:
        await bot.send_message(chat_id=chat_id, text="🟢 *Mercado ABIERTO*\nEjecutando scanner...", parse_mode="Markdown")
    else:
        await bot.send_message(chat_id=chat_id, text="🔴 *Mercado CERRADO*\nEjecutando scanner de prueba...", parse_mode="Markdown")

    # Run scanner on a subset of popular stocks
    test_symbols = ["NVDA", "TSLA", "AAPL", "AMD", "META", "GOOGL", "MSFT", "AMZN", "SPY", "QQQ"]

    logger.info(f"Scanning {len(test_symbols)} symbols...")

    candidates = []
    for symbol in test_symbols:
        try:
            # Get current price and daily change
            quote = market_data.get_latest_quote(symbol)
            if not quote:
                continue

            daily_bars = market_data.get_daily_bars(symbol, days=2)
            if daily_bars.empty:
                continue

            prev_close = daily_bars['close'].iloc[-2] if len(daily_bars) >= 2 else daily_bars['close'].iloc[-1]
            current_price = quote['mid']
            change_pct = ((current_price - prev_close) / prev_close) * 100

            avg_volume = int(daily_bars['volume'].mean())

            candidates.append({
                'symbol': symbol,
                'price': current_price,
                'change_pct': change_pct,
                'avg_volume': avg_volume
            })

            logger.info(f"{symbol}: ${current_price:.2f} ({change_pct:+.2f}%)")

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")

    # Sort by absolute change
    candidates.sort(key=lambda x: abs(x['change_pct']), reverse=True)

    # Format message
    lines = ["📊 *SCAN RESULTS*\n"]
    for c in candidates[:10]:
        emoji = "🟢" if c['change_pct'] > 0 else "🔴"
        lines.append(f"{emoji} *{c['symbol']}* ${c['price']:.2f} ({c['change_pct']:+.1f}%)")

    message = "\n".join(lines)
    await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")

    logger.info("Scan complete!")

    # Get account info
    account = order_executor.get_account()
    positions = order_executor.get_positions()

    account_msg = f"""
📈 *CUENTA*

Equity: ${account.get('equity', 0):,.2f}
Cash: ${account.get('cash', 0):,.2f}
Buying Power: ${account.get('buying_power', 0):,.2f}

Posiciones abiertas: {len(positions)}
    """
    await bot.send_message(chat_id=chat_id, text=account_msg.strip(), parse_mode="Markdown")

    if positions:
        pos_lines = ["📊 *POSICIONES*\n"]
        for p in positions:
            emoji = "🟢" if p.unrealized_pl >= 0 else "🔴"
            pos_lines.append(f"{emoji} *{p.symbol}* {p.qty} @ ${p.entry_price:.2f}")
            pos_lines.append(f"   P/L: ${p.unrealized_pl:+.2f} ({p.unrealized_pl_pct:+.1f}%)")
        await bot.send_message(chat_id=chat_id, text="\n".join(pos_lines), parse_mode="Markdown")

    logger.info("Live test complete!")

if __name__ == "__main__":
    asyncio.run(run_live_test())
