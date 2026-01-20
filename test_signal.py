"""Test signal message format"""
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

async def test_signal():
    from telegram import Bot

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    bot = Bot(token=token)

    # Simulated signal message
    message = """
🟢 *SIGNAL: LONG NVDA*

*Entry:* $142.50
*ORB High:* $141.80 ✓
*ORB Low:* $139.20
*VWAP:* $140.20 ✓
*Volume:* 2.3x ✓
*RSI:* 58

━━━━━━━━━━━━━━━━━━━━
*Stop Loss:* $139.20 (-2.3%)
*Take Profit:* $148.10 (+3.9%)
*Position:* 17 acciones ($2,422)
*Risk:* $56.10 (1.6%)
*Reward:* $95.20 (2:1)
━━━━━━━━━━━━━━━━━━━━

Reply *SI* para ejecutar o *NO* para ignorar
    """

    await bot.send_message(
        chat_id=chat_id,
        text=message.strip(),
        parse_mode="Markdown"
    )
    print("Signal message sent!")

if __name__ == "__main__":
    asyncio.run(test_signal())
