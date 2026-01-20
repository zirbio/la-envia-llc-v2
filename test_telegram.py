"""Quick test for Telegram connection"""
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

async def test_telegram():
    from telegram import Bot

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    print(f"Token: {token[:20]}...")
    print(f"Chat ID: {chat_id}")

    bot = Bot(token=token)

    # Send test message
    await bot.send_message(
        chat_id=chat_id,
        text="🤖 *Test de conexión exitoso!*\n\nEl bot de trading está listo.",
        parse_mode="Markdown"
    )
    print("Message sent successfully!")

if __name__ == "__main__":
    asyncio.run(test_telegram())
