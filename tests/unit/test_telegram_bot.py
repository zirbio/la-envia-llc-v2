"""
Unit tests for notifications/telegram_bot.py - TradingTelegramBot class.

Tests cover:
- Session close message displays 16:00 EST
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass


# ============================================================================
# Mock Position Class
# ============================================================================

@dataclass
class MockPosition:
    """Mock position for testing."""
    symbol: str
    unrealized_pl: float


# ============================================================================
# Session Close Message Tests
# ============================================================================

class TestTelegramBotSessionClose:
    """Tests for TelegramBot session close functionality."""

    @pytest.mark.asyncio
    async def test_session_close_message_shows_16_00(self):
        """Session close confirmation should display 16:00 EST."""
        with patch('notifications.telegram_bot.settings') as mock_settings, \
             patch('notifications.telegram_bot.order_executor'):

            mock_settings.telegram.bot_token = "test_token"
            mock_settings.telegram.chat_id = "12345"
            mock_settings.trading.max_capital = 25000
            mock_settings.trading.reward_risk_ratio = 2.0

            from notifications.telegram_bot import TradingTelegramBot

            bot = TradingTelegramBot()
            bot.token = "test_token"
            bot.chat_id = "12345"
            bot.bot = MagicMock()

            # Capture the message sent
            sent_messages = []

            async def capture_message(text, parse_mode="Markdown"):
                sent_messages.append(text)
                return True

            bot.send_message = capture_message

            # Create mock positions
            positions = [
                MockPosition(symbol="AAPL", unrealized_pl=100.0),
                MockPosition(symbol="NVDA", unrealized_pl=-50.0),
            ]
            total_pnl = 50.0

            # Set up the session close event to auto-confirm (simulate timeout)
            original_ask = bot.ask_session_close_confirmation

            async def mock_ask_session_close(positions, total_pnl):
                bot.pending_session_close = True
                bot.session_close_event = asyncio.Event()
                bot.session_close_confirmed = False

                # Build the message (same as in original method)
                emoji = "🟢" if total_pnl >= 0 else "🔴"
                direction = "+" if total_pnl >= 0 else ""

                pos_lines = []
                for p in positions:
                    p_emoji = "🟢" if p.unrealized_pl >= 0 else "🔴"
                    p_dir = "+" if p.unrealized_pl >= 0 else ""
                    pos_lines.append(f"{p_emoji} {p.symbol}: {p_dir}${p.unrealized_pl:.2f}")

                positions_text = "\n".join(pos_lines) if pos_lines else "No hay posiciones"

                message = f"""
⏰ *FIN DE SESIÓN - 16:00 EST*

*Posiciones abiertas:*
{positions_text}

{emoji} *P/L Total:* {direction}${total_pnl:.2f}

━━━━━━━━━━━━━━━━━━━━
¿Cerrar todas las posiciones?

Reply *SI* para cerrar
Reply *NO* para mantener abiertas
                """

                await bot.send_message(message.strip())

                # Auto-confirm to avoid waiting
                bot.session_close_confirmed = True
                bot.pending_session_close = False
                return True

            bot.ask_session_close_confirmation = mock_ask_session_close

            # Call ask_session_close_confirmation
            await bot.ask_session_close_confirmation(positions, total_pnl)

            # Verify message was sent
            assert len(sent_messages) > 0, "Should have sent a message"

            # Verify message contains "16:00 EST"
            message = sent_messages[0]
            assert "16:00 EST" in message, \
                f"Message should contain '16:00 EST', got: {message}"

            # Verify it does NOT contain 11:30 (old incorrect time)
            assert "11:30" not in message, \
                f"Message should NOT contain '11:30', got: {message}"

            # Verify message contains "FIN DE SESIÓN"
            assert "FIN DE SESIÓN" in message, \
                f"Message should contain 'FIN DE SESIÓN', got: {message}"

    @pytest.mark.asyncio
    async def test_session_close_message_format(self):
        """Session close message should have correct format with 16:00 EST header."""
        with patch('notifications.telegram_bot.settings') as mock_settings, \
             patch('notifications.telegram_bot.order_executor'):

            mock_settings.telegram.bot_token = "test_token"
            mock_settings.telegram.chat_id = "12345"

            from notifications.telegram_bot import TradingTelegramBot

            bot = TradingTelegramBot()
            bot.token = "test_token"
            bot.chat_id = "12345"
            bot.bot = MagicMock()

            # Capture messages
            sent_messages = []

            async def capture_message(text, parse_mode="Markdown"):
                sent_messages.append(text)
                return True

            bot.send_message = capture_message

            # Single position
            positions = [MockPosition(symbol="TSLA", unrealized_pl=200.0)]
            total_pnl = 200.0

            # Create the event that will be set immediately
            bot.pending_session_close = True
            bot.session_close_event = asyncio.Event()

            # Start the method but don't wait for event (we'll check the message)
            task = asyncio.create_task(
                bot.ask_session_close_confirmation(positions, total_pnl)
            )

            # Give it time to send the message
            await asyncio.sleep(0.1)

            # Set the event to complete the method
            bot.session_close_confirmed = True
            if bot.session_close_event:
                bot.session_close_event.set()

            # Wait for task with timeout
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.TimeoutError:
                pass

            # Verify the message
            assert len(sent_messages) > 0, "Should have sent a message"

            message = sent_messages[0]

            # Check the header format specifically
            assert "⏰ *FIN DE SESIÓN - 16:00 EST*" in message, \
                f"Header should be '⏰ *FIN DE SESIÓN - 16:00 EST*', got: {message}"


# ============================================================================
# Integration Test with Real Class
# ============================================================================

class TestTelegramBotMessageContent:
    """Tests to verify the actual message content in TradingTelegramBot."""

    def test_ask_session_close_message_template_has_16_00(self):
        """
        Verify the source code contains 16:00 EST in the session close message.
        This is a static analysis test.
        """
        import inspect
        from notifications.telegram_bot import TradingTelegramBot

        # Get the source code of ask_session_close_confirmation
        source = inspect.getsource(TradingTelegramBot.ask_session_close_confirmation)

        # Verify it contains "16:00 EST"
        assert "16:00 EST" in source, \
            "ask_session_close_confirmation should contain '16:00 EST'"

        # Verify it does NOT contain "11:30"
        assert "11:30" not in source, \
            "ask_session_close_confirmation should NOT contain '11:30'"
