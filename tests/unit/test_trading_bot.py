"""
Unit tests for main.py - TradingBot class.

Tests cover:
- Scheduler sets end_session job at 16:00 EST (market close)
- Monitor breakouts stops after 16:00 EST
- _check_immediate_start respects 16:00 trading window end
"""
import pytest
import asyncio
from datetime import datetime, time
from unittest.mock import MagicMock, AsyncMock, patch
import pytz

# Import will be done after mocking to avoid side effects
EST = pytz.timezone('US/Eastern')


def safe_shutdown_scheduler(scheduler):
    """Safely shutdown scheduler only if it's running."""
    if scheduler.running:
        scheduler.shutdown(wait=False)


# ============================================================================
# Scheduler Tests
# ============================================================================

class TestTradingBotScheduler:
    """Tests for TradingBot scheduler configuration."""

    def test_scheduler_end_session_job_at_16_00(self):
        """Scheduler should set end_session job at 16:00 EST (market close)."""
        # Mock all dependencies before importing TradingBot
        with patch('main.settings') as mock_settings, \
             patch('main.market_data'), \
             patch('main.premarket_scanner'), \
             patch('main.orb_strategy'), \
             patch('main.order_executor'), \
             patch('main.telegram_bot'):

            mock_settings.telegram.bot_token = ""
            mock_settings.alpaca.paper = True
            mock_settings.trading.max_capital = 25000

            from main import TradingBot

            bot = TradingBot()
            bot._schedule_tasks()

            # Get the end_session job
            end_session_job = bot.scheduler.get_job('end_session')

            assert end_session_job is not None, "end_session job should exist"

            # Check the trigger is CronTrigger with hour=16, minute=0
            trigger = end_session_job.trigger

            # Extract hour and minute from the CronTrigger fields
            # CronTrigger stores fields as list of CronField objects
            hour_field = None
            minute_field = None

            for field in trigger.fields:
                if field.name == 'hour':
                    hour_field = field
                elif field.name == 'minute':
                    minute_field = field

            assert hour_field is not None, "Hour field should exist"
            assert minute_field is not None, "Minute field should exist"

            # The expressions contain the scheduled values
            assert '16' in str(hour_field), f"Hour should be 16, got {hour_field}"
            assert '0' in str(minute_field), f"Minute should be 0, got {minute_field}"

            # Clean up scheduler
            safe_shutdown_scheduler(bot.scheduler)


# ============================================================================
# Monitor Breakouts Tests
# ============================================================================

class TestTradingBotMonitorBreakouts:
    """Tests for TradingBot _monitor_breakouts method."""

    @pytest.mark.asyncio
    async def test_monitor_breakouts_stops_after_16_00(self):
        """Monitoring should stop when time >= 16:00 EST."""
        # Mock time to return 16:01 EST
        mock_datetime = MagicMock()
        mock_now = datetime(2024, 1, 15, 16, 1, 0, tzinfo=EST)
        mock_datetime.now.return_value = mock_now

        with patch('main.settings') as mock_settings, \
             patch('main.market_data'), \
             patch('main.premarket_scanner'), \
             patch('main.orb_strategy'), \
             patch('main.order_executor'), \
             patch('main.telegram_bot') as mock_telegram, \
             patch('main.datetime', mock_datetime):

            mock_settings.telegram.bot_token = ""
            mock_settings.alpaca.paper = True
            mock_settings.trading.max_capital = 25000
            mock_telegram.is_paused = False

            from main import TradingBot

            bot = TradingBot()
            bot.is_running = True
            bot.watchlist = ["AAPL", "NVDA"]

            # Track if _check_symbol was called
            check_symbol_called = False

            async def mock_check_symbol(symbol):
                nonlocal check_symbol_called
                check_symbol_called = True

            bot._check_symbol = mock_check_symbol

            # Run _monitor_breakouts - it should exit immediately
            await bot._monitor_breakouts()

            # _check_symbol should NOT be called because time >= 16:00
            assert check_symbol_called is False, \
                "Should not check symbols after 16:00 EST"

            # Clean up scheduler
            safe_shutdown_scheduler(bot.scheduler)

    @pytest.mark.asyncio
    async def test_monitor_breakouts_continues_before_16_00(self):
        """Monitoring should continue when time < 16:00 EST."""
        # Mock time to return 15:30 EST
        mock_datetime = MagicMock()
        mock_now = datetime(2024, 1, 15, 15, 30, 0, tzinfo=EST)
        mock_datetime.now.return_value = mock_now

        with patch('main.settings') as mock_settings, \
             patch('main.market_data'), \
             patch('main.premarket_scanner'), \
             patch('main.orb_strategy'), \
             patch('main.order_executor'), \
             patch('main.telegram_bot') as mock_telegram, \
             patch('main.datetime', mock_datetime), \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:

            mock_settings.telegram.bot_token = ""
            mock_settings.alpaca.paper = True
            mock_settings.trading.max_capital = 25000
            mock_telegram.is_paused = False

            from main import TradingBot

            bot = TradingBot()
            bot.is_running = True
            bot.watchlist = ["AAPL"]

            # Track if _check_symbol was called
            check_symbol_called = False
            call_count = 0

            async def mock_check_symbol(symbol):
                nonlocal check_symbol_called, call_count
                check_symbol_called = True
                call_count += 1
                # Stop after first check to avoid infinite loop
                bot.is_running = False

            bot._check_symbol = mock_check_symbol

            # Run _monitor_breakouts
            await bot._monitor_breakouts()

            # _check_symbol should be called at least once before 16:00
            assert check_symbol_called is True, \
                "Should check symbols before 16:00 EST"
            assert call_count >= 1, \
                "Should have called _check_symbol at least once"

            # Clean up scheduler
            safe_shutdown_scheduler(bot.scheduler)


# ============================================================================
# Immediate Start Tests
# ============================================================================

class TestTradingBotCheckImmediateStart:
    """Tests for TradingBot _check_immediate_start method."""

    @pytest.mark.asyncio
    async def test_check_immediate_start_respects_16_00_window(self):
        """_check_immediate_start should respect 16:00 trading window end."""
        # Mock time to return 16:01 EST
        mock_datetime = MagicMock()
        mock_now = datetime(2024, 1, 15, 16, 1, 0, tzinfo=EST)
        mock_datetime.now.return_value = mock_now

        with patch('main.settings') as mock_settings, \
             patch('main.market_data'), \
             patch('main.premarket_scanner'), \
             patch('main.orb_strategy'), \
             patch('main.order_executor') as mock_executor, \
             patch('main.telegram_bot') as mock_telegram, \
             patch('main.datetime', mock_datetime):

            mock_settings.telegram.bot_token = ""
            mock_settings.alpaca.paper = True
            mock_settings.trading.max_capital = 25000

            # Mock market as open
            mock_executor.get_next_market_times.return_value = {'is_open': True}

            # Track messages sent
            sent_messages = []
            mock_telegram.send_message = AsyncMock(
                side_effect=lambda msg: sent_messages.append(msg)
            )

            from main import TradingBot

            bot = TradingBot()

            # Track if monitoring was started
            monitoring_started = False

            async def mock_start_monitoring():
                nonlocal monitoring_started
                monitoring_started = True

            bot._start_monitoring = mock_start_monitoring

            # Call _check_immediate_start
            await bot._check_immediate_start()

            # Monitoring should NOT be started after 16:00
            assert monitoring_started is False, \
                "Should not start monitoring after 16:00 EST"

            # Should send "Ventana de trading cerrada" message
            assert len(sent_messages) > 0, "Should send a message"
            assert any("16:00 EST" in msg for msg in sent_messages), \
                "Message should mention 16:00 EST"
            assert any("cerrada" in msg.lower() for msg in sent_messages), \
                "Message should indicate window is closed"

            # Clean up scheduler
            safe_shutdown_scheduler(bot.scheduler)

    @pytest.mark.asyncio
    async def test_check_immediate_start_uses_time_16_0_as_session_end(self):
        """session_end variable should be time(16, 0) in _check_immediate_start."""
        # This test verifies the code uses time(16, 0) as the session end

        # Mock time to return exactly 16:00 EST (boundary case)
        mock_datetime = MagicMock()
        mock_now = datetime(2024, 1, 15, 16, 0, 0, tzinfo=EST)
        mock_datetime.now.return_value = mock_now

        with patch('main.settings') as mock_settings, \
             patch('main.market_data'), \
             patch('main.premarket_scanner'), \
             patch('main.orb_strategy'), \
             patch('main.order_executor') as mock_executor, \
             patch('main.telegram_bot') as mock_telegram, \
             patch('main.datetime', mock_datetime):

            mock_settings.telegram.bot_token = ""
            mock_settings.alpaca.paper = True
            mock_settings.trading.max_capital = 25000

            # Mock market as open
            mock_executor.get_next_market_times.return_value = {'is_open': True}

            sent_messages = []
            mock_telegram.send_message = AsyncMock(
                side_effect=lambda msg: sent_messages.append(msg)
            )

            from main import TradingBot

            bot = TradingBot()

            monitoring_started = False

            async def mock_start_monitoring():
                nonlocal monitoring_started
                monitoring_started = True

            bot._start_monitoring = mock_start_monitoring

            # Call _check_immediate_start at exactly 16:00
            await bot._check_immediate_start()

            # At exactly 16:00, the condition current_time >= session_end is True
            # so monitoring should NOT be started
            assert monitoring_started is False, \
                "Should not start monitoring at exactly 16:00 EST (boundary)"

            # Clean up scheduler
            safe_shutdown_scheduler(bot.scheduler)

    @pytest.mark.asyncio
    async def test_check_immediate_start_starts_monitoring_before_16_00(self):
        """_check_immediate_start should start monitoring before 16:00 EST."""
        # Mock time to return 10:00 EST (within trading window)
        mock_datetime = MagicMock()
        mock_now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=EST)
        mock_datetime.now.return_value = mock_now

        with patch('main.settings') as mock_settings, \
             patch('main.market_data'), \
             patch('main.premarket_scanner') as mock_scanner, \
             patch('main.orb_strategy'), \
             patch('main.order_executor') as mock_executor, \
             patch('main.telegram_bot') as mock_telegram, \
             patch('main.datetime', mock_datetime):

            mock_settings.telegram.bot_token = ""
            mock_settings.alpaca.paper = True
            mock_settings.trading.max_capital = 25000

            # Mock market as open
            mock_executor.get_next_market_times.return_value = {'is_open': True}

            # Mock scanner
            mock_scanner.scan_watchlist_with_sentiment = AsyncMock(return_value=[])
            mock_scanner.format_watchlist_message.return_value = "Watchlist"

            mock_telegram.send_message = AsyncMock()
            mock_telegram.send_watchlist = AsyncMock()

            from main import TradingBot

            bot = TradingBot()

            monitoring_started = False

            async def mock_start_monitoring():
                nonlocal monitoring_started
                monitoring_started = True

            bot._start_monitoring = mock_start_monitoring

            # Call _check_immediate_start at 10:00 (after 9:46, in monitoring window)
            await bot._check_immediate_start()

            # Monitoring SHOULD be started before 16:00
            assert monitoring_started is True, \
                "Should start monitoring before 16:00 EST"

            # Clean up scheduler
            safe_shutdown_scheduler(bot.scheduler)
