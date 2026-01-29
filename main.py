"""
Alpaca ORB Trading Bot - Main Entry Point

Semi-autonomous day trading bot using Opening Range Breakout strategy
with VWAP and Volume confirmation. Sends signals via Telegram for confirmation.
Supports extended hours trading (premarket/postmarket) with interactive CLI.
"""
import argparse
import asyncio
import signal
import sys
import threading
from datetime import datetime, time
from pathlib import Path
from typing import Optional
import pytz
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import settings, TradingMode
from data.market_data import market_data
from scanner.premarket import premarket_scanner, SCAN_UNIVERSE_FALLBACK
from scanner.universe import universe_manager
from strategy.orb import orb_strategy, TradeSignal
from strategy.extended_hours import (
    premarket_strategy, postmarket_strategy,
    get_current_session, is_in_trading_window, ExtendedHoursSignal
)
from execution.orders import order_executor
from execution.position_manager import position_manager, PositionEvent
from notifications.telegram_bot import telegram_bot


# Ensure logs directory exists (Windows compatibility)
Path("logs").mkdir(exist_ok=True)

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO"
)
logger.add(
    "logs/trading_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG"
)

# Timezone
EST = pytz.timezone('US/Eastern')


def _categorize_order_error(error: str) -> tuple[str, str]:
    """
    Categorize order errors into user-friendly messages.

    Args:
        error: Raw error string from order execution

    Returns:
        Tuple of (user_friendly_message, suggested_action)
    """
    error_lower = error.lower() if error else ""

    if "insufficient" in error_lower or "buying power" in error_lower:
        return (
            "Capital insuficiente para esta operacion",
            "Revisar balance de cuenta o reducir tamano de posicion"
        )
    elif "market closed" in error_lower or "market is closed" in error_lower:
        return (
            "Mercado cerrado",
            "Esperar apertura del mercado"
        )
    elif "not found" in error_lower or "symbol" in error_lower:
        return (
            "Simbolo no encontrado o no disponible",
            "Verificar que el simbolo es correcto y tradeable"
        )
    elif "timeout" in error_lower:
        return (
            "Tiempo de espera agotado",
            "El mercado puede estar muy volatil. Reintentar mas tarde"
        )
    elif "rejected" in error_lower:
        return (
            "Orden rechazada por el broker",
            "Verificar parametros de la orden y restricciones de cuenta"
        )
    elif "partial" in error_lower:
        return (
            "Orden parcialmente ejecutada",
            "Verificar posicion actual en Alpaca dashboard"
        )
    elif "rate limit" in error_lower or "too many" in error_lower:
        return (
            "Limite de peticiones excedido",
            "Esperar unos segundos antes de reintentar"
        )
    else:
        # Truncate long errors for user display
        truncated = error[:100] + "..." if len(error) > 100 else error
        return (
            f"Error tecnico: {truncated}",
            "Contactar soporte si persiste"
        )


class TradingBot:
    """Main trading bot orchestrator"""

    def __init__(self, trading_mode: TradingMode = None):
        self.scheduler = AsyncIOScheduler(timezone=EST)
        self.is_running = False
        self.watchlist: list[str] = []
        self.trades_today: list[dict] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.position_monitoring_task: Optional[asyncio.Task] = None
        self.premarket_monitoring_task: Optional[asyncio.Task] = None
        self.postmarket_monitoring_task: Optional[asyncio.Task] = None
        # Trading mode with lock for thread safety
        self._mode_lock = threading.Lock()
        self.trading_mode = trading_mode or settings.trading.trading_mode
        # Adaptive polling
        self.current_interval = settings.trading.base_monitoring_interval
        self.last_signal_time: Optional[datetime] = None
        self.checks_without_signal = 0

    def set_trading_mode(self, mode: TradingMode, force: bool = False) -> bool:
        """
        Change the trading mode at runtime.

        Args:
            mode: New trading mode to set
            force: If True, allow change even when bot is running (use with caution)

        Returns:
            True if mode was changed, False if change was blocked
        """
        with self._mode_lock:
            # Block mode changes while bot is actively running unless forced
            if self.is_running and not force:
                logger.warning(
                    f"Cannot change trading mode while bot is running. "
                    f"Stop the bot first or use force=True (not recommended)."
                )
                return False

            old_mode = self.trading_mode
            self.trading_mode = mode

            # Update extended hours config based on mode
            if mode in (TradingMode.PREMARKET, TradingMode.ALL_SESSIONS):
                settings.extended_hours.premarket_enabled = True
            else:
                settings.extended_hours.premarket_enabled = False

            if mode in (TradingMode.POSTMARKET, TradingMode.ALL_SESSIONS):
                settings.extended_hours.postmarket_enabled = True
            else:
                settings.extended_hours.postmarket_enabled = False

            logger.info(f"Trading mode changed: {old_mode.value} -> {mode.value}")
            return True

    async def start(self):
        """Start the trading bot"""
        logger.info("=" * 50)
        logger.info("Starting Alpaca ORB Trading Bot")
        logger.info("=" * 50)

        # Initialize Telegram bot
        if settings.telegram.bot_token:
            await telegram_bot.initialize()
            telegram_bot.on_confirmation = self._on_trade_confirmed
            asyncio.create_task(telegram_bot.start_polling())

        # Initialize position manager event handler
        position_manager.on_event = self._on_position_event

        # Schedule daily tasks
        self._schedule_tasks()

        # Start scheduler
        self.scheduler.start()
        self.is_running = True

        logger.info("Bot started successfully")
        logger.info(f"Paper Trading: {settings.alpaca.paper}")
        logger.info(f"Trading Mode: {self.trading_mode.value.upper()}")
        logger.info(f"Max Capital: ${settings.trading.max_capital:,}")
        logger.info(f"Risk per trade: {settings.trading.risk_per_trade * 100}%")
        # Log signal level configuration
        level_info = orb_strategy.get_signal_level_info()
        logger.info(f"Signal Level: {level_info['level']}")
        logger.info(f"  - Min Signal Score: {level_info['min_signal_score']}")
        logger.info(f"  - Min Relative Volume: {level_info['min_relative_volume']}x")
        logger.info(f"  - ORB Range: {level_info['min_orb_range_pct']}% - {level_info['max_orb_range_pct']}%")
        logger.info(f"  - Latest Trade Time: {level_info['latest_trade_time']}")
        # Log extended hours config if enabled
        if self.trading_mode in (TradingMode.PREMARKET, TradingMode.ALL_SESSIONS):
            logger.info(f"Premarket: {settings.extended_hours.premarket_trade_start} - {settings.extended_hours.premarket_trade_end}")
        if self.trading_mode in (TradingMode.POSTMARKET, TradingMode.ALL_SESSIONS):
            logger.info(f"Postmarket: {settings.extended_hours.postmarket_trade_start} - {settings.extended_hours.postmarket_trade_end}")

        # Check if we're already in trading window
        await self._check_immediate_start()

        # Keep running
        try:
            while self.is_running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            await self.stop()

    async def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping trading bot...")
        self.is_running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()

        if self.position_monitoring_task:
            self.position_monitoring_task.cancel()

        self.scheduler.shutdown()

        if telegram_bot.is_running:
            await telegram_bot.send_message("🛑 *Bot detenido*")
            await telegram_bot.stop_polling()

        logger.info("Bot stopped")

    def _schedule_tasks(self):
        """Schedule daily trading tasks based on trading mode"""

        # Extended hours: Premarket tasks
        if self.trading_mode in (TradingMode.PREMARKET, TradingMode.ALL_SESSIONS):
            # Start premarket monitoring at 8:00 AM EST
            self.scheduler.add_job(
                self._start_premarket_monitoring,
                CronTrigger(hour=8, minute=0, timezone=EST),
                id='start_premarket',
                replace_existing=True
            )

            # End premarket at 9:15 AM (before regular scan)
            self.scheduler.add_job(
                self._end_premarket_session,
                CronTrigger(hour=9, minute=15, timezone=EST),
                id='end_premarket',
                replace_existing=True
            )

            logger.info("Premarket tasks scheduled (8:00 - 9:15 EST)")

        # Regular hours tasks
        if self.trading_mode in (TradingMode.REGULAR, TradingMode.ALL_SESSIONS):
            # Pre-market scan at 9:25 AM EST
            self.scheduler.add_job(
                self._run_premarket_scan,
                CronTrigger(hour=9, minute=25, timezone=EST),
                id='premarket_scan',
                replace_existing=True
            )

            # Calculate Opening Range at 9:35 AM EST (5 min after open)
            self.scheduler.add_job(
                self._calculate_opening_ranges,
                CronTrigger(hour=9, minute=35, timezone=EST),
                id='calculate_orb',
                replace_existing=True
            )

            # Start monitoring at 9:36 AM EST
            self.scheduler.add_job(
                self._start_monitoring,
                CronTrigger(hour=9, minute=36, timezone=EST),
                id='start_monitoring',
                replace_existing=True
            )

            # Close all positions at 16:00 (market close) EST
            self.scheduler.add_job(
                self._end_session,
                CronTrigger(hour=16, minute=0, timezone=EST),
                id='end_session',
                replace_existing=True
            )

            logger.info("Regular hours tasks scheduled (9:25 scan, 9:35 ORB, 9:36-16:00 monitoring)")

        # Extended hours: Postmarket tasks
        if self.trading_mode in (TradingMode.POSTMARKET, TradingMode.ALL_SESSIONS):
            # Start postmarket monitoring at 16:05 PM EST
            self.scheduler.add_job(
                self._start_postmarket_monitoring,
                CronTrigger(hour=16, minute=5, timezone=EST),
                id='start_postmarket',
                replace_existing=True
            )

            # End postmarket at 18:00 PM EST
            self.scheduler.add_job(
                self._end_postmarket_session,
                CronTrigger(hour=18, minute=0, timezone=EST),
                id='end_postmarket',
                replace_existing=True
            )

            # Force close all extended hours positions at 19:30
            self.scheduler.add_job(
                self._force_close_extended_positions,
                CronTrigger(hour=19, minute=30, timezone=EST),
                id='force_close_extended',
                replace_existing=True
            )

            logger.info("Postmarket tasks scheduled (16:05 - 18:00 EST, force close 19:30)")

        # Reset daily at 4:00 AM EST (before premarket)
        self.scheduler.add_job(
            self._reset_daily,
            CronTrigger(hour=4, minute=0, timezone=EST),
            id='reset_daily',
            replace_existing=True
        )

        logger.info(f"Scheduled tasks configured for mode: {self.trading_mode.value}")

    async def _run_premarket_scan(self):
        """Run pre-market scanner with sentiment analysis using dynamic universe"""
        logger.info("Running pre-market scan with sentiment analysis...")

        try:
            # Build universe if not already built (e.g., if bot started after 4 AM)
            if not universe_manager.is_cache_valid:
                logger.info("Building universe before premarket scan...")
                await universe_manager.build_universe()

            # Scan using dynamic universe with sentiment analysis
            candidates = await premarket_scanner.scan_dynamic_universe()

            if not candidates:
                logger.warning("No candidates found in pre-market scan")
                await telegram_bot.send_message(
                    f"⚠️ No se encontraron candidatos hoy\n"
                    f"(Escaneados: {len(universe_manager.get_cached_universe())} símbolos)"
                )
                return

            # Update watchlist
            self.watchlist = [c.symbol for c in candidates]

            # Update strategy sentiment cache
            for c in candidates:
                orb_strategy.update_sentiment(c.symbol, c.sentiment_score)

            # Phase 2: Cache intraday volume profiles for time-adjusted RVOL
            logger.info("Building intraday volume profiles for watchlist...")
            for symbol in self.watchlist:
                market_data.cache_volume_profile(symbol, lookback_days=20)

            # Send watchlist to Telegram
            message = premarket_scanner.format_watchlist_message()
            await telegram_bot.send_watchlist(message)

            # Log universe stats
            stats = universe_manager.stats
            logger.info(
                f"Watchlist: {self.watchlist} "
                f"(from {stats.tier3_filtered} high-volume universe)"
            )

        except Exception as e:
            logger.error(f"Error in pre-market scan: {e}")
            await telegram_bot.send_message(f"❌ Error en scanner: {e}")

    async def _calculate_opening_ranges(self):
        """Calculate Opening Range for watchlist symbols"""
        logger.info("Calculating Opening Ranges...")

        # Update open prices for context score calculation
        premarket_scanner.update_open_prices()

        for symbol in self.watchlist:
            orb = orb_strategy.calculate_opening_range(symbol)
            if orb:
                logger.info(f"{symbol} ORB: High=${orb.high:.2f}, Low=${orb.low:.2f}")

        await telegram_bot.send_message(
            f"📊 Opening Range calculado para {len(self.watchlist)} símbolos\n"
            "Comenzando monitoreo de breakouts..."
        )

    async def _start_monitoring(self):
        """Start monitoring for breakouts"""
        logger.info("Starting breakout monitoring...")
        self.monitoring_task = asyncio.create_task(self._monitor_breakouts())

        # Start position monitoring for trailing stops and partial closes
        logger.info("Starting position monitoring...")
        self.position_monitoring_task = asyncio.create_task(self._monitor_positions())

    async def _monitor_breakouts(self):
        """Monitor watchlist for breakout signals with adaptive polling"""
        logger.info("Monitoring for breakouts...")
        daily_loss_alert_sent = False  # Avoid spamming
        signal_found_this_cycle = False

        while self.is_running and not telegram_bot.is_paused:
            try:
                now = datetime.now(EST)

                # Stop monitoring after 16:00 (market close)
                if now.time() >= time(16, 0):
                    logger.info("Trading window closed")
                    break

                # Phase 6: Check daily loss limit circuit breaker
                if orb_strategy.daily_pnl <= -settings.trading.max_daily_loss:
                    if not daily_loss_alert_sent:
                        await telegram_bot.send_message(
                            f"🛑 *CIRCUIT BREAKER*\n"
                            f"Daily loss limit reached: ${orb_strategy.daily_pnl:.2f}\n"
                            f"No new trades until tomorrow."
                        )
                        daily_loss_alert_sent = True
                        logger.warning(f"Daily loss limit hit: ${orb_strategy.daily_pnl:.2f}")
                    await asyncio.sleep(60)  # Check less frequently when paused
                    continue

                # Check each symbol in watchlist
                signal_found_this_cycle = False
                for symbol in self.watchlist:
                    signal = await self._check_symbol(symbol)
                    if signal:
                        signal_found_this_cycle = True
                        self.last_signal_time = now
                        self.checks_without_signal = 0

                # Adaptive polling: adjust interval based on activity
                # This reduces API calls and CPU usage during quiet periods
                if signal_found_this_cycle:
                    # Signal found - use minimum interval for rapid response
                    self.current_interval = settings.trading.min_monitoring_interval
                else:
                    self.checks_without_signal += 1
                    # Adjust every 10 checks (~100s at 10s base interval) to avoid
                    # too-frequent changes while still being responsive to market activity.
                    # Increment by 5s to gradually reduce polling without large jumps.
                    if self.checks_without_signal % 10 == 0:
                        self.current_interval = min(
                            self.current_interval + 5,
                            settings.trading.max_monitoring_interval
                        )

                # Log monitoring status every 30 cycles (~5 min at 10s interval)
                # to provide visibility without excessive log spam
                if self.checks_without_signal > 0 and self.checks_without_signal % 30 == 0:
                    logger.info(
                        f"Monitoring: {len(self.watchlist)} symbols | "
                        f"Interval: {self.current_interval}s | "
                        f"Checks w/o signal: {self.checks_without_signal}"
                    )

                # Wait before next check (adaptive interval)
                await asyncio.sleep(self.current_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _check_symbol(self, symbol: str) -> Optional[TradeSignal]:
        """Check a symbol for breakout signal"""
        try:
            # Skip if already have an active position for this symbol
            if position_manager.get_position(symbol) is not None:
                return None

            # Skip if there's a pending confirmation for this symbol (only in manual mode)
            if settings.trading.execution_mode == "manual" and symbol in telegram_bot.pending_signals:
                return None

            # Get current quote
            quote = market_data.get_latest_quote(symbol)
            if not quote:
                return None

            # Validate quote freshness - reject stale quotes to prevent bad signals
            MAX_QUOTE_AGE_SECONDS = 60
            quote_age = quote.get('age_seconds', 0)
            if quote_age > MAX_QUOTE_AGE_SECONDS:
                logger.warning(
                    f"{symbol}: Quote too stale ({quote_age:.0f}s > {MAX_QUOTE_AGE_SECONDS}s), skipping"
                )
                return None

            current_price = quote['mid']

            # Get average volume for relative volume calculation
            avg_volume = market_data.get_avg_daily_volume(symbol)

            # Get recent bars for current volume
            bars = market_data.get_bars(symbol, limit=5)
            if bars.empty:
                return None

            current_volume = int(bars['volume'].iloc[-1])

            # Validate price is not anomalous (too far from ORB range)
            orb = orb_strategy.opening_ranges.get(symbol)
            if orb:
                orb_mid = (orb.high + orb.low) / 2
                price_deviation_pct = abs(current_price - orb_mid) / orb_mid

                # If price deviates more than 50% from ORB midpoint, likely bad data
                if price_deviation_pct > 0.5:
                    quote_age = quote.get('age_seconds', 0)

                    # Try fallback to last bar close
                    if not bars.empty:
                        bar_close = float(bars['close'].iloc[-1])
                        bar_deviation = abs(bar_close - orb_mid) / orb_mid

                        if bar_deviation <= 0.5:
                            logger.warning(
                                f"{symbol}: Quote ${current_price:.2f} (age={quote_age:.0f}s) anomalous, "
                                f"using bar close ${bar_close:.2f} instead"
                            )
                            current_price = bar_close
                        else:
                            logger.warning(
                                f"{symbol}: Both quote ${current_price:.2f} (age={quote_age:.0f}s) and "
                                f"bar close ${bar_close:.2f} anomalous vs ORB "
                                f"(${orb.low:.2f}-${orb.high:.2f}), skipping"
                            )
                            return None
                    else:
                        logger.warning(
                            f"{symbol}: Price ${current_price:.2f} (age={quote_age:.0f}s) too far from ORB "
                            f"(${orb.low:.2f}-${orb.high:.2f}), deviation={price_deviation_pct:.1%}, skipping"
                        )
                        return None

            # Check for breakout
            signal = orb_strategy.check_breakout(
                symbol=symbol,
                current_price=current_price,
                current_volume=current_volume,
                avg_volume=avg_volume
            )

            if signal:
                logger.info(f"Signal detected for {symbol}")

                if settings.trading.execution_mode == "auto":
                    # AUTO mode: execute immediately without confirmation
                    await self._on_trade_confirmed(signal)
                    # Get the fill price from the position manager
                    position = position_manager.get_position(symbol)
                    fill_price = position.entry_price if position else signal.entry_price
                    # Send informative notification (no buttons)
                    await telegram_bot.send_auto_execution_notification(signal, fill_price)
                else:
                    # MANUAL mode: send alert and wait for confirmation
                    await telegram_bot.send_signal_alert(signal)

                return signal

            return None

        except Exception as e:
            logger.error(f"Error checking {symbol}: {e}")
            return None

    async def _monitor_positions(self):
        """Monitor managed positions for partial closes and trailing stops"""
        logger.info("Position monitoring started...")

        while self.is_running and not telegram_bot.is_paused:
            try:
                now = datetime.now(EST)

                # Stop monitoring after 16:00 (market close)
                if now.time() >= time(16, 0):
                    logger.info("Position monitoring window closed")
                    break

                # Check all managed positions
                events = await position_manager.check_all_positions()

                # Handle any events that occurred
                for event in events:
                    await self._on_position_event(event)

                # Wait before next check (configurable interval)
                await asyncio.sleep(settings.trading.position_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in position monitoring loop: {e}")
                await asyncio.sleep(5)

        logger.info("Position monitoring stopped")

    async def _on_position_event(self, event: PositionEvent):
        """Handle position management events and send notifications"""
        try:
            if event.event_type == 'partial_close':
                await telegram_bot.send_partial_close_alert(
                    symbol=event.symbol,
                    closed_qty=event.details.get('closed_qty', 0),
                    close_price=event.details.get('close_price', 0),
                    remaining_qty=event.details.get('remaining_qty', 0),
                    new_stop=event.details.get('new_stop', 0),
                    realized_pnl=event.details.get('realized_pnl', 0),
                    r_multiple=event.details.get('r_multiple', 0)
                )

            elif event.event_type == 'trailing_activated':
                await telegram_bot.send_trailing_activated(
                    symbol=event.symbol,
                    ema_period=event.details.get('ema_period', 9),
                    timeframe=event.details.get('timeframe', '5min')
                )

            elif event.event_type == 'stop_updated':
                await telegram_bot.send_trailing_stop_update(
                    symbol=event.symbol,
                    old_stop=event.details.get('old_stop', 0),
                    new_stop=event.details.get('new_stop', 0),
                    ema9_value=event.details.get('ema9', 0)
                )

            elif event.event_type == 'position_closed':
                position = position_manager.get_position(event.symbol)
                if position:
                    # Use actual exit price from event (fill price), not trigger price
                    actual_exit_price = event.details.get('exit_price', position.current_stop_loss)

                    await telegram_bot.send_position_stopped(
                        symbol=event.symbol,
                        entry_price=position.entry_price,
                        stop_price=actual_exit_price,  # Use actual fill price for P&L
                        qty=position.current_qty,
                        reason=event.details.get('reason', 'unknown')
                    )

                    # Record trade result for Kelly calculation with actual fill price
                    orb_strategy.record_trade_result(
                        symbol=event.symbol,
                        entry_price=position.entry_price,
                        exit_price=actual_exit_price,  # Use actual fill price
                        is_long=(position.side == 'long')
                    )

        except Exception as e:
            logger.error(f"Error handling position event: {e}")

    async def _on_trade_confirmed(self, signal: TradeSignal):
        """Handle confirmed trade with correct stop loss based on actual fill price.

        Uses two-step order flow to calculate stop loss from fill price instead of signal price.
        This avoids the bug where OTO stop orders are placed at incorrect prices due to slippage.
        """
        logger.info(f"Trade confirmed: {signal.signal_type.value} {signal.symbol}")

        # Determine order side
        entry_side = 'buy' if signal.signal_type.value == 'LONG' else 'sell'
        position_side = 'long' if signal.signal_type.value == 'LONG' else 'short'

        # Calculate limit price with buffer based on direction
        limit_price = None
        if settings.trading.use_limit_entry:
            buffer_pct = settings.trading.limit_entry_buffer_pct
            if signal.signal_type.value == 'LONG':
                # For LONG (buy limit), set limit ABOVE entry to ensure fill
                limit_price = signal.entry_price * (1 + buffer_pct)
            else:
                # For SHORT (sell limit), set limit BELOW entry to ensure fill
                limit_price = signal.entry_price * (1 - buffer_pct)

        # STEP 1: Execute ONLY the entry order (no stop loss attached)
        entry_result = await order_executor.execute_entry_order_and_wait(
            symbol=signal.symbol,
            qty=signal.position_size,
            side=entry_side,
            use_limit=settings.trading.use_limit_entry,
            limit_price=limit_price,
            timeout=settings.trading.limit_order_fill_timeout
        )

        if not entry_result.success:
            # Use categorized error message for better UX
            user_msg, action = _categorize_order_error(entry_result.error or "Unknown error")
            logger.error(
                f"Entry order failed for {signal.symbol}: {entry_result.error}",
                extra={"symbol": signal.symbol, "error": entry_result.error}
            )
            await telegram_bot.send_message(
                f"❌ Error ejecutando orden para {signal.symbol}\n"
                f"Razon: {user_msg}\n"
                f"Accion: {action}"
            )
            return

        if not entry_result.filled_price:
            await telegram_bot.send_message(
                f"⚠️ Orden no ejecutada: {signal.symbol}\n"
                f"El precio limite no fue alcanzado. Orden cancelada."
            )
            return

        fill_price = entry_result.filled_price

        # STEP 2: Calculate stop loss based on ACTUAL FILL PRICE
        # Preserve the original risk per share from the signal
        original_risk_per_share = abs(signal.entry_price - signal.stop_loss)

        # Validate risk per share is reasonable (not zero or negative)
        MIN_RISK_PER_SHARE = 0.01
        if original_risk_per_share < MIN_RISK_PER_SHARE:
            logger.error(
                f"CRITICAL: Invalid risk per share ${original_risk_per_share:.4f} for {signal.symbol}. "
                f"signal.entry_price=${signal.entry_price:.2f}, signal.stop_loss=${signal.stop_loss:.2f}. "
                f"Closing position to prevent invalid stop loss."
            )
            # Close position immediately to avoid unprotected trade
            close_result = order_executor.close_position(signal.symbol)
            await telegram_bot.send_message(
                f"🚨 EMERGENCIA: Riesgo por accion invalido (${original_risk_per_share:.4f})\n"
                f"Posicion {signal.symbol} CERRADA automaticamente.\n"
                f"Resultado: {'Cerrada' if close_result.success else 'Error al cerrar - verificar manualmente'}"
            )
            return

        if signal.signal_type.value == 'LONG':
            # For LONG: stop loss is BELOW entry price
            actual_stop_loss = fill_price - original_risk_per_share
            # Floor: stop loss cannot be negative or zero
            actual_stop_loss = max(0.01, actual_stop_loss)
        else:
            # For SHORT: stop loss is ABOVE entry price
            actual_stop_loss = fill_price + original_risk_per_share

        actual_stop_loss = round(actual_stop_loss, 2)

        # STEP 3: Create stop loss as separate order WITH RETRY LOGIC
        MAX_STOP_RETRIES = 3
        stop_result = None
        stop_order_id = None

        for attempt in range(MAX_STOP_RETRIES):
            # Use asyncio.to_thread to avoid blocking the event loop
            stop_result = await asyncio.to_thread(
                order_executor.create_stop_loss_order,
                signal.symbol,
                signal.position_size,
                actual_stop_loss,
                position_side
            )

            if stop_result.success:
                stop_order_id = stop_result.order_id
                logger.info(f"Stop loss created: {signal.symbol} @ ${actual_stop_loss:.2f}")
                break

            if attempt < MAX_STOP_RETRIES - 1:
                wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s exponential backoff
                logger.warning(
                    f"Stop loss creation attempt {attempt + 1}/{MAX_STOP_RETRIES} failed for {signal.symbol}: "
                    f"{stop_result.error}. Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

        # Handle stop loss creation failure after all retries
        if not stop_result or not stop_result.success:
            logger.error(
                f"CRITICAL: Stop loss creation failed for {signal.symbol} after {MAX_STOP_RETRIES} attempts. "
                f"Position is UNPROTECTED. Error: {stop_result.error if stop_result else 'Unknown'}"
            )
            await telegram_bot.send_message(
                f"🚨 CRÍTICO: Stop loss FALLIDO para {signal.symbol}\n"
                f"Entrada: ${fill_price:.2f}\n"
                f"Stop requerido: ${actual_stop_loss:.2f}\n"
                f"Intentos: {MAX_STOP_RETRIES}\n"
                f"⚠️ CREAR STOP MANUAL INMEDIATAMENTE"
            )

        # Log verification that stop is correctly placed
        if signal.signal_type.value == 'LONG':
            stop_correct = actual_stop_loss < fill_price
        else:
            stop_correct = actual_stop_loss > fill_price

        logger.info(
            f"Position opened: {signal.signal_type.value} {signal.symbol} "
            f"Entry=${fill_price:.2f}, Stop=${actual_stop_loss:.2f}, "
            f"Risk=${original_risk_per_share:.2f}/share, StopCorrect={stop_correct}, "
            f"StopProtected={stop_order_id is not None}"
        )

        if not stop_correct:
            logger.error(
                f"CRITICAL: Stop loss incorrectly placed! "
                f"{signal.signal_type.value} entry=${fill_price:.2f} stop=${actual_stop_loss:.2f}"
            )

        # Register position for management with actual stop price
        try:
            await position_manager.register_position(
                signal=signal,
                fill_price=fill_price,
                qty=signal.position_size,
                stop_order_id=stop_order_id,
                actual_stop_price=actual_stop_loss
            )
        except Exception as e:
            logger.error(
                f"CRITICAL: Failed to register position for {signal.symbol}: {e}. "
                f"Position exists but is UNTRACKED."
            )
            await telegram_bot.send_message(
                f"⚠️ Posicion {signal.symbol} ejecutada pero NO registrada internamente.\n"
                f"Fill: ${fill_price:.2f}, Qty: {signal.position_size}\n"
                f"Stop: ${actual_stop_loss:.2f}\n"
                f"El bot NO monitoreara esta posicion automaticamente."
            )

        # Send confirmation
        await telegram_bot.send_execution_confirmation(
            symbol=signal.symbol,
            side=signal.signal_type.value,
            qty=signal.position_size,
            price=fill_price
        )

        # Notify about position management with actual stop price
        protection_status = "✅ Protegida" if stop_order_id else "⚠️ SIN PROTECCION"
        await telegram_bot.send_message(
            f"📊 *Gestion de posicion activada*\n"
            f"- Stop Loss: ${actual_stop_loss:.2f} ({protection_status})\n"
            f"- Cierre parcial (50%) al alcanzar 1R\n"
            f"- Stop movido a breakeven tras cierre parcial\n"
            f"- Trailing con EMA9 (5min) para el resto"
        )

        self.trades_today.append({
            'symbol': signal.symbol,
            'side': signal.signal_type.value,
            'entry': fill_price,
            'stop': actual_stop_loss,  # Use actual stop, not signal stop
            'target': signal.take_profit,
            'qty': signal.position_size,
            'time': datetime.now(EST),
            'protected': stop_order_id is not None  # Track protection status
        })

    async def _end_session(self):
        """End trading session"""
        logger.info("Ending trading session...")

        # Stop monitoring
        if self.monitoring_task:
            self.monitoring_task.cancel()

        # Stop position monitoring
        if self.position_monitoring_task:
            self.position_monitoring_task.cancel()

        # Get positions and P/L
        positions = order_executor.get_positions()
        total_pnl = sum(p.unrealized_pl for p in positions)

        if positions:
            # Ask for confirmation before closing
            should_close = await telegram_bot.ask_session_close_confirmation(positions, total_pnl)

            if should_close:
                order_executor.close_all_positions()
                await telegram_bot.send_message("✅ *Posiciones cerradas*")
                logger.info("User confirmed - positions closed")
            else:
                await telegram_bot.send_message("⚠️ *Posiciones mantenidas*\nUsa /close para cerrar manualmente")
                logger.info("User declined - positions kept open")
                return  # Don't send daily summary if positions still open
        else:
            await telegram_bot.send_message("🔔 *Sesión terminada* - No hay posiciones abiertas")

        # Calculate daily stats
        winners = sum(1 for t in self.trades_today if t.get('pnl', 0) > 0)

        await telegram_bot.send_daily_summary(
            trades=len(self.trades_today),
            winners=winners,
            total_pnl=total_pnl
        )

        logger.info(f"Session ended. Trades: {len(self.trades_today)}, P/L: ${total_pnl:.2f}")

    async def _reset_daily(self):
        """Reset daily data and build fresh universe"""
        logger.info("Resetting daily data...")
        orb_strategy.reset_daily()
        premarket_strategy.reset_daily()
        postmarket_strategy.reset_daily()
        premarket_scanner.reset()  # Reset premarket context cache
        position_manager.reset()
        self.watchlist.clear()
        self.trades_today.clear()
        # Reset adaptive polling state
        self.current_interval = settings.trading.base_monitoring_interval
        self.last_signal_time = None
        self.checks_without_signal = 0
        # Clear indicator cache
        from data.indicators import indicator_cache
        indicator_cache.invalidate()

        # Build fresh universe for the new day (4 AM - no Telegram notification needed)
        logger.info("Building fresh universe for the day...")
        universe_manager.reset()
        try:
            await universe_manager.build_universe(force=True)
            stats = universe_manager.stats
            logger.info(
                f"Universe ready: {stats.tier3_filtered} high-volume symbols "
                f"(build time: {stats.build_time_seconds:.1f}s)"
            )
        except Exception as e:
            logger.error(f"Error building universe: {e}")
            logger.warning("Will use fallback universe for today")

        logger.info("Daily reset complete")

    # ========== Extended Hours Methods ==========

    async def _start_premarket_monitoring(self):
        """Start premarket Gap & Go monitoring"""
        logger.info("Starting premarket monitoring (Gap & Go)...")
        await telegram_bot.send_message(
            "🌅 *PREMARKET SESSION*\n"
            f"Modo: Gap & Go\n"
            f"Horario: {settings.extended_hours.premarket_trade_start} - {settings.extended_hours.premarket_trade_end} EST\n"
            f"Tamaño posición: {settings.extended_hours.premarket_position_size_mult*100:.0f}%\n"
            f"Max trades: {settings.extended_hours.premarket_max_trades}"
        )
        self.premarket_monitoring_task = asyncio.create_task(self._monitor_premarket())

    async def _monitor_premarket(self):
        """Monitor watchlist for premarket Gap & Go signals"""
        logger.info("Monitoring premarket for gap opportunities...")

        # Get dynamic universe or fallback
        premarket_symbols = universe_manager.get_cached_universe()[:50]  # Top 50 for premarket
        if not premarket_symbols:
            premarket_symbols = SCAN_UNIVERSE_FALLBACK[:20]

        logger.info(f"Premarket monitoring {len(premarket_symbols)} symbols")

        while self.is_running:
            try:
                now = datetime.now(EST)

                # Check if still in premarket trading window
                if not is_in_trading_window(TradingMode.PREMARKET):
                    logger.info("Premarket trading window closed")
                    break

                # Scan for premarket candidates using dynamic universe
                for symbol in premarket_symbols:
                    signal = await self._check_premarket_symbol(symbol)
                    if signal:
                        await self._send_extended_hours_alert(signal)

                await asyncio.sleep(30)  # 30 second intervals for premarket

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in premarket monitoring: {e}")
                await asyncio.sleep(10)

    async def _check_premarket_symbol(self, symbol: str) -> Optional[ExtendedHoursSignal]:
        """Check a symbol for premarket trading opportunity"""
        try:
            quote = market_data.get_latest_quote(symbol)
            if not quote:
                return None

            current_price = quote['mid']
            avg_volume = market_data.get_avg_daily_volume(symbol)

            signal = premarket_strategy.check_gap_momentum(
                symbol=symbol,
                current_price=current_price,
                avg_volume=avg_volume
            )

            return signal

        except Exception as e:
            logger.error(f"Error checking premarket {symbol}: {e}")
            return None

    async def _end_premarket_session(self):
        """End premarket session, close positions before regular open"""
        logger.info("Ending premarket session...")

        if self.premarket_monitoring_task:
            self.premarket_monitoring_task.cancel()

        # Get premarket positions
        positions = order_executor.get_positions()
        if positions:
            await telegram_bot.send_message(
                f"🌅 *Fin sesión premarket*\n"
                f"{len(positions)} posicion(es) abiertas\n"
                f"Considerar cerrar antes de apertura regular"
            )
        else:
            await telegram_bot.send_message("🌅 *Premarket terminado* - Sin posiciones abiertas")

        logger.info("Premarket session ended")

    async def _start_postmarket_monitoring(self):
        """Start postmarket earnings/news monitoring"""
        logger.info("Starting postmarket monitoring (Earnings/News)...")
        await telegram_bot.send_message(
            "🌙 *POSTMARKET SESSION*\n"
            f"Modo: Earnings & News\n"
            f"Horario: {settings.extended_hours.postmarket_trade_start} - {settings.extended_hours.postmarket_trade_end} EST\n"
            f"Tamaño posición: {settings.extended_hours.postmarket_position_size_mult*100:.0f}%\n"
            f"Max trades: {settings.extended_hours.postmarket_max_trades}\n"
            f"Cierre forzado: {settings.extended_hours.postmarket_force_close} EST"
        )
        self.postmarket_monitoring_task = asyncio.create_task(self._monitor_postmarket())

    async def _monitor_postmarket(self):
        """Monitor for postmarket earnings/news reactions"""
        logger.info("Monitoring postmarket for earnings/news reactions...")

        while self.is_running:
            try:
                now = datetime.now(EST)

                # Check if still in postmarket trading window
                if not is_in_trading_window(TradingMode.POSTMARKET):
                    logger.info("Postmarket trading window closed")
                    break

                # Check watchlist symbols for earnings reactions
                for symbol in self.watchlist:
                    signal = await self._check_postmarket_symbol(symbol)
                    if signal:
                        await self._send_extended_hours_alert(signal)

                await asyncio.sleep(30)  # 30 second intervals for postmarket

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in postmarket monitoring: {e}")
                await asyncio.sleep(10)

    async def _check_postmarket_symbol(self, symbol: str) -> Optional[ExtendedHoursSignal]:
        """Check a symbol for postmarket trading opportunity"""
        try:
            quote = market_data.get_latest_quote(symbol)
            if not quote:
                return None

            current_price = quote['mid']
            prev_close = market_data.get_previous_close(symbol)

            if not prev_close:
                return None

            # Check for significant move (earnings reaction)
            move_pct = abs((current_price - prev_close) / prev_close) * 100
            if move_pct >= settings.extended_hours.postmarket_min_move_pct:
                signal = postmarket_strategy.check_earnings_play(
                    symbol=symbol,
                    current_price=current_price,
                    prev_close=prev_close,
                    catalyst_type="earnings"  # Could enhance with actual catalyst detection
                )
                return signal

            return None

        except Exception as e:
            logger.error(f"Error checking postmarket {symbol}: {e}")
            return None

    async def _end_postmarket_session(self):
        """End postmarket trading session"""
        logger.info("Ending postmarket session...")

        if self.postmarket_monitoring_task:
            self.postmarket_monitoring_task.cancel()

        positions = order_executor.get_positions()
        if positions:
            await telegram_bot.send_message(
                f"🌙 *Fin trading postmarket*\n"
                f"{len(positions)} posicion(es) abiertas\n"
                f"⚠️ Cierre forzado a las {settings.extended_hours.postmarket_force_close} EST"
            )
        else:
            await telegram_bot.send_message("🌙 *Postmarket terminado* - Sin posiciones abiertas")

        logger.info("Postmarket trading session ended")

    async def _force_close_extended_positions(self):
        """Force close all positions at extended hours deadline"""
        logger.info("Force closing extended hours positions...")

        positions = order_executor.get_positions()
        if positions:
            total_pnl = sum(p.unrealized_pl for p in positions)
            await telegram_bot.send_message(
                f"⏰ *CIERRE FORZADO*\n"
                f"Cerrando {len(positions)} posicion(es)\n"
                f"P/L antes de cierre: ${total_pnl:.2f}"
            )
            order_executor.close_all_positions()
            await telegram_bot.send_message("✅ *Todas las posiciones cerradas*")
            logger.info(f"Force closed {len(positions)} positions, P/L: ${total_pnl:.2f}")
        else:
            logger.info("No positions to force close")

    async def _send_extended_hours_alert(self, signal: ExtendedHoursSignal):
        """Send Telegram alert for extended hours signal"""
        session_emoji = "🌅" if signal.session == 'premarket' else "🌙"

        message = (
            f"{session_emoji} *{signal.session.upper()} SIGNAL*\n\n"
            f"{signal}\n\n"
            f"⚠️ *IMPORTANTE - Extended Hours:*\n"
            f"• Solo órdenes límite permitidas\n"
            f"• ⛔ Stop loss NO activo hasta apertura regular\n"
            f"• Monitoreo manual de precio recomendado\n\n"
            f"Responder SI/NO para ejecutar"
        )

        await telegram_bot.send_message(message)
        # Note: Would need to add extended hours confirmation handling to telegram_bot
        logger.info(f"Extended hours alert sent: {signal.signal_type.value} {signal.symbol}")

    async def _on_extended_trade_confirmed(self, signal: ExtendedHoursSignal):
        """Handle confirmed extended hours trade"""
        logger.info(f"Extended hours trade confirmed: {signal.signal_type.value} {signal.symbol}")

        side = 'buy' if 'LONG' in signal.signal_type.value else 'sell'

        # Execute extended hours order (limit only)
        result = order_executor.execute_extended_hours_order(
            symbol=signal.symbol,
            qty=signal.position_size,
            side=side,
            limit_price=signal.limit_price,
            stop_price=signal.stop_loss
        )

        if result.success:
            await telegram_bot.send_message(
                f"✅ *Orden ejecutada (extended hours)*\n"
                f"{signal.symbol} {side.upper()} {signal.position_size} @ ${result.filled_price:.2f}\n\n"
                f"⚠️ *ALERTA CRÍTICA*\n"
                f"Stop loss @ ${signal.stop_loss:.2f} NO se activará hasta\n"
                f"la apertura del mercado regular (9:30 EST).\n"
                f"Monitorea el precio manualmente para gestionar riesgo."
            )
        else:
            await telegram_bot.send_message(f"❌ Error: {result.error}")

    # ========== End Extended Hours Methods ==========

    async def _check_immediate_start(self):
        """Check if we're already in trading window and start immediately with adaptive logic"""
        now = datetime.now(EST)
        current_time = now.time()

        # Key times (EST) - updated for 5-min ORB
        premarket_scan = time(9, 25)
        market_open = time(9, 30)
        orb_ready = time(9, 35)      # ORB ready after 5 min
        monitor_start = time(9, 36)  # Start monitoring
        session_end = time(16, 0)

        # Check if market is open today
        clock_info = order_executor.get_next_market_times()
        is_market_open = clock_info.get('is_open', False)

        if not is_market_open:
            await telegram_bot.send_message("🤖 *Bot iniciado*\n🔴 Mercado cerrado\nEsperando próxima apertura...")
            logger.info("Market is closed, waiting for scheduled times")
            return

        # Build universe if not already built (bot started after 4 AM reset)
        if not universe_manager.is_cache_valid:
            logger.info("Building universe on startup...")
            try:
                # Pass telegram_bot.send_message as progress callback for real-time updates
                await universe_manager.build_universe(
                    notify_callback=telegram_bot.send_message
                )
            except Exception as e:
                logger.error(f"Error building universe on startup: {e}")
                await telegram_bot.send_message(
                    "⚠️ Error construyendo universo dinámico, usando fallback"
                )

        # Market is open, check where we are in the trading window
        if current_time >= session_end:
            await telegram_bot.send_message("🤖 *Bot iniciado*\n⏰ Ventana de trading cerrada (después de 16:00 EST)\nEsperando mañana...")
            logger.info("Trading window closed for today")
            return

        if current_time >= monitor_start:
            # After 9:36 - run full startup sequence immediately
            await telegram_bot.send_message(
                "🤖 *Bot iniciado*\n"
                "🟢 *Mercado ABIERTO*\n"
                "⚡ Iniciando secuencia completa..."
            )
            logger.info("Market open - running immediate startup sequence")

            # Run premarket scan
            await self._run_premarket_scan()

            # Calculate ORBs (data available)
            await self._calculate_opening_ranges()

            # Start monitoring
            await self._start_monitoring()

        elif current_time >= orb_ready:
            # Between 9:35 and 9:36 - scan + ORB, then wait for monitor schedule
            await telegram_bot.send_message(
                "🤖 *Bot iniciado*\n"
                "🟢 *Mercado ABIERTO*\n"
                "📊 Ejecutando scanner y calculando ORB..."
            )
            logger.info("Market open - running scan and ORB calculation")
            await self._run_premarket_scan()
            await self._calculate_opening_ranges()
            # Monitoring will start via scheduler at 9:36

        elif current_time >= premarket_scan:
            # Between 9:25 and 9:35 - run scan, wait for ORB data
            minutes_to_orb = (orb_ready.hour * 60 + orb_ready.minute) - (current_time.hour * 60 + current_time.minute)
            await telegram_bot.send_message(
                "🤖 *Bot iniciado*\n"
                "🟢 *Mercado ABIERTO*\n"
                f"📊 Ejecutando scanner...\n"
                f"⏳ ORB en {minutes_to_orb} min (9:35 EST)"
            )
            logger.info(f"Market open - running scan, ORB in {minutes_to_orb} min")
            await self._run_premarket_scan()

        else:
            # Before 9:25 AM - just wait for scheduled tasks
            await telegram_bot.send_message(
                "🤖 *Bot iniciado*\n"
                "🟡 Mercado abrirá pronto\n"
                f"Scanner a las 9:25 AM EST"
            )
            logger.info("Waiting for market open")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Alpaca ORB Trading Bot - Day trading with extended hours support"
    )
    parser.add_argument(
        '--no-cli',
        action='store_true',
        help='Skip interactive CLI and start directly (uses -m mode or default)'
    )
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['regular', 'premarket', 'postmarket', 'all'],
        default=None,
        help='Trading mode: regular, premarket, postmarket, or all (requires --no-cli)'
    )
    parser.add_argument(
        '-l', '--level',
        type=str,
        choices=['STRICT', 'MODERATE', 'RELAXED'],
        default=None,
        help='Signal sensitivity level'
    )
    return parser.parse_args()


async def run_interactive_mode():
    """Run the interactive CLI for mode selection"""
    try:
        from cli.app import CLIApp
        app = CLIApp()
        trading_mode = await app.run()
        return trading_mode
    except ImportError:
        logger.warning("CLI module not available, falling back to regular mode")
        return TradingMode.REGULAR


async def main():
    """Main entry point"""
    args = parse_args()

    # Determine trading mode
    trading_mode = None

    if args.no_cli:
        # Skip CLI, use command line mode or default
        if args.mode:
            mode_map = {
                'regular': TradingMode.REGULAR,
                'premarket': TradingMode.PREMARKET,
                'postmarket': TradingMode.POSTMARKET,
                'all': TradingMode.ALL_SESSIONS
            }
            trading_mode = mode_map.get(args.mode, TradingMode.REGULAR)
        else:
            trading_mode = settings.trading.trading_mode
    else:
        # Default: Run interactive CLI
        trading_mode = await run_interactive_mode()
        if trading_mode is None:
            logger.info("User exited CLI, shutting down")
            return

    # Set signal level if specified
    if args.level:
        orb_strategy.set_signal_level(args.level)

    # Create bot with trading mode
    bot = TradingBot(trading_mode=trading_mode)

    # Handle graceful shutdown - cross-platform
    if sys.platform != 'win32':
        # Unix: Use loop signal handlers (works correctly with asyncio)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(bot.stop())
            )
        await bot.start()
    else:
        # Windows: Use try/except for KeyboardInterrupt (Ctrl+C)
        # Signal handlers on Windows don't work well with asyncio
        try:
            await bot.start()
        except KeyboardInterrupt:
            logger.info("Shutdown signal received (KeyboardInterrupt)")
            await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
