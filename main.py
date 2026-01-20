"""
Alpaca ORB Trading Bot - Main Entry Point

Semi-autonomous day trading bot using Opening Range Breakout strategy
with VWAP and Volume confirmation. Sends signals via Telegram for confirmation.
"""
import asyncio
import signal
import sys
from datetime import datetime, time
from typing import Optional
import pytz
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import settings
from data.market_data import market_data
from scanner.premarket import premarket_scanner, SCAN_UNIVERSE
from strategy.orb import orb_strategy, TradeSignal
from execution.orders import order_executor
from notifications.telegram_bot import telegram_bot


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


class TradingBot:
    """Main trading bot orchestrator"""

    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone=EST)
        self.is_running = False
        self.watchlist: list[str] = []
        self.trades_today: list[dict] = []
        self.monitoring_task: Optional[asyncio.Task] = None

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

        # Schedule daily tasks
        self._schedule_tasks()

        # Start scheduler
        self.scheduler.start()
        self.is_running = True

        logger.info("Bot started successfully")
        logger.info(f"Paper Trading: {settings.alpaca.paper}")
        logger.info(f"Max Capital: ${settings.trading.max_capital:,}")
        logger.info(f"Risk per trade: {settings.trading.risk_per_trade * 100}%")

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

        self.scheduler.shutdown()

        if telegram_bot.is_running:
            await telegram_bot.send_message("🛑 *Bot detenido*")
            await telegram_bot.stop_polling()

        logger.info("Bot stopped")

    def _schedule_tasks(self):
        """Schedule daily trading tasks"""
        # Pre-market scan at 9:25 AM EST
        self.scheduler.add_job(
            self._run_premarket_scan,
            CronTrigger(hour=9, minute=25, timezone=EST),
            id='premarket_scan',
            replace_existing=True
        )

        # Calculate Opening Range at 9:45 AM EST
        self.scheduler.add_job(
            self._calculate_opening_ranges,
            CronTrigger(hour=9, minute=45, timezone=EST),
            id='calculate_orb',
            replace_existing=True
        )

        # Start monitoring at 9:46 AM EST
        self.scheduler.add_job(
            self._start_monitoring,
            CronTrigger(hour=9, minute=46, timezone=EST),
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

        # Reset daily at 6:00 AM EST
        self.scheduler.add_job(
            self._reset_daily,
            CronTrigger(hour=6, minute=0, timezone=EST),
            id='reset_daily',
            replace_existing=True
        )

        logger.info("Scheduled tasks configured")

    async def _run_premarket_scan(self):
        """Run pre-market scanner with sentiment analysis"""
        logger.info("Running pre-market scan with sentiment analysis...")

        try:
            # Scan for candidates with sentiment analysis
            candidates = await premarket_scanner.scan_watchlist_with_sentiment(SCAN_UNIVERSE)

            if not candidates:
                logger.warning("No candidates found in pre-market scan")
                await telegram_bot.send_message("⚠️ No se encontraron candidatos hoy")
                return

            # Update watchlist
            self.watchlist = [c.symbol for c in candidates]

            # Update strategy sentiment cache
            for c in candidates:
                orb_strategy.update_sentiment(c.symbol, c.sentiment_score)

            # Send watchlist to Telegram
            message = premarket_scanner.format_watchlist_message()
            await telegram_bot.send_watchlist(message)

            logger.info(f"Watchlist: {self.watchlist}")

        except Exception as e:
            logger.error(f"Error in pre-market scan: {e}")
            await telegram_bot.send_message(f"❌ Error en scanner: {e}")

    async def _calculate_opening_ranges(self):
        """Calculate Opening Range for watchlist symbols"""
        logger.info("Calculating Opening Ranges...")

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

    async def _monitor_breakouts(self):
        """Monitor watchlist for breakout signals"""
        logger.info("Monitoring for breakouts...")

        while self.is_running and not telegram_bot.is_paused:
            try:
                now = datetime.now(EST)

                # Stop monitoring after 16:00 (market close)
                if now.time() >= time(16, 0):
                    logger.info("Trading window closed")
                    break

                # Check each symbol in watchlist
                for symbol in self.watchlist:
                    await self._check_symbol(symbol)

                # Wait before next check
                await asyncio.sleep(10)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _check_symbol(self, symbol: str):
        """Check a symbol for breakout signal"""
        try:
            # Get current quote
            quote = market_data.get_latest_quote(symbol)
            if not quote:
                return

            current_price = quote['mid']

            # Get average volume for relative volume calculation
            avg_volume = market_data.get_avg_daily_volume(symbol)

            # Get recent bars for current volume
            bars = market_data.get_bars(symbol, limit=5)
            if bars.empty:
                return

            current_volume = int(bars['volume'].iloc[-1])

            # Check for breakout
            signal = orb_strategy.check_breakout(
                symbol=symbol,
                current_price=current_price,
                current_volume=current_volume,
                avg_volume=avg_volume
            )

            if signal:
                logger.info(f"Signal detected for {symbol}")
                await telegram_bot.send_signal_alert(signal)

        except Exception as e:
            logger.error(f"Error checking {symbol}: {e}")

    async def _on_trade_confirmed(self, signal: TradeSignal):
        """Handle confirmed trade from Telegram"""
        logger.info(f"Trade confirmed: {signal.signal_type.value} {signal.symbol}")

        result = order_executor.execute_signal(signal)

        if result.success:
            await telegram_bot.send_execution_confirmation(
                symbol=signal.symbol,
                side=signal.signal_type.value,
                qty=signal.position_size,
                price=signal.entry_price
            )

            self.trades_today.append({
                'symbol': signal.symbol,
                'side': signal.signal_type.value,
                'entry': signal.entry_price,
                'stop': signal.stop_loss,
                'target': signal.take_profit,
                'qty': signal.position_size,
                'time': datetime.now(EST)
            })
        else:
            await telegram_bot.send_message(f"❌ Error ejecutando orden: {result.error}")

    async def _end_session(self):
        """End trading session"""
        logger.info("Ending trading session...")

        # Stop monitoring
        if self.monitoring_task:
            self.monitoring_task.cancel()

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
        """Reset daily data"""
        logger.info("Resetting daily data...")
        orb_strategy.reset_daily()
        self.watchlist.clear()
        self.trades_today.clear()
        logger.info("Daily reset complete")

    async def _check_immediate_start(self):
        """Check if we're already in trading window and start immediately"""
        now = datetime.now(EST)
        current_time = now.time()

        # Trading window: 9:30 AM - 16:00 EST (market close)
        market_open = time(9, 30)
        premarket_scan = time(9, 25)
        orb_ready = time(9, 45)
        monitor_start = time(9, 46)
        session_end = time(16, 0)

        # Check if market is open today
        clock_info = order_executor.get_next_market_times()
        is_market_open = clock_info.get('is_open', False)

        if not is_market_open:
            await telegram_bot.send_message("🤖 *Bot iniciado*\n🔴 Mercado cerrado\nEsperando próxima apertura...")
            logger.info("Market is closed, waiting for scheduled times")
            return

        # Market is open, check where we are in the trading window
        if current_time >= session_end:
            await telegram_bot.send_message("🤖 *Bot iniciado*\n⏰ Ventana de trading cerrada (después de 16:00 EST)\nEsperando mañana...")
            logger.info("Trading window closed for today")
            return

        if current_time >= monitor_start:
            # We're in the monitoring window - run full startup sequence
            await telegram_bot.send_message(
                "🤖 *Bot iniciado*\n"
                "🟢 *Mercado ABIERTO*\n"
                "⚡ Iniciando secuencia completa..."
            )
            logger.info("Market open - running immediate startup sequence")

            # Run premarket scan
            await self._run_premarket_scan()

            # Calculate ORBs
            await self._calculate_opening_ranges()

            # Start monitoring
            await self._start_monitoring()

        elif current_time >= premarket_scan:
            # Between 9:25 and 9:46 - run scan, will catch up to schedule
            await telegram_bot.send_message(
                "🤖 *Bot iniciado*\n"
                "🟢 *Mercado ABIERTO*\n"
                "📊 Ejecutando scanner..."
            )
            logger.info("Market open - running premarket scan")
            await self._run_premarket_scan()

        else:
            # Before 9:25 AM - just wait for scheduled tasks
            await telegram_bot.send_message(
                "🤖 *Bot iniciado*\n"
                "🟡 Mercado abrirá pronto\n"
                f"Scanner a las 9:25 AM EST"
            )
            logger.info("Waiting for market open")


async def main():
    """Main entry point"""
    bot = TradingBot()

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        asyncio.create_task(bot.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
