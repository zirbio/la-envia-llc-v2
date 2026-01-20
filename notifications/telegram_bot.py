"""
Telegram bot for notifications and trade confirmations
"""
import asyncio
from typing import Optional, Callable, Awaitable
from telegram import Update, Bot
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
from loguru import logger

from config.settings import settings
from strategy.orb import TradeSignal, SignalType
from execution.orders import order_executor


class TradingTelegramBot:
    """Telegram bot for trading notifications and confirmations"""

    def __init__(self):
        self.token = settings.telegram.bot_token
        self.chat_id = settings.telegram.chat_id
        self.bot: Optional[Bot] = None
        self.app: Optional[Application] = None
        self.pending_signals: dict[str, TradeSignal] = {}
        self.is_running = False
        self.is_paused = False

        # Callbacks for external handlers
        self.on_confirmation: Optional[Callable[[TradeSignal], Awaitable[None]]] = None

    async def initialize(self):
        """Initialize the bot"""
        if not self.token:
            logger.warning("Telegram bot token not configured")
            return False

        try:
            self.bot = Bot(token=self.token)
            self.app = Application.builder().token(self.token).build()

            # Add command handlers
            self.app.add_handler(CommandHandler("start", self._cmd_start))
            self.app.add_handler(CommandHandler("stop", self._cmd_stop))
            self.app.add_handler(CommandHandler("status", self._cmd_status))
            self.app.add_handler(CommandHandler("positions", self._cmd_positions))
            self.app.add_handler(CommandHandler("watchlist", self._cmd_watchlist))
            self.app.add_handler(CommandHandler("close", self._cmd_close_all))
            self.app.add_handler(CommandHandler("help", self._cmd_help))

            # Handle text messages (for confirmations)
            self.app.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
            )

            logger.info("Telegram bot initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            return False

    async def start_polling(self):
        """Start the bot polling"""
        if self.app:
            self.is_running = True
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            logger.info("Telegram bot started polling")

    async def stop_polling(self):
        """Stop the bot polling"""
        if self.app:
            self.is_running = False
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram bot stopped")

    async def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a message to the configured chat

        Args:
            text: Message text (supports Markdown)
            parse_mode: Parse mode for formatting

        Returns:
            True if sent successfully
        """
        if not self.bot or not self.chat_id:
            logger.warning("Bot not configured, cannot send message")
            return False

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def send_signal_alert(self, signal: TradeSignal) -> bool:
        """
        Send a trade signal alert and wait for confirmation

        Args:
            signal: TradeSignal to alert

        Returns:
            True if sent successfully
        """
        # Store pending signal
        self.pending_signals[signal.symbol] = signal

        emoji = "🟢" if signal.signal_type == SignalType.LONG else "🔴"
        direction = signal.signal_type.value

        risk_pct = (signal.risk_amount / settings.trading.max_capital) * 100
        reward = signal.risk_amount * settings.trading.reward_risk_ratio

        message = f"""
{emoji} *SIGNAL: {direction} {signal.symbol}*

*Entry:* ${signal.entry_price:.2f}
*ORB High:* ${signal.orb_high:.2f} {'✓' if signal.signal_type == SignalType.LONG else ''}
*ORB Low:* ${signal.orb_low:.2f} {'✓' if signal.signal_type == SignalType.SHORT else ''}
*VWAP:* ${signal.vwap:.2f} ✓
*Volume:* {signal.relative_volume:.1f}x ✓
*RSI:* {signal.rsi:.0f}

━━━━━━━━━━━━━━━━━━━━
*Stop Loss:* ${signal.stop_loss:.2f}
*Take Profit:* ${signal.take_profit:.2f}
*Position:* {signal.position_size} shares (${signal.position_size * signal.entry_price:,.0f})
*Risk:* ${signal.risk_amount:.2f} ({risk_pct:.1f}%)
*Reward:* ${reward:.2f} (2:1)
━━━━━━━━━━━━━━━━━━━━

Reply *SI* to execute or *NO* to skip
        """

        return await self.send_message(message.strip())

    async def send_watchlist(self, watchlist_message: str) -> bool:
        """Send watchlist alert"""
        return await self.send_message(watchlist_message)

    async def send_execution_confirmation(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float
    ) -> bool:
        """Send order execution confirmation"""
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        action = "Comprado" if side.upper() == "BUY" else "Vendido"

        message = f"""
✅ *Orden Ejecutada*

{emoji} {action} {qty} {symbol} @ ${price:.2f}
Total: ${qty * price:,.2f}
        """
        return await self.send_message(message.strip())

    async def send_position_update(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float
    ) -> bool:
        """Send position P/L update"""
        emoji = "📈" if pnl >= 0 else "📉"
        direction = "+" if pnl >= 0 else ""

        message = f"{emoji} *{symbol}* {direction}${pnl:.2f} ({direction}{pnl_pct:.1f}%)"
        return await self.send_message(message)

    async def send_trade_closed(
        self,
        symbol: str,
        pnl: float,
        reason: str
    ) -> bool:
        """Send trade closed notification"""
        emoji = "🎯" if pnl > 0 else "🛑" if pnl < 0 else "➖"
        direction = "+" if pnl >= 0 else ""

        if pnl > 0:
            status = "TARGET alcanzado!"
        elif pnl < 0:
            status = "STOP LOSS activado"
        else:
            status = "Posición cerrada"

        message = f"""
{emoji} *{status}*

{symbol}: {direction}${pnl:.2f}
Razón: {reason}
        """
        return await self.send_message(message.strip())

    async def send_daily_summary(
        self,
        trades: int,
        winners: int,
        total_pnl: float
    ) -> bool:
        """Send end of day summary"""
        win_rate = (winners / trades * 100) if trades > 0 else 0
        emoji = "🟢" if total_pnl >= 0 else "🔴"
        direction = "+" if total_pnl >= 0 else ""

        message = f"""
📊 *RESUMEN DEL DÍA*

Trades: {trades}
Ganados: {winners}
Win Rate: {win_rate:.0f}%

{emoji} P/L: {direction}${total_pnl:.2f}
        """
        return await self.send_message(message.strip())

    async def _handle_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle incoming text messages"""
        if not update.message or not update.message.text:
            return

        text = update.message.text.strip().upper()
        chat_id = str(update.message.chat_id)

        # Verify it's from the correct chat
        if chat_id != self.chat_id:
            return

        # Handle confirmation responses
        if text in ["SI", "SÍ", "YES", "Y", "OK"]:
            await self._handle_confirmation(True)
        elif text in ["NO", "N", "CANCEL"]:
            await self._handle_confirmation(False)

    async def _handle_confirmation(self, confirmed: bool):
        """Handle trade confirmation response"""
        if not self.pending_signals:
            await self.send_message("No hay señales pendientes")
            return

        # Get the most recent pending signal
        symbol = list(self.pending_signals.keys())[-1]
        signal = self.pending_signals.pop(symbol)

        if confirmed:
            await self.send_message(f"⏳ Ejecutando orden para {symbol}...")

            if self.on_confirmation:
                await self.on_confirmation(signal)
            else:
                # Execute directly if no callback
                result = order_executor.execute_signal(signal)
                if result.success:
                    await self.send_execution_confirmation(
                        symbol, signal.signal_type.value,
                        signal.position_size, signal.entry_price
                    )
                else:
                    await self.send_message(f"❌ Error: {result.error}")
        else:
            await self.send_message(f"⏭️ Señal ignorada: {symbol}")

    async def _cmd_start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /start command"""
        self.is_paused = False
        await update.message.reply_text("🤖 Bot activado y monitoreando")

    async def _cmd_stop(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /stop command"""
        self.is_paused = True
        await update.message.reply_text("⏸️ Bot pausado")

    async def _cmd_status(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /status command"""
        account = order_executor.get_account()
        status = "🟢 Activo" if not self.is_paused else "⏸️ Pausado"
        market = "🟢 Abierto" if order_executor.is_market_open() else "🔴 Cerrado"

        message = f"""
*Estado del Bot*

{status}
Mercado: {market}

*Cuenta:*
Equity: ${account.get('equity', 0):,.2f}
Cash: ${account.get('cash', 0):,.2f}
Buying Power: ${account.get('buying_power', 0):,.2f}
        """
        await update.message.reply_text(message.strip(), parse_mode="Markdown")

    async def _cmd_positions(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /positions command"""
        message = order_executor.format_positions_message()
        await update.message.reply_text(message, parse_mode="Markdown")

    async def _cmd_watchlist(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /watchlist command"""
        # This will be populated by the main loop
        await update.message.reply_text(
            "Usa este comando durante la sesión pre-market para ver la watchlist"
        )

    async def _cmd_close_all(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /close command"""
        await update.message.reply_text("⏳ Cerrando todas las posiciones...")
        results = order_executor.close_all_positions()
        await update.message.reply_text("✅ Todas las posiciones cerradas")

    async def _cmd_help(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /help command"""
        message = """
*Comandos disponibles:*

/status - Ver estado del bot y cuenta
/positions - Ver posiciones abiertas
/watchlist - Ver watchlist del día
/start - Reanudar el bot
/stop - Pausar el bot
/close - Cerrar todas las posiciones
/help - Ver esta ayuda

*Respuestas a señales:*
SI / YES - Ejecutar trade
NO - Ignorar señal
        """
        await update.message.reply_text(message.strip(), parse_mode="Markdown")


# Global bot instance
telegram_bot = TradingTelegramBot()
