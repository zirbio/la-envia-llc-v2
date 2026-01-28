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

from config.settings import settings, SignalLevel
from strategy.orb import TradeSignal, SignalType, orb_strategy
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

        # Session close confirmation
        self.pending_session_close = False
        self.session_close_event: Optional[asyncio.Event] = None
        self.session_close_confirmed = False

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
            self.app.add_handler(CommandHandler("level", self._cmd_level))
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

        # Get quality level badge
        quality_badge = self._get_quality_badge(signal.quality_level)
        current_level = settings.trading.signal_level.value
        min_score = settings.trading.min_signal_score

        # Calculate individual score components for breakdown
        score_breakdown = self._calculate_score_breakdown(signal)

        # Build breakdown section
        breakdown_lines = []
        for component, (points, max_pts, detail) in score_breakdown.items():
            if points >= max_pts * 0.6:  # Good score
                marker = "✓"
            elif points >= max_pts * 0.3:  # Partial score
                marker = "○"
            else:  # Low score
                marker = "✗"
            breakdown_lines.append(f"{marker} {component}: {points:.0f}/{max_pts} pts {detail}")

        breakdown_text = "\n".join(breakdown_lines)

        # Warning if signal doesn't meet current threshold
        warning_text = ""
        threshold_quality = self._get_threshold_quality_name(min_score)
        if signal.signal_score < min_score:
            warning_text = f"\n⚠️ *Calidad {signal.quality_level} - no alcanza umbral {threshold_quality} ({min_score:.0f} pts)*\n"

        message = f"""
{emoji} *SEÑAL: {direction} {signal.symbol}*

{quality_badge} Score: {signal.signal_score:.0f}/100 → *{signal.quality_level}*
⚙️ Umbral actual: {min_score:.0f} pts ({threshold_quality})

*Entry:* ${signal.entry_price:.2f}
*Stop Loss:* ${signal.stop_loss:.2f}
*Take Profit:* ${signal.take_profit:.2f}
*Position:* {signal.position_size} shares

━━━ Desglose ━━━
{breakdown_text}
━━━━━━━━━━━━━━━━━━━━
*Risk:* ${signal.risk_amount:.2f} ({risk_pct:.1f}%)
*Reward:* ${reward:.2f} (2:1)
{warning_text}
Reply *SI* para ejecutar, *NO* para skip
        """

        return await self.send_message(message.strip())

    async def send_auto_execution_notification(self, signal: TradeSignal, fill_price: float) -> bool:
        """
        Send informative notification for automatic execution (no confirmation buttons)

        Args:
            signal: TradeSignal that was executed
            fill_price: Actual fill price from the order

        Returns:
            True if sent successfully
        """
        is_long = signal.signal_type == SignalType.LONG
        emoji = "🟢" if is_long else "🔴"
        direction = signal.signal_type.value

        risk_pct = (signal.risk_amount / settings.trading.max_capital) * 100
        reward = signal.risk_amount * settings.trading.reward_risk_ratio

        # Get quality level badge
        quality_badge = self._get_quality_badge(signal.quality_level)

        message = f"""
{emoji} *EJECUTADO: {direction} {signal.symbol}*

{quality_badge} Score: {signal.signal_score:.0f}/100 → *{signal.quality_level}*

*Entry:* ${fill_price:.2f}
*Stop Loss:* ${signal.stop_loss:.2f}
*Take Profit:* ${signal.take_profit:.2f}
*Position:* {signal.position_size} shares

*Risk:* ${signal.risk_amount:.2f} ({risk_pct:.1f}%)
*Reward:* ${reward:.2f} (2:1)

📊 *Gestión de posición activada*
        """
        return await self.send_message(message.strip())

    def _get_quality_badge(self, quality_level: str) -> str:
        """Get emoji badge for quality level"""
        badges = {
            'ÓPTIMA': '🟢',
            'BUENA': '🟡',
            'REGULAR': '🟠',
            'DÉBIL': '🔴'
        }
        return badges.get(quality_level, '⚪')

    def _get_threshold_quality_name(self, min_score: float) -> str:
        """Get Spanish quality name for threshold score"""
        if min_score >= 70:
            return 'ÓPTIMA'
        elif min_score >= 55:
            return 'BUENA'
        elif min_score >= 40:
            return 'REGULAR'
        else:
            return 'DÉBIL'

    def _calculate_score_breakdown(self, signal: TradeSignal) -> dict:
        """
        Calculate individual score components for display.

        Returns dict of component -> (points, max_points, detail_string)
        """
        breakdown = {}
        direction = signal.signal_type.value

        # BREAKOUT (0-25 pts)
        # Approximation based on signal having been triggered
        if direction == 'LONG':
            breakout_pct = ((signal.entry_price - signal.orb_high) / signal.orb_high) * 100
        else:
            breakout_pct = ((signal.orb_low - signal.entry_price) / signal.orb_low) * 100
        breakout_pts = min(breakout_pct * 50, 25) if breakout_pct > 0 else 0
        breakdown['Breakout'] = (breakout_pts, 25, f"({breakout_pct:.2f}%)")

        # VWAP (0-15 pts)
        if signal.vwap > 0:
            vwap_dist_pct = abs(signal.entry_price - signal.vwap) / signal.vwap * 100
            vwap_aligned = (direction == 'LONG' and signal.entry_price > signal.vwap) or \
                          (direction == 'SHORT' and signal.entry_price < signal.vwap)
            vwap_pts = min(vwap_dist_pct * 15, 15) if vwap_aligned else 0
        else:
            vwap_pts = 0
            vwap_dist_pct = 0
        breakdown['VWAP'] = (vwap_pts, 15, f"({'✓' if vwap_pts > 0 else '✗'})")

        # VOLUME (0-20 pts)
        vol = signal.relative_volume
        if vol >= 2.5:
            vol_pts = 20
        elif vol >= 2.0:
            vol_pts = 15
        elif vol >= 1.5:
            vol_pts = 10
        elif vol >= 1.2:
            vol_pts = 5
        else:
            vol_pts = 0
        breakdown['Volume'] = (vol_pts, 20, f"({vol:.1f}x)")

        # RSI (0-15 pts)
        rsi = signal.rsi
        if direction == 'LONG':
            if 40 <= rsi <= 60:
                rsi_pts = 15
            elif 30 <= rsi <= 70:
                rsi_pts = 10
            elif rsi < 30:
                rsi_pts = 5
            else:
                rsi_pts = 0
        else:
            if 40 <= rsi <= 60:
                rsi_pts = 15
            elif 30 <= rsi <= 70:
                rsi_pts = 10
            elif rsi > 70:
                rsi_pts = 5
            else:
                rsi_pts = 0
        breakdown['RSI'] = (rsi_pts, 15, f"({rsi:.0f})")

        # MACD (0-15 pts)
        hist = signal.macd_histogram
        if (direction == 'LONG' and hist > 0) or (direction == 'SHORT' and hist < 0):
            macd_pts = min(abs(hist) * 100, 15)
        else:
            macd_pts = 0
        macd_status = "✓" if macd_pts > 7 else ("○" if macd_pts > 0 else "✗")
        breakdown['MACD'] = (macd_pts, 15, f"({macd_status})")

        # SENTIMENT (0-10 pts)
        sent = signal.sentiment_score
        if direction == 'LONG':
            sent_pts = max(0, min((sent + 1) * 5, 10))
        else:
            sent_pts = max(0, min((1 - sent) * 5, 10))
        breakdown['Sentiment'] = (sent_pts, 10, f"({sent:.2f})")

        return breakdown

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
        # Handle both BUY/SELL and LONG/SHORT nomenclature
        is_long = side.upper() in ("BUY", "LONG")
        emoji = "🟢" if is_long else "🔴"
        action = "Comprado" if is_long else "Vendido"

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

    async def send_partial_close_alert(
        self,
        symbol: str,
        closed_qty: int,
        close_price: float,
        remaining_qty: int,
        new_stop: float,
        realized_pnl: float,
        r_multiple: float
    ) -> bool:
        """
        Send notification when partial position is closed at 1R

        Args:
            symbol: Stock symbol
            closed_qty: Shares closed
            close_price: Price at which shares were closed
            remaining_qty: Remaining shares
            new_stop: New stop loss price (breakeven)
            realized_pnl: Realized P/L from partial close
            r_multiple: Current R-multiple

        Returns:
            True if sent successfully
        """
        pnl_emoji = "🟢" if realized_pnl >= 0 else "🔴"
        direction = "+" if realized_pnl >= 0 else ""

        message = f"""
🎯 *CIERRE PARCIAL - 1R ALCANZADO*

*{symbol}*

━━━━━━━━━━━━━━━━━━━━
Cerrado: {closed_qty} acciones @ ${close_price:.2f}
Restante: {remaining_qty} acciones
R-Multiple: {r_multiple:.1f}R
{pnl_emoji} P/L Realizado: {direction}${realized_pnl:.2f}
━━━━━━━━━━━━━━━━━━━━

*Stop movido a breakeven:* ${new_stop:.2f}
*Trailing EMA9 activado*
        """
        return await self.send_message(message.strip())

    async def send_trailing_stop_update(
        self,
        symbol: str,
        old_stop: float,
        new_stop: float,
        ema9_value: float
    ) -> bool:
        """
        Send notification when trailing stop is updated

        Args:
            symbol: Stock symbol
            old_stop: Previous stop price
            new_stop: New stop price
            ema9_value: Current EMA9 value

        Returns:
            True if sent successfully
        """
        # Determine direction
        if new_stop > old_stop:
            arrow = "⬆️"
            direction = f"+${new_stop - old_stop:.2f}"
        else:
            arrow = "⬇️"
            direction = f"-${old_stop - new_stop:.2f}"

        message = f"""
🔄 *TRAILING STOP ACTUALIZADO*

*{symbol}*

{arrow} Stop: ${old_stop:.2f} → ${new_stop:.2f} ({direction})
EMA9 (5min): ${ema9_value:.2f}
        """
        return await self.send_message(message.strip())

    async def send_trailing_activated(
        self,
        symbol: str,
        ema_period: int,
        timeframe: str
    ) -> bool:
        """
        Send notification when trailing stop is activated

        Args:
            symbol: Stock symbol
            ema_period: EMA period (e.g., 9)
            timeframe: Bar timeframe (e.g., "5min")

        Returns:
            True if sent successfully
        """
        message = f"""
🎚️ *TRAILING STOP ACTIVADO*

*{symbol}*

Trailing con EMA{ema_period} en barras de {timeframe}
El stop se ajustara automaticamente siguiendo el EMA
        """
        return await self.send_message(message.strip())

    async def send_position_stopped(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        qty: int,
        reason: str
    ) -> bool:
        """
        Send notification when position is stopped out

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            stop_price: Stop price that was hit
            qty: Quantity closed
            reason: Reason for stop (e.g., 'stop_loss', 'trailing_ema9')

        Returns:
            True if sent successfully
        """
        pnl = (stop_price - entry_price) * qty
        if pnl < 0:
            emoji = "🛑"
            status = "STOP LOSS"
        else:
            emoji = "🎯"
            status = "TRAILING STOP"

        direction = "+" if pnl >= 0 else ""

        message = f"""
{emoji} *{status} ACTIVADO*

*{symbol}*

Entry: ${entry_price:.2f}
Stop: ${stop_price:.2f}
Qty: {qty}
P/L: {direction}${pnl:.2f}

Razon: {reason}
        """
        return await self.send_message(message.strip())

    async def ask_session_close_confirmation(self, positions: list, total_pnl: float) -> bool:
        """
        Ask user for confirmation before closing session

        Args:
            positions: List of open positions
            total_pnl: Current unrealized P/L

        Returns:
            True if user confirms, False otherwise
        """
        self.pending_session_close = True
        self.session_close_event = asyncio.Event()
        self.session_close_confirmed = False

        emoji = "🟢" if total_pnl >= 0 else "🔴"
        direction = "+" if total_pnl >= 0 else ""

        # Build positions summary
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

        await self.send_message(message.strip())

        # Wait for response with timeout (5 minutes)
        try:
            await asyncio.wait_for(self.session_close_event.wait(), timeout=300)
        except asyncio.TimeoutError:
            await self.send_message("⏰ Timeout - Cerrando posiciones automáticamente...")
            self.session_close_confirmed = True

        self.pending_session_close = False
        return self.session_close_confirmed

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

        # Handle session close confirmation first (priority)
        if self.pending_session_close:
            if text in ["SI", "SÍ", "YES", "Y", "OK"]:
                self.session_close_confirmed = True
                self.session_close_event.set()
                return
            elif text in ["NO", "N", "CANCEL"]:
                self.session_close_confirmed = False
                self.session_close_event.set()
                await self.send_message("📌 Posiciones mantenidas abiertas\n⚠️ Recuerda cerrarlas manualmente con /close")
                return

        # Handle trade signal confirmation responses
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

    async def _cmd_level(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /level command - view or change signal sensitivity level"""
        args = context.args

        if not args:
            # Show current level and configuration
            info = orb_strategy.get_signal_level_info()
            message = f"""
📊 *Nivel de Sensibilidad de Señales*

*Nivel Actual:* {info['level']}

*Configuración:*
• Min Score: {info['min_signal_score']}/100
• Min Volume: {info['min_relative_volume']}x
• ORB Range: {info['min_orb_range_pct']}% - {info['max_orb_range_pct']}%
• Último Trade: {info['latest_trade_time']}
• Confirmar Cierre: {'Sí' if info['require_candle_close'] else 'No'}
• RSI Sobrecompra: {info['rsi_overbought']}
• RSI Sobreventa: {info['rsi_oversold']}
• Sentiment Long: ≥ {info['min_sentiment_long']}
• Sentiment Short: ≤ {info['max_sentiment_short']}

*Cambiar nivel:*
/level STRICT - Conservador (menos señales, alta confianza)
/level MODERATE - Equilibrado (recomendado)
/level RELAXED - Agresivo (más señales, mayor riesgo)
            """
            await update.message.reply_text(message.strip(), parse_mode="Markdown")
        else:
            # Try to change level
            new_level = args[0].upper()

            if new_level not in ['STRICT', 'MODERATE', 'RELAXED']:
                await update.message.reply_text(
                    "❌ Nivel inválido. Usa: STRICT, MODERATE, o RELAXED"
                )
                return

            if orb_strategy.set_signal_level(new_level):
                info = orb_strategy.get_signal_level_info()
                message = f"""
✅ *Nivel cambiado a {info['level']}*

*Nueva configuración:*
• Min Score: {info['min_signal_score']}/100
• Min Volume: {info['min_relative_volume']}x
• Último Trade: {info['latest_trade_time']}
• ORB Range: {info['min_orb_range_pct']}% - {info['max_orb_range_pct']}%

⚠️ El cambio aplica inmediatamente
                """
                await update.message.reply_text(message.strip(), parse_mode="Markdown")
            else:
                await update.message.reply_text("❌ Error al cambiar nivel")

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
/level - Ver/cambiar sensibilidad de señales
/start - Reanudar el bot
/stop - Pausar el bot
/close - Cerrar todas las posiciones
/help - Ver esta ayuda

*Niveles de sensibilidad:*
/level STRICT - Conservador
/level MODERATE - Equilibrado
/level RELAXED - Agresivo

*Respuestas a señales:*
SI / YES - Ejecutar trade
NO - Ignorar señal
        """
        await update.message.reply_text(message.strip(), parse_mode="Markdown")


# Global bot instance
telegram_bot = TradingTelegramBot()
