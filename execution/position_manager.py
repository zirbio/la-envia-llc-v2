"""
Position management with partial profit taking and EMA9 trailing stop

Manages position lifecycle:
- Entry with standalone stop loss
- Partial close at 1R profit (50%)
- Move stop to breakeven
- Trail remaining 50% with EMA9 on 5-minute bars
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Awaitable
from enum import Enum
from loguru import logger

from config.settings import settings
from data.market_data import market_data
from data.indicators import calculate_ema
from execution.orders import order_executor, OrderResult
from strategy.orb import TradeSignal, SignalType


class PositionState(Enum):
    """Position lifecycle states"""
    OPEN = "OPEN"                      # Full position, original stop
    PARTIAL_CLOSED = "PARTIAL_CLOSED"  # 50% closed, stop at breakeven, trailing EMA9
    STOPPED = "STOPPED"                # Exit via stop loss
    TRAILED = "TRAILED"                # Exit via trailing EMA9 stop
    CLOSED = "CLOSED"                  # Manual close or target hit


@dataclass
class PositionEvent:
    """Event emitted when position state changes"""
    event_type: str  # 'partial_close', 'stop_updated', 'position_closed', 'trailing_activated'
    symbol: str
    details: dict = field(default_factory=dict)


@dataclass
class ManagedPosition:
    """A position being actively managed"""
    symbol: str
    side: str                          # 'long' or 'short'
    original_qty: int
    entry_price: float
    original_stop_loss: float
    risk_amount: float                 # Value of 1R
    state: PositionState
    current_qty: int
    current_stop_loss: float
    current_stop_order_id: Optional[str] = None
    entry_order_id: Optional[str] = None
    partial_close_order_id: Optional[str] = None
    entry_time: datetime = field(default_factory=datetime.now)
    partial_close_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    realized_pnl: float = 0.0
    exit_reason: Optional[str] = None

    @property
    def risk_per_share(self) -> float:
        """Calculate risk per share"""
        return abs(self.entry_price - self.original_stop_loss)

    def calculate_r_multiple(self, current_price: float) -> float:
        """
        Calculate current R-multiple (profit in terms of risk)

        Args:
            current_price: Current market price

        Returns:
            R-multiple (1.0 = 1R profit, -1.0 = 1R loss)
        """
        if self.risk_per_share == 0:
            return 0.0

        if self.side == 'long':
            profit_per_share = current_price - self.entry_price
        else:  # short
            profit_per_share = self.entry_price - current_price

        return profit_per_share / self.risk_per_share


class PositionManager:
    """
    Manages positions with partial profit taking and trailing stops

    Flow:
    1. register_position() - Register new position after entry fill
    2. check_all_positions() - Called every N seconds to monitor positions
       - If R >= 1: Execute partial close, move stop to breakeven
       - If trailing: Update stop to EMA9 if tighter
    3. Events emitted for Telegram notifications
    """

    def __init__(self):
        self.config = settings.trading
        self.managed_positions: dict[str, ManagedPosition] = {}
        # Callback for notifications
        self.on_event: Optional[Callable[[PositionEvent], Awaitable[None]]] = None

    async def register_position(
        self,
        signal: TradeSignal,
        fill_price: float,
        qty: int,
        stop_order_id: str,
        actual_stop_price: float = None
    ) -> ManagedPosition:
        """
        Register a new position for management

        Args:
            signal: Original trade signal
            fill_price: Actual fill price
            qty: Filled quantity
            stop_order_id: ID of the stop loss order
            actual_stop_price: Actual stop price sent to Alpaca (for 1R consistency)

        Returns:
            ManagedPosition object
        """
        side = 'long' if signal.signal_type == SignalType.LONG else 'short'

        # Use actual stop price if provided, otherwise use signal's stop_loss
        effective_stop = actual_stop_price if actual_stop_price is not None else signal.stop_loss

        risk_per_share = abs(fill_price - effective_stop)
        risk_amount = risk_per_share * qty

        position = ManagedPosition(
            symbol=signal.symbol,
            side=side,
            original_qty=qty,
            entry_price=fill_price,
            original_stop_loss=effective_stop,
            risk_amount=risk_amount,
            state=PositionState.OPEN,
            current_qty=qty,
            current_stop_loss=effective_stop,
            current_stop_order_id=stop_order_id
        )

        self.managed_positions[signal.symbol] = position

        logger.info(
            f"Position registered: {side.upper()} {qty} {signal.symbol} "
            f"@ ${fill_price:.2f}, stop ${effective_stop:.2f}, "
            f"1R = ${risk_amount:.2f}"
        )

        return position

    async def check_all_positions(self) -> list[PositionEvent]:
        """
        Check all managed positions and execute management logic

        Returns:
            List of events that occurred
        """
        events = []

        for symbol, position in list(self.managed_positions.items()):
            if position.state in [PositionState.STOPPED, PositionState.TRAILED, PositionState.CLOSED]:
                continue

            try:
                position_events = await self._check_position(position)
                events.extend(position_events)
            except Exception as e:
                logger.error(f"Error checking position {symbol}: {e}")

        return events

    async def _check_position(self, position: ManagedPosition) -> list[PositionEvent]:
        """
        Check a single position and apply management rules

        Args:
            position: Position to check

        Returns:
            List of events
        """
        events = []

        # Get current price
        quote = market_data.get_latest_quote(position.symbol)
        if not quote:
            return events

        current_price = quote['mid']
        r_multiple = position.calculate_r_multiple(current_price)

        # Check if position still exists in Alpaca
        alpaca_position = order_executor.get_position(position.symbol)
        if not alpaca_position:
            # Before assuming position was closed, verify the entry actually filled
            # If stop order is 'held', it means entry never filled (phantom position)
            if position.current_stop_order_id:
                stop_order = order_executor.get_order_by_id(position.current_stop_order_id)
                if stop_order and stop_order.get('status') == 'held':
                    logger.warning(
                        f"Position {position.symbol}: entry order never filled "
                        f"(stop order status='held'), removing phantom position"
                    )
                    self.remove_position(position.symbol)
                    return events  # No position_closed event for phantom position

            # Entry was filled, position is now closed (stop hit or manual)
            position.state = PositionState.STOPPED
            position.close_time = datetime.now()
            position.exit_reason = "Stop loss triggered"

            # Get the actual exit price from the stop order (not trigger price)
            actual_exit_price = position.current_stop_loss  # Fallback to trigger price
            if position.current_stop_order_id:
                stop_order = order_executor.get_order_by_id(position.current_stop_order_id)
                if stop_order and stop_order.get('filled_avg_price'):
                    actual_exit_price = stop_order['filled_avg_price']
                    logger.info(
                        f"Position {position.symbol} closed: "
                        f"stop trigger=${position.current_stop_loss:.2f}, "
                        f"actual fill=${actual_exit_price:.2f}"
                    )

            events.append(PositionEvent(
                event_type='position_closed',
                symbol=position.symbol,
                details={
                    'reason': 'stop_loss',
                    'entry_price': position.entry_price,
                    'stop_loss': position.current_stop_loss,
                    'exit_price': actual_exit_price,  # Real fill price
                    'qty': position.current_qty
                }
            ))
            return events

        # Update current quantity from Alpaca
        position.current_qty = alpaca_position.qty

        # STATE: OPEN - Check for 1R profit to trigger partial close
        if position.state == PositionState.OPEN:
            if self.config.partial_close_enabled and r_multiple >= self.config.partial_close_at_r:
                partial_events = await self._execute_partial_close(position, current_price)
                events.extend(partial_events)

        # STATE: PARTIAL_CLOSED - Update trailing stop based on EMA9
        if position.state == PositionState.PARTIAL_CLOSED:
            if self.config.trailing_stop_enabled:
                trail_events = await self._update_trailing_stop(position)
                events.extend(trail_events)

        return events

    async def _execute_partial_close(
        self,
        position: ManagedPosition,
        current_price: float
    ) -> list[PositionEvent]:
        """
        Execute partial close: sell 50%, move stop to breakeven

        Args:
            position: Position to partially close
            current_price: Current market price (used as fallback only)

        Returns:
            List of events
        """
        events = []
        close_qty = int(position.original_qty * self.config.partial_close_percent)

        if close_qty < 1:
            close_qty = 1

        logger.info(
            f"Executing partial close for {position.symbol}: "
            f"closing {close_qty} of {position.current_qty} shares"
        )

        # 1. Close partial position and get actual fill price
        close_result = await order_executor.close_partial_position(
            symbol=position.symbol,
            qty=close_qty,
            position_side=position.side
        )

        if not close_result.success:
            logger.error(f"Failed to close partial position: {close_result.error}")
            return events

        position.partial_close_order_id = close_result.order_id
        position.partial_close_time = datetime.now()

        # Use actual fill price from the order, fallback to quote mid if unavailable
        actual_fill_price = close_result.filled_price or current_price

        if close_result.filled_price:
            logger.info(f"Partial close filled at actual price: ${actual_fill_price:.2f}")
        else:
            logger.warning(f"Using quote price as fallback for P/L calculation: ${current_price:.2f}")

        # Calculate realized P/L using ACTUAL fill price
        if position.side == 'long':
            partial_pnl = (actual_fill_price - position.entry_price) * close_qty
        else:
            partial_pnl = (position.entry_price - actual_fill_price) * close_qty

        position.realized_pnl += partial_pnl

        # 2. Cancel old stop and create new stop at breakeven
        new_stop_price = position.entry_price  # Breakeven

        if position.current_stop_order_id:
            replace_result = order_executor.cancel_and_replace_stop(
                order_id=position.current_stop_order_id,
                new_stop_price=new_stop_price
            )

            if replace_result.success:
                position.current_stop_order_id = replace_result.order_id
                position.current_stop_loss = new_stop_price
            else:
                # If replace fails, try cancel and create new
                logger.warning(f"Replace failed, trying cancel+create: {replace_result.error}")
                order_executor.cancel_order(position.current_stop_order_id)

                remaining_qty = position.current_qty - close_qty
                stop_result = order_executor.create_stop_loss_order(
                    symbol=position.symbol,
                    qty=remaining_qty,
                    stop_price=new_stop_price,
                    position_side=position.side
                )

                if stop_result.success:
                    position.current_stop_order_id = stop_result.order_id
                    position.current_stop_loss = new_stop_price

        # 3. Update position state
        position.state = PositionState.PARTIAL_CLOSED
        position.current_qty -= close_qty

        # Calculate R-multiple using actual fill price
        r_multiple = (actual_fill_price - position.entry_price) / position.risk_per_share if position.side == 'long' else (position.entry_price - actual_fill_price) / position.risk_per_share

        logger.info(
            f"Partial close complete for {position.symbol}: "
            f"closed {close_qty} @ ${actual_fill_price:.2f} (actual fill), "
            f"remaining {position.current_qty}, "
            f"stop moved to breakeven ${new_stop_price:.2f}, "
            f"realized P/L: ${partial_pnl:.2f}"
        )

        events.append(PositionEvent(
            event_type='partial_close',
            symbol=position.symbol,
            details={
                'closed_qty': close_qty,
                'close_price': actual_fill_price,  # Use actual fill price, not quote
                'remaining_qty': position.current_qty,
                'new_stop': new_stop_price,
                'realized_pnl': partial_pnl,
                'r_multiple': r_multiple  # Calculate using actual fill price
            }
        ))

        events.append(PositionEvent(
            event_type='trailing_activated',
            symbol=position.symbol,
            details={
                'ema_period': self.config.trailing_ema_period,
                'timeframe': f"{self.config.trailing_bar_timeframe}min"
            }
        ))

        return events

    async def _update_trailing_stop(self, position: ManagedPosition) -> list[PositionEvent]:
        """
        Update trailing stop based on EMA9 of 5-minute bars

        Args:
            position: Position to update

        Returns:
            List of events
        """
        events = []

        # Get EMA9 value from 5-minute bars
        ema9 = await self._get_ema9_value(position.symbol)

        if ema9 is None:
            return events

        # Only move stop in favorable direction
        should_update = False
        old_stop = position.current_stop_loss

        if position.side == 'long':
            # For longs: only raise stop, never lower
            if ema9 > position.current_stop_loss:
                should_update = True
                new_stop = ema9
        else:  # short
            # For shorts: only lower stop, never raise
            if ema9 < position.current_stop_loss:
                should_update = True
                new_stop = ema9

        if not should_update:
            return events

        # Update the stop order
        if position.current_stop_order_id:
            replace_result = order_executor.cancel_and_replace_stop(
                order_id=position.current_stop_order_id,
                new_stop_price=new_stop
            )

            if replace_result.success:
                position.current_stop_order_id = replace_result.order_id
                position.current_stop_loss = new_stop

                logger.info(
                    f"Trailing stop updated for {position.symbol}: "
                    f"${old_stop:.2f} -> ${new_stop:.2f} (EMA9)"
                )

                events.append(PositionEvent(
                    event_type='stop_updated',
                    symbol=position.symbol,
                    details={
                        'old_stop': old_stop,
                        'new_stop': new_stop,
                        'ema9': ema9,
                        'method': 'trailing_ema9'
                    }
                ))
            else:
                logger.warning(f"Failed to update trailing stop: {replace_result.error}")

        return events

    async def _get_ema9_value(self, symbol: str) -> Optional[float]:
        """
        Get current EMA9 value from 5-minute bars

        Args:
            symbol: Stock symbol

        Returns:
            EMA9 value or None
        """
        try:
            bars = market_data.get_5min_bars(symbol, limit=50)

            if bars.empty or len(bars) < self.config.trailing_ema_period:
                logger.warning(f"Not enough 5min bars for EMA9 calculation: {symbol}")
                return None

            ema = calculate_ema(bars['close'], period=self.config.trailing_ema_period)

            if ema.empty:
                return None

            ema9_value = float(ema.iloc[-1])
            return ema9_value

        except Exception as e:
            logger.error(f"Error calculating EMA9 for {symbol}: {e}")
            return None

    def get_position(self, symbol: str) -> Optional[ManagedPosition]:
        """Get managed position by symbol"""
        return self.managed_positions.get(symbol)

    def get_all_positions(self) -> list[ManagedPosition]:
        """Get all managed positions"""
        return list(self.managed_positions.values())

    def get_active_positions(self) -> list[ManagedPosition]:
        """Get only active (non-closed) positions"""
        return [
            p for p in self.managed_positions.values()
            if p.state in [PositionState.OPEN, PositionState.PARTIAL_CLOSED]
        ]

    async def close_position(
        self,
        symbol: str,
        reason: str = "manual"
    ) -> Optional[PositionEvent]:
        """
        Manually close a managed position

        Args:
            symbol: Symbol to close
            reason: Reason for closing

        Returns:
            PositionEvent or None
        """
        position = self.managed_positions.get(symbol)
        if not position:
            return None

        # Cancel stop order if exists
        if position.current_stop_order_id:
            order_executor.cancel_order(position.current_stop_order_id)

        # Close remaining position
        result = order_executor.close_position(symbol)

        if result.success:
            position.state = PositionState.CLOSED
            position.close_time = datetime.now()
            position.exit_reason = reason

            return PositionEvent(
                event_type='position_closed',
                symbol=symbol,
                details={
                    'reason': reason,
                    'realized_pnl': position.realized_pnl
                }
            )

        return None

    def remove_position(self, symbol: str):
        """Remove a position from management"""
        if symbol in self.managed_positions:
            del self.managed_positions[symbol]

    def reset(self):
        """Reset all managed positions"""
        self.managed_positions.clear()
        logger.info("Position manager reset")


# Global position manager instance
position_manager = PositionManager()
