"""
Opening Range Breakout (ORB) Strategy Implementation
"""
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional
from enum import Enum
import pandas as pd
from loguru import logger

from config.settings import settings
from data.market_data import market_data
from data.indicators import (
    calculate_rsi, calculate_vwap, detect_volume_spike,
    calculate_macd, is_macd_bullish, is_macd_bearish,
    IndicatorCalculator
)


class SignalType(Enum):
    """Type of trading signal"""
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class OpeningRange:
    """Opening Range data for a symbol"""
    symbol: str
    high: float
    low: float
    range_size: float
    vwap: float
    timestamp: datetime

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2


@dataclass
class TradeSignal:
    """Trading signal with all relevant data"""
    symbol: str
    signal_type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    risk_amount: float
    orb_high: float
    orb_low: float
    vwap: float
    rsi: float
    relative_volume: float
    timestamp: datetime
    # MACD confirmation
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    # Sentiment score (-1 to 1)
    sentiment_score: float = 0.0

    def __str__(self) -> str:
        emoji = "🟢" if self.signal_type == SignalType.LONG else "🔴"
        macd_status = "✓" if (self.macd_histogram > 0 and self.signal_type == SignalType.LONG) or \
                            (self.macd_histogram < 0 and self.signal_type == SignalType.SHORT) else "○"
        return (
            f"{emoji} {self.signal_type.value} {self.symbol}\n"
            f"Entry: ${self.entry_price:.2f}\n"
            f"Stop: ${self.stop_loss:.2f}\n"
            f"Target: ${self.take_profit:.2f}\n"
            f"Size: {self.position_size} shares\n"
            f"Risk: ${self.risk_amount:.2f}\n"
            f"MACD: {macd_status}"
        )


@dataclass
class TradeResult:
    """Result of a completed trade for Kelly calculation"""
    symbol: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    won: bool


class ORBStrategy:
    """
    Opening Range Breakout Strategy with MACD Confirmation and Kelly Sizing

    Rules:
    - Calculate high/low of first N minutes (Opening Range)
    - LONG: Price breaks above ORB high + price > VWAP + volume spike + RSI < 70 + MACD bullish
    - SHORT: Price breaks below ORB low + price < VWAP + volume spike + RSI > 30 + MACD bearish
    - Stop: Opposite side of ORB
    - Target: 2:1 risk/reward ratio
    - Position size: Kelly Criterion (half-Kelly for safety)
    """

    def __init__(self):
        self.config = settings.trading
        self.opening_ranges: dict[str, OpeningRange] = {}
        self.signals_today: list[TradeSignal] = []
        # Trade history for Kelly calculation
        self.trade_history: list[TradeResult] = []
        # Sentiment scores cache
        self.sentiment_cache: dict[str, float] = {}
        # Kelly parameters (updated after each trade)
        self.win_rate: float = 0.5  # Default 50%
        self.avg_win: float = 0.02  # Default 2%
        self.avg_loss: float = 0.01  # Default 1%
        self.kelly_fraction: float = 0.5  # Half-Kelly for safety

    def calculate_opening_range(self, symbol: str) -> Optional[OpeningRange]:
        """
        Calculate the Opening Range for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            OpeningRange if successful, None otherwise
        """
        try:
            # Get bars for the opening range period
            bars = market_data.get_bars(
                symbol=symbol,
                limit=self.config.orb_period_minutes + 5
            )

            if bars.empty or len(bars) < self.config.orb_period_minutes:
                logger.warning(f"Not enough bars to calculate ORB for {symbol}")
                return None

            # Get first N minutes of bars
            orb_bars = bars.head(self.config.orb_period_minutes)

            orb = OpeningRange(
                symbol=symbol,
                high=orb_bars['high'].max(),
                low=orb_bars['low'].min(),
                range_size=orb_bars['high'].max() - orb_bars['low'].min(),
                vwap=calculate_vwap(orb_bars).iloc[-1],
                timestamp=datetime.now()
            )

            self.opening_ranges[symbol] = orb
            logger.info(
                f"ORB for {symbol}: High=${orb.high:.2f}, "
                f"Low=${orb.low:.2f}, Range=${orb.range_size:.2f}"
            )

            return orb

        except Exception as e:
            logger.error(f"Error calculating ORB for {symbol}: {e}")
            return None

    def check_breakout(
        self,
        symbol: str,
        current_price: float,
        current_volume: int,
        avg_volume: int
    ) -> Optional[TradeSignal]:
        """
        Check if price has broken out of the Opening Range

        Args:
            symbol: Stock symbol
            current_price: Current price
            current_volume: Current bar's volume
            avg_volume: Average volume for comparison

        Returns:
            TradeSignal if breakout detected, None otherwise
        """
        # Check if we have an opening range for this symbol
        if symbol not in self.opening_ranges:
            logger.debug(f"No ORB calculated for {symbol}")
            return None

        orb = self.opening_ranges[symbol]

        # Check daily trade limit (use debug to avoid spam during monitoring)
        if len(self.signals_today) >= self.config.max_trades_per_day:
            logger.debug("Daily trade limit reached")
            return None

        # Get current market data with indicators
        bars = market_data.get_bars(symbol, limit=50)
        if bars.empty:
            return None

        # Calculate all indicators including MACD
        indicator_calc = IndicatorCalculator(bars)
        indicator_calc.add_all_indicators()
        indicators = indicator_calc.get_latest_indicators()

        current_vwap = indicators.get('vwap', 0)
        current_rsi = indicators.get('rsi', 50)
        relative_volume = current_volume / max(avg_volume / 390, 1)  # Per minute avg

        # MACD values
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_histogram = indicators.get('macd_histogram', 0)
        prev_histogram = indicators.get('prev_macd_histogram', 0)

        # Get sentiment score if available
        sentiment = self.sentiment_cache.get(symbol, 0.0)

        # Check for LONG breakout (with MACD confirmation)
        macd_bullish = is_macd_bullish(macd, macd_signal, macd_histogram, prev_histogram)
        if self._check_long_conditions(
            current_price, orb, current_vwap, current_rsi, relative_volume, macd_bullish, sentiment
        ):
            return self._create_long_signal(
                symbol, current_price, orb, current_vwap, current_rsi, relative_volume,
                macd, macd_signal, macd_histogram, sentiment
            )

        # Check for SHORT breakout (with MACD confirmation)
        macd_bearish = is_macd_bearish(macd, macd_signal, macd_histogram, prev_histogram)
        if self._check_short_conditions(
            current_price, orb, current_vwap, current_rsi, relative_volume, macd_bearish, sentiment
        ):
            return self._create_short_signal(
                symbol, current_price, orb, current_vwap, current_rsi, relative_volume,
                macd, macd_signal, macd_histogram, sentiment
            )

        return None

    def _check_long_conditions(
        self,
        price: float,
        orb: OpeningRange,
        vwap: float,
        rsi: float,
        rel_volume: float,
        macd_bullish: bool = True,
        sentiment: float = 0.0
    ) -> bool:
        """Check if all LONG conditions are met (including MACD and sentiment)"""
        # Check each condition individually for logging
        c1 = price > orb.high
        c2 = price > vwap
        c3 = rel_volume >= self.config.min_relative_volume
        c4 = rsi < self.config.rsi_overbought
        c5 = macd_bullish
        c6 = sentiment >= settings.sentiment.min_score_long

        conditions = [c1, c2, c3, c4, c5, c6]

        # Log diagnostic info for this symbol
        status = (
            f"{'✓' if c1 else '✗'} price>${'ORB' if c1 else f'ORB({orb.high:.2f})'} "
            f"{'✓' if c2 else '✗'} VWAP "
            f"{'✓' if c3 else '✗'} vol({rel_volume:.1f}x) "
            f"{'✓' if c4 else '✗'} RSI({rsi:.0f}) "
            f"{'✓' if c5 else '✗'} MACD "
            f"{'✓' if c6 else '✗'} sent({sentiment:.2f})"
        )

        if any([c1, c2]):  # Only log if price is near breakout levels
            logger.info(f"LONG {orb.symbol}: ${price:.2f} | {status}")

        if all(conditions):
            logger.info(
                f"🟢 LONG SIGNAL {orb.symbol}: price={price:.2f} > ORB={orb.high:.2f}, "
                f"VWAP={vwap:.2f}, vol={rel_volume:.1f}x, RSI={rsi:.0f}, sent={sentiment:.2f}"
            )

        return all(conditions)

    def _check_short_conditions(
        self,
        price: float,
        orb: OpeningRange,
        vwap: float,
        rsi: float,
        rel_volume: float,
        macd_bearish: bool = True,
        sentiment: float = 0.0
    ) -> bool:
        """Check if all SHORT conditions are met (including MACD and sentiment)"""
        # Check each condition individually for logging
        c1 = price < orb.low
        c2 = price < vwap
        c3 = rel_volume >= self.config.min_relative_volume
        c4 = rsi > self.config.rsi_oversold
        c5 = macd_bearish
        c6 = sentiment <= settings.sentiment.max_score_short

        conditions = [c1, c2, c3, c4, c5, c6]

        # Log diagnostic info for this symbol
        status = (
            f"{'✓' if c1 else '✗'} price<{'ORB' if c1 else f'ORB({orb.low:.2f})'} "
            f"{'✓' if c2 else '✗'} VWAP "
            f"{'✓' if c3 else '✗'} vol({rel_volume:.1f}x) "
            f"{'✓' if c4 else '✗'} RSI({rsi:.0f}) "
            f"{'✓' if c5 else '✗'} MACD "
            f"{'✓' if c6 else '✗'} sent({sentiment:.2f})"
        )

        if any([c1, c2]):  # Only log if price is near breakout levels
            logger.info(f"SHORT {orb.symbol}: ${price:.2f} | {status}")

        if all(conditions):
            logger.info(
                f"🔴 SHORT SIGNAL {orb.symbol}: price={price:.2f} < ORB={orb.low:.2f}, "
                f"VWAP={vwap:.2f}, vol={rel_volume:.1f}x, RSI={rsi:.0f}, sent={sentiment:.2f}"
            )

        return all(conditions)

    def _create_long_signal(
        self,
        symbol: str,
        entry_price: float,
        orb: OpeningRange,
        vwap: float,
        rsi: float,
        rel_volume: float,
        macd: float = 0.0,
        macd_signal: float = 0.0,
        macd_histogram: float = 0.0,
        sentiment: float = 0.0
    ) -> TradeSignal:
        """Create a LONG trade signal"""
        stop_loss = orb.low
        risk_per_share = entry_price - stop_loss
        take_profit = entry_price + (risk_per_share * self.config.reward_risk_ratio)

        position_size, risk_amount = self._calculate_position_size_kelly(
            entry_price, stop_loss
        )

        signal = TradeSignal(
            symbol=symbol,
            signal_type=SignalType.LONG,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            risk_amount=risk_amount,
            orb_high=orb.high,
            orb_low=orb.low,
            vwap=vwap,
            rsi=rsi,
            relative_volume=rel_volume,
            timestamp=datetime.now(),
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            sentiment_score=sentiment
        )

        self.signals_today.append(signal)
        logger.info(f"LONG signal generated: {signal}")

        return signal

    def _create_short_signal(
        self,
        symbol: str,
        entry_price: float,
        orb: OpeningRange,
        vwap: float,
        rsi: float,
        rel_volume: float,
        macd: float = 0.0,
        macd_signal: float = 0.0,
        macd_histogram: float = 0.0,
        sentiment: float = 0.0
    ) -> TradeSignal:
        """Create a SHORT trade signal"""
        stop_loss = orb.high
        risk_per_share = stop_loss - entry_price
        take_profit = entry_price - (risk_per_share * self.config.reward_risk_ratio)

        position_size, risk_amount = self._calculate_position_size_kelly(
            entry_price, stop_loss
        )

        signal = TradeSignal(
            symbol=symbol,
            signal_type=SignalType.SHORT,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            risk_amount=risk_amount,
            orb_high=orb.high,
            orb_low=orb.low,
            vwap=vwap,
            rsi=rsi,
            relative_volume=rel_volume,
            timestamp=datetime.now(),
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            sentiment_score=sentiment
        )

        self.signals_today.append(signal)
        logger.info(f"SHORT signal generated: {signal}")

        return signal

    def _calculate_position_size_kelly(
        self,
        entry_price: float,
        stop_loss: float
    ) -> tuple[int, float]:
        """
        Calculate position size using Kelly Criterion

        Kelly formula: f* = (p * b - q) / b
        Where:
            p = probability of winning (win_rate)
            q = probability of losing (1 - win_rate)
            b = ratio of avg_win to avg_loss

        Returns:
            Tuple of (position_size, risk_amount)
        """
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share == 0:
            return 0, 0.0

        # Calculate Kelly fraction
        kelly = self._calculate_kelly_fraction()

        # Apply Half-Kelly for safety (reduces variance)
        adjusted_kelly = kelly * self.kelly_fraction

        # Cap Kelly at max risk per trade from config
        max_kelly = self.config.risk_per_trade
        effective_risk_pct = min(adjusted_kelly, max_kelly)

        # Calculate position size
        max_risk = self.config.max_capital * effective_risk_pct
        position_size = int(max_risk / risk_per_share)

        # Ensure we don't exceed max capital
        max_shares = int(self.config.max_capital / entry_price)
        position_size = min(position_size, max_shares)

        # Minimum position size
        position_size = max(position_size, 1)

        actual_risk = position_size * risk_per_share

        logger.debug(
            f"Kelly sizing: win_rate={self.win_rate:.2f}, "
            f"kelly={kelly:.3f}, adjusted={adjusted_kelly:.3f}, "
            f"position={position_size} shares, risk=${actual_risk:.2f}"
        )

        return position_size, actual_risk

    def _calculate_kelly_fraction(self) -> float:
        """
        Calculate Kelly fraction based on trade history

        Returns:
            Kelly fraction (0.0 to 1.0)
        """
        if len(self.trade_history) < 5:
            # Not enough history, use conservative default
            return self.config.risk_per_trade

        # Calculate win rate
        wins = sum(1 for t in self.trade_history if t.won)
        self.win_rate = wins / len(self.trade_history)

        # Calculate average win and loss percentages
        win_pcts = [t.pnl_pct for t in self.trade_history if t.won]
        loss_pcts = [abs(t.pnl_pct) for t in self.trade_history if not t.won]

        self.avg_win = sum(win_pcts) / len(win_pcts) if win_pcts else 0.02
        self.avg_loss = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0.01

        # Avoid division by zero
        if self.avg_loss == 0:
            return self.config.risk_per_trade

        # Kelly formula: f* = (p * b - q) / b
        # Where b = avg_win / avg_loss
        b = self.avg_win / self.avg_loss
        p = self.win_rate
        q = 1 - p

        kelly = (p * b - q) / b

        # Clamp between 0 and max allowed
        kelly = max(0.0, min(kelly, 0.25))  # Max 25% of capital

        return kelly

    def record_trade_result(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        is_long: bool
    ):
        """
        Record a completed trade for Kelly calculation

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            exit_price: Exit price
            is_long: True if long position
        """
        if is_long:
            pnl = exit_price - entry_price
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl = entry_price - exit_price
            pnl_pct = (entry_price - exit_price) / entry_price

        result = TradeResult(
            symbol=symbol,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            won=pnl > 0
        )

        self.trade_history.append(result)

        # Keep only last 50 trades
        if len(self.trade_history) > 50:
            self.trade_history = self.trade_history[-50:]

        logger.info(
            f"Trade recorded: {symbol} {'WIN' if result.won else 'LOSS'} "
            f"PnL: {pnl_pct*100:.2f}% | Win rate: {self.win_rate:.1%}"
        )

    def update_sentiment(self, symbol: str, sentiment: float):
        """
        Update sentiment score for a symbol

        Args:
            symbol: Stock symbol
            sentiment: Sentiment score (-1 to 1)
        """
        self.sentiment_cache[symbol] = max(-1.0, min(1.0, sentiment))

    def get_kelly_stats(self) -> dict:
        """Get current Kelly statistics"""
        return {
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'kelly_fraction': self._calculate_kelly_fraction(),
            'trade_count': len(self.trade_history)
        }

    def reset_daily(self):
        """Reset daily tracking data"""
        self.opening_ranges.clear()
        self.signals_today.clear()
        self.sentiment_cache.clear()
        logger.info("Daily data reset")


# Global strategy instance
orb_strategy = ORBStrategy()
