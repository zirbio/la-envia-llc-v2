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
from data.indicators import calculate_rsi, calculate_vwap, detect_volume_spike


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

    def __str__(self) -> str:
        emoji = "🟢" if self.signal_type == SignalType.LONG else "🔴"
        return (
            f"{emoji} {self.signal_type.value} {self.symbol}\n"
            f"Entry: ${self.entry_price:.2f}\n"
            f"Stop: ${self.stop_loss:.2f}\n"
            f"Target: ${self.take_profit:.2f}\n"
            f"Size: {self.position_size} shares\n"
            f"Risk: ${self.risk_amount:.2f}"
        )


class ORBStrategy:
    """
    Opening Range Breakout Strategy

    Rules:
    - Calculate high/low of first N minutes (Opening Range)
    - LONG: Price breaks above ORB high + price > VWAP + volume spike + RSI < 70
    - SHORT: Price breaks below ORB low + price < VWAP + volume spike + RSI > 30
    - Stop: Opposite side of ORB
    - Target: 2:1 risk/reward ratio
    """

    def __init__(self):
        self.config = settings.trading
        self.opening_ranges: dict[str, OpeningRange] = {}
        self.signals_today: list[TradeSignal] = []

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

        # Check daily trade limit
        if len(self.signals_today) >= self.config.max_trades_per_day:
            logger.info("Daily trade limit reached")
            return None

        # Get current market data with indicators
        bars = market_data.get_bars(symbol, limit=50)
        if bars.empty:
            return None

        # Calculate indicators
        current_vwap = calculate_vwap(bars).iloc[-1]
        current_rsi = calculate_rsi(bars['close']).iloc[-1]
        relative_volume = current_volume / max(avg_volume / 390, 1)  # Per minute avg

        # Check for LONG breakout
        if self._check_long_conditions(
            current_price, orb, current_vwap, current_rsi, relative_volume
        ):
            return self._create_long_signal(
                symbol, current_price, orb, current_vwap, current_rsi, relative_volume
            )

        # Check for SHORT breakout
        if self._check_short_conditions(
            current_price, orb, current_vwap, current_rsi, relative_volume
        ):
            return self._create_short_signal(
                symbol, current_price, orb, current_vwap, current_rsi, relative_volume
            )

        return None

    def _check_long_conditions(
        self,
        price: float,
        orb: OpeningRange,
        vwap: float,
        rsi: float,
        rel_volume: float
    ) -> bool:
        """Check if all LONG conditions are met"""
        conditions = [
            price > orb.high,  # Price above ORB high
            price > vwap,  # Price above VWAP
            rel_volume >= self.config.min_relative_volume,  # Volume spike
            rsi < self.config.rsi_overbought  # Not overbought
        ]

        if all(conditions):
            logger.debug(
                f"LONG conditions met: price={price:.2f} > ORB high={orb.high:.2f}, "
                f"price > VWAP={vwap:.2f}, rel_vol={rel_volume:.1f}x, RSI={rsi:.0f}"
            )

        return all(conditions)

    def _check_short_conditions(
        self,
        price: float,
        orb: OpeningRange,
        vwap: float,
        rsi: float,
        rel_volume: float
    ) -> bool:
        """Check if all SHORT conditions are met"""
        conditions = [
            price < orb.low,  # Price below ORB low
            price < vwap,  # Price below VWAP
            rel_volume >= self.config.min_relative_volume,  # Volume spike
            rsi > self.config.rsi_oversold  # Not oversold
        ]

        if all(conditions):
            logger.debug(
                f"SHORT conditions met: price={price:.2f} < ORB low={orb.low:.2f}, "
                f"price < VWAP={vwap:.2f}, rel_vol={rel_volume:.1f}x, RSI={rsi:.0f}"
            )

        return all(conditions)

    def _create_long_signal(
        self,
        symbol: str,
        entry_price: float,
        orb: OpeningRange,
        vwap: float,
        rsi: float,
        rel_volume: float
    ) -> TradeSignal:
        """Create a LONG trade signal"""
        stop_loss = orb.low
        risk_per_share = entry_price - stop_loss
        take_profit = entry_price + (risk_per_share * self.config.reward_risk_ratio)

        position_size, risk_amount = self._calculate_position_size(
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
            timestamp=datetime.now()
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
        rel_volume: float
    ) -> TradeSignal:
        """Create a SHORT trade signal"""
        stop_loss = orb.high
        risk_per_share = stop_loss - entry_price
        take_profit = entry_price - (risk_per_share * self.config.reward_risk_ratio)

        position_size, risk_amount = self._calculate_position_size(
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
            timestamp=datetime.now()
        )

        self.signals_today.append(signal)
        logger.info(f"SHORT signal generated: {signal}")

        return signal

    def _calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float
    ) -> tuple[int, float]:
        """
        Calculate position size based on risk management rules

        Returns:
            Tuple of (position_size, risk_amount)
        """
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share == 0:
            return 0, 0.0

        max_risk = self.config.max_capital * self.config.risk_per_trade
        position_size = int(max_risk / risk_per_share)

        # Ensure we don't exceed max capital
        max_shares = int(self.config.max_capital / entry_price)
        position_size = min(position_size, max_shares)

        actual_risk = position_size * risk_per_share

        return position_size, actual_risk

    def reset_daily(self):
        """Reset daily tracking data"""
        self.opening_ranges.clear()
        self.signals_today.clear()
        logger.info("Daily data reset")


# Global strategy instance
orb_strategy = ORBStrategy()
