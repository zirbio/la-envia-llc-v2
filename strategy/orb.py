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
    calculate_atr, IndicatorCalculator, indicator_cache
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
    # Phase 4: Signal quality score (0-100)
    signal_score: float = 0.0
    # Signal quality classification (ÓPTIMA/BUENA/REGULAR/DÉBIL)
    quality_level: str = ''

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
            f"Score: {self.signal_score:.0f}/100\n"
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
    - LONG: Price breaks above ORB high + price > VWAP + volume spike + RSI < overbought + MACD bullish + sentiment >= threshold
    - SHORT: Price breaks below ORB low + price < VWAP + volume spike + RSI > oversold + MACD bearish + sentiment <= threshold
    - Stop: Opposite side of ORB (or tighter ATR-based)
    - Target: 2:1 risk/reward ratio
    - Position size: Kelly Criterion (half-Kelly for safety)

    Note: RSI, volume, and sentiment thresholds are configurable via signal levels
    (STRICT, MODERATE, RELAXED). See config/settings.py for values per level.
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
        # Phase 6: Risk management tracking
        self.daily_pnl: float = 0.0  # Tracks P/L for circuit breaker
        self.consecutive_losses: int = 0  # Tracks consecutive losses for cooldown

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

            high = orb_bars['high'].max()
            low = orb_bars['low'].min()
            range_size = high - low

            # Phase 3: Filter by ORB range percentage
            range_pct = (range_size / low) * 100

            if range_pct < self.config.min_orb_range_pct:
                logger.info(
                    f"{symbol}: ORB range {range_pct:.2f}% too narrow "
                    f"(min {self.config.min_orb_range_pct}%), skipping"
                )
                return None

            if range_pct > self.config.max_orb_range_pct:
                logger.info(
                    f"{symbol}: ORB range {range_pct:.2f}% too wide "
                    f"(max {self.config.max_orb_range_pct}%), skipping"
                )
                return None

            orb = OpeningRange(
                symbol=symbol,
                high=high,
                low=low,
                range_size=range_size,
                vwap=calculate_vwap(orb_bars).iloc[-1],
                timestamp=datetime.now()
            )

            self.opening_ranges[symbol] = orb
            logger.info(
                f"ORB for {symbol}: High=${orb.high:.2f}, "
                f"Low=${orb.low:.2f}, Range=${orb.range_size:.2f} ({range_pct:.2f}%)"
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

        # Phase 6: Daily loss limit circuit breaker
        if self.daily_pnl <= -self.config.max_daily_loss:
            logger.warning(f"Daily loss limit hit: ${self.daily_pnl:.2f}, no new trades")
            return None

        # Phase 6: Consecutive loss cooldown
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            logger.info(f"Cooldown: {self.consecutive_losses} consecutive losses, pausing trades")
            return None

        # Phase 6: Trading window restriction
        try:
            latest_time_str = self.config.latest_trade_time
            latest_hour, latest_minute = map(int, latest_time_str.split(':'))
            latest_time = time(latest_hour, latest_minute)
            if datetime.now().time() > latest_time:
                logger.debug(f"Past trading window ({latest_time_str}), no new entries")
                return None
        except (ValueError, AttributeError):
            pass  # If parsing fails, ignore this check

        # Get current market data with indicators
        bars = market_data.get_bars(symbol, limit=50)
        if bars.empty:
            return None

        # Try to get cached indicators first
        indicators = indicator_cache.get(symbol, bars)
        if indicators is None:
            # Calculate all indicators including MACD
            indicator_calc = IndicatorCalculator(bars)
            indicator_calc.add_all_indicators()
            indicators = indicator_calc.get_latest_indicators()
            # Cache the results
            indicator_cache.set(symbol, bars, indicators)

        current_vwap = indicators.get('vwap', 0)
        current_rsi = indicators.get('rsi', 50)

        # Phase 2: Use time-adjusted RVOL if volume profile is available
        current_minute = market_data.get_current_minute_index()
        cumulative_volume = market_data.get_cumulative_volume_today(symbol)

        if symbol in market_data.volume_profiles and market_data.volume_profiles[symbol]:
            # Use time-adjusted RVOL (more accurate during first hour)
            relative_volume = market_data.calculate_time_adjusted_rvol(
                symbol, current_minute, cumulative_volume
            )
        else:
            # Fallback to simple RVOL calculation
            relative_volume = current_volume / max(avg_volume / 390, 1)  # Per minute avg

        # MACD values
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_histogram = indicators.get('macd_histogram', 0)
        prev_histogram = indicators.get('prev_macd_histogram', 0)

        # Get sentiment score if available
        sentiment = self.sentiment_cache.get(symbol, 0.0)

        # Get last completed candle close for confirmation (Phase 1)
        # bars.iloc[-1] is current forming candle, bars.iloc[-2] is last completed
        last_candle_close = None
        if len(bars) >= 2:
            last_candle_close = float(bars['close'].iloc[-2])

        # Phase 4: Use soft scoring system instead of hard gates
        result = self._check_breakout_with_scoring(
            symbol=symbol,
            price=current_price,
            orb=orb,
            vwap=current_vwap,
            rsi=current_rsi,
            rel_volume=relative_volume,
            macd_histogram=macd_histogram,
            sentiment=sentiment,
            last_candle_close=last_candle_close
        )

        if result is not None:
            direction, score = result
            if direction == 'LONG':
                return self._create_long_signal(
                    symbol, current_price, orb, current_vwap, current_rsi, relative_volume,
                    macd, macd_signal, macd_histogram, sentiment, score
                )
            else:  # SHORT
                return self._create_short_signal(
                    symbol, current_price, orb, current_vwap, current_rsi, relative_volume,
                    macd, macd_signal, macd_histogram, sentiment, score
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
        sentiment: float = 0.0,
        last_candle_close: Optional[float] = None
    ) -> bool:
        """Check if all LONG conditions are met (including MACD and sentiment)"""
        # Apply breakout buffer (Phase 1)
        buffer = self.config.breakout_buffer_pct
        breakout_level = orb.high * (1 + buffer)

        # Check breakout with buffer
        c1 = price > breakout_level

        # If require_candle_close is enabled, verify last completed candle closed above level
        if self.config.require_candle_close and last_candle_close is not None:
            c1 = c1 and last_candle_close > breakout_level

        c2 = price > vwap
        c3 = rel_volume >= self.config.min_relative_volume
        c4 = rsi < self.config.rsi_overbought
        c5 = macd_bullish
        c6 = sentiment >= self.config.signal_config.min_sentiment_long

        conditions = [c1, c2, c3, c4, c5, c6]

        # Log diagnostic info for this symbol
        close_status = f"close={last_candle_close:.2f}" if last_candle_close else "no_close"
        status = (
            f"{'✓' if c1 else '✗'} price>${'ORB+buf' if c1 else f'ORB({breakout_level:.2f})'} "
            f"{'✓' if c2 else '✗'} VWAP "
            f"{'✓' if c3 else '✗'} vol({rel_volume:.1f}x) "
            f"{'✓' if c4 else '✗'} RSI({rsi:.0f}) "
            f"{'✓' if c5 else '✗'} MACD "
            f"{'✓' if c6 else '✗'} sent({sentiment:.2f})"
        )

        if any([c1, c2]):  # Only log if price is near breakout levels
            logger.info(f"LONG {orb.symbol}: ${price:.2f} [{close_status}] | {status}")

        if all(conditions):
            logger.info(
                f"🟢 LONG SIGNAL {orb.symbol}: price={price:.2f} > ORB+buf={breakout_level:.2f}, "
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
        sentiment: float = 0.0,
        last_candle_close: Optional[float] = None
    ) -> bool:
        """Check if all SHORT conditions are met (including MACD and sentiment)"""
        # Apply breakout buffer (Phase 1)
        buffer = self.config.breakout_buffer_pct
        breakout_level = orb.low * (1 - buffer)

        # Check breakout with buffer
        c1 = price < breakout_level

        # If require_candle_close is enabled, verify last completed candle closed below level
        if self.config.require_candle_close and last_candle_close is not None:
            c1 = c1 and last_candle_close < breakout_level

        c2 = price < vwap
        c3 = rel_volume >= self.config.min_relative_volume
        c4 = rsi > self.config.rsi_oversold
        c5 = macd_bearish
        c6 = sentiment <= self.config.signal_config.max_sentiment_short

        conditions = [c1, c2, c3, c4, c5, c6]

        # Log diagnostic info for this symbol
        close_status = f"close={last_candle_close:.2f}" if last_candle_close else "no_close"
        status = (
            f"{'✓' if c1 else '✗'} price<{'ORB-buf' if c1 else f'ORB({breakout_level:.2f})'} "
            f"{'✓' if c2 else '✗'} VWAP "
            f"{'✓' if c3 else '✗'} vol({rel_volume:.1f}x) "
            f"{'✓' if c4 else '✗'} RSI({rsi:.0f}) "
            f"{'✓' if c5 else '✗'} MACD "
            f"{'✓' if c6 else '✗'} sent({sentiment:.2f})"
        )

        if any([c1, c2]):  # Only log if price is near breakout levels
            logger.info(f"SHORT {orb.symbol}: ${price:.2f} [{close_status}] | {status}")

        if all(conditions):
            logger.info(
                f"🔴 SHORT SIGNAL {orb.symbol}: price={price:.2f} < ORB-buf={breakout_level:.2f}, "
                f"VWAP={vwap:.2f}, vol={rel_volume:.1f}x, RSI={rsi:.0f}, sent={sentiment:.2f}"
            )

        return all(conditions)

    def _calculate_signal_score(
        self,
        price: float,
        orb: OpeningRange,
        vwap: float,
        rsi: float,
        rel_volume: float,
        macd_histogram: float,
        sentiment: float,
        direction: str,
        last_candle_close: Optional[float] = None
    ) -> float:
        """
        Calculate signal quality score (0-100). Only trade if score >= min_signal_score.

        Scoring breakdown:
        - Breakout strength: 0-25 pts (how far past ORB level)
        - VWAP alignment: 0-15 pts (distance from VWAP in right direction)
        - Volume: 0-20 pts (relative volume strength)
        - RSI: 0-15 pts (favor middle zone, penalize extremes)
        - MACD: 0-15 pts (histogram strength in right direction)
        - Sentiment: 0-10 pts (alignment with trade direction)
        """
        score = 0.0
        buffer = self.config.breakout_buffer_pct

        # BREAKOUT STRENGTH (0-25 pts)
        if direction == 'LONG':
            breakout_level = orb.high * (1 + buffer)
            if price > breakout_level:
                breakout_pct = (price - orb.high) / orb.high * 100
                score += min(breakout_pct * 50, 25)  # 0.5% breakout = 25 pts
        else:  # SHORT
            breakout_level = orb.low * (1 - buffer)
            if price < breakout_level:
                breakout_pct = (orb.low - price) / orb.low * 100
                score += min(breakout_pct * 50, 25)

        # VWAP ALIGNMENT (0-15 pts)
        vwap_dist_pct = abs(price - vwap) / vwap * 100 if vwap > 0 else 0
        if (direction == 'LONG' and price > vwap) or (direction == 'SHORT' and price < vwap):
            score += min(vwap_dist_pct * 15, 15)

        # VOLUME (0-20 pts)
        if rel_volume >= 2.5:
            score += 20
        elif rel_volume >= 2.0:
            score += 15
        elif rel_volume >= 1.5:
            score += 10
        elif rel_volume >= 1.2:
            score += 5

        # RSI (0-15 pts) - reward middle zone, penalize extremes
        if direction == 'LONG':
            if 40 <= rsi <= 60:
                score += 15  # Sweet spot
            elif 30 <= rsi <= 70:
                score += 10  # Acceptable
            elif rsi < 30:
                score += 5   # Oversold can bounce
            # rsi > 70: 0 pts (overbought)
        else:  # SHORT
            if 40 <= rsi <= 60:
                score += 15  # Sweet spot
            elif 30 <= rsi <= 70:
                score += 10  # Acceptable
            elif rsi > 70:
                score += 5   # Overbought can drop
            # rsi < 30: 0 pts (oversold)

        # MACD (0-15 pts)
        if direction == 'LONG' and macd_histogram > 0:
            score += min(abs(macd_histogram) * 100, 15)
        elif direction == 'SHORT' and macd_histogram < 0:
            score += min(abs(macd_histogram) * 100, 15)

        # SENTIMENT (0-10 pts)
        if direction == 'LONG':
            # sentiment ranges from -1 to +1, convert to 0-10 pts
            score += max(0, min((sentiment + 1) * 5, 10))
        else:  # SHORT
            # Negative sentiment is good for shorts
            score += max(0, min((1 - sentiment) * 5, 10))

        return score

    def classify_signal_quality(self, score: float) -> str:
        """
        Classify signal by quality level based on score.

        Args:
            score: Signal score (0-100)

        Returns:
            Quality level: 'ÓPTIMA', 'BUENA', 'REGULAR', or 'DÉBIL'
        """
        if score >= 70:
            return 'ÓPTIMA'
        elif score >= 55:
            return 'BUENA'
        elif score >= 40:
            return 'REGULAR'
        else:
            return 'DÉBIL'

    def _check_breakout_with_scoring(
        self,
        symbol: str,
        price: float,
        orb: OpeningRange,
        vwap: float,
        rsi: float,
        rel_volume: float,
        macd_histogram: float,
        sentiment: float,
        last_candle_close: Optional[float] = None
    ) -> Optional[tuple[str, float]]:
        """
        Check for breakout using soft scoring system (Phase 4).

        Returns all signals without filtering - classification shown to user.

        Returns:
            Tuple of (direction, score) if breakout detected, None otherwise
        """
        buffer = self.config.breakout_buffer_pct

        # Check LONG breakout
        long_breakout_level = orb.high * (1 + buffer)
        long_breakout = price > long_breakout_level

        # If require_candle_close, also check last completed candle
        if self.config.require_candle_close and last_candle_close is not None:
            long_breakout = long_breakout and last_candle_close > long_breakout_level

        # Check SHORT breakout
        short_breakout_level = orb.low * (1 - buffer)
        short_breakout = price < short_breakout_level

        if self.config.require_candle_close and last_candle_close is not None:
            short_breakout = short_breakout and last_candle_close < short_breakout_level

        # No breakout detected
        if not long_breakout and not short_breakout:
            return None

        # Calculate scores and return signal (no filtering - show all to user)
        if long_breakout:
            long_score = self._calculate_signal_score(
                price, orb, vwap, rsi, rel_volume, macd_histogram, sentiment, 'LONG', last_candle_close
            )
            quality = self.classify_signal_quality(long_score)
            logger.info(
                f"🟢 LONG {symbol}: score={long_score:.0f}/100 [{quality}], "
                f"price=${price:.2f} > ORB+buf=${long_breakout_level:.2f}, "
                f"vol={rel_volume:.1f}x, RSI={rsi:.0f}"
            )
            return ('LONG', long_score)

        if short_breakout:
            short_score = self._calculate_signal_score(
                price, orb, vwap, rsi, rel_volume, macd_histogram, sentiment, 'SHORT', last_candle_close
            )
            quality = self.classify_signal_quality(short_score)
            logger.info(
                f"🔴 SHORT {symbol}: score={short_score:.0f}/100 [{quality}], "
                f"price=${price:.2f} < ORB-buf=${short_breakout_level:.2f}, "
                f"vol={rel_volume:.1f}x, RSI={rsi:.0f}"
            )
            return ('SHORT', short_score)

        return None

    def _calculate_hybrid_stop(
        self,
        symbol: str,
        entry_price: float,
        orb: OpeningRange,
        direction: str
    ) -> float:
        """
        Phase 5: Calculate hybrid stop using tighter of ORB opposite or ATR-based.

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            orb: Opening Range data
            direction: 'LONG' or 'SHORT'

        Returns:
            Stop loss price
        """
        # Get ATR for the symbol
        bars = market_data.get_bars(symbol, limit=20)
        if bars.empty or len(bars) < 14:
            # Fallback to ORB-based stop if no ATR data
            return orb.low if direction == 'LONG' else orb.high

        atr_series = calculate_atr(bars)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0

        atr_multiplier = self.config.stop_atr_multiplier

        if direction == 'LONG':
            orb_stop = orb.low
            atr_stop = entry_price - (atr * atr_multiplier)
            # Use the tighter (higher) stop for longs
            stop_loss = max(orb_stop, atr_stop)
            logger.debug(
                f"{symbol} LONG stop: ORB=${orb_stop:.2f}, ATR=${atr_stop:.2f}, "
                f"using ${stop_loss:.2f} (tighter)"
            )
        else:  # SHORT
            orb_stop = orb.high
            atr_stop = entry_price + (atr * atr_multiplier)
            # Use the tighter (lower) stop for shorts
            stop_loss = min(orb_stop, atr_stop)
            logger.debug(
                f"{symbol} SHORT stop: ORB=${orb_stop:.2f}, ATR=${atr_stop:.2f}, "
                f"using ${stop_loss:.2f} (tighter)"
            )

        return stop_loss

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
        sentiment: float = 0.0,
        signal_score: float = 0.0
    ) -> TradeSignal:
        """Create a LONG trade signal"""
        # Phase 5: Use hybrid stop (tighter of ORB or ATR-based)
        stop_loss = self._calculate_hybrid_stop(symbol, entry_price, orb, 'LONG')
        risk_per_share = entry_price - stop_loss
        take_profit = entry_price + (risk_per_share * self.config.reward_risk_ratio)

        position_size, risk_amount = self._calculate_position_size_kelly(
            entry_price, stop_loss
        )

        quality_level = self.classify_signal_quality(signal_score)

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
            sentiment_score=sentiment,
            signal_score=signal_score,
            quality_level=quality_level
        )

        self.signals_today.append(signal)
        logger.info(f"LONG signal generated [{quality_level}]: {signal}")

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
        sentiment: float = 0.0,
        signal_score: float = 0.0
    ) -> TradeSignal:
        """Create a SHORT trade signal"""
        # Phase 5: Use hybrid stop (tighter of ORB or ATR-based)
        stop_loss = self._calculate_hybrid_stop(symbol, entry_price, orb, 'SHORT')
        risk_per_share = stop_loss - entry_price
        take_profit = entry_price - (risk_per_share * self.config.reward_risk_ratio)

        position_size, risk_amount = self._calculate_position_size_kelly(
            entry_price, stop_loss
        )

        quality_level = self.classify_signal_quality(signal_score)

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
            sentiment_score=sentiment,
            signal_score=signal_score,
            quality_level=quality_level
        )

        self.signals_today.append(signal)
        logger.info(f"SHORT signal generated [{quality_level}]: {signal}")

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

        # Phase 6: Update daily P/L tracking
        self.daily_pnl += pnl

        # Phase 6: Track consecutive losses
        if result.won:
            self.consecutive_losses = 0  # Reset on win
        else:
            self.consecutive_losses += 1

        logger.info(
            f"Trade recorded: {symbol} {'WIN' if result.won else 'LOSS'} "
            f"PnL: {pnl_pct*100:.2f}% | Win rate: {self.win_rate:.1%} | "
            f"Daily P/L: ${self.daily_pnl:.2f} | Consec losses: {self.consecutive_losses}"
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
        # Phase 6: Reset risk management tracking
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        logger.info("Daily data reset")

    @property
    def signal_level(self):
        """Get current signal sensitivity level"""
        return self.config.signal_level

    def set_signal_level(self, level) -> bool:
        """
        Change signal sensitivity level at runtime

        Args:
            level: SignalLevel enum or string ('STRICT', 'MODERATE', 'RELAXED')

        Returns:
            True if level was changed successfully
        """
        from config.settings import SignalLevel, SIGNAL_LEVEL_CONFIGS

        # Convert string to enum if needed
        if isinstance(level, str):
            try:
                level = SignalLevel(level.upper())
            except ValueError:
                logger.error(f"Invalid signal level: {level}")
                return False

        if level not in SIGNAL_LEVEL_CONFIGS:
            logger.error(f"Invalid signal level: {level}")
            return False

        old_level = self.config.signal_level
        self.config.signal_level = level

        logger.info(
            f"Signal level changed: {old_level.value} -> {level.value}\n"
            f"  min_signal_score: {self.config.min_signal_score}\n"
            f"  min_relative_volume: {self.config.min_relative_volume}\n"
            f"  latest_trade_time: {self.config.latest_trade_time}\n"
            f"  min_orb_range_pct: {self.config.min_orb_range_pct}%"
        )

        return True

    def get_signal_level_info(self) -> dict:
        """Get current signal level and its configuration"""
        config = self.config.signal_config
        return {
            'level': self.config.signal_level.value,
            'min_signal_score': config.min_signal_score,
            'min_relative_volume': config.min_relative_volume,
            'min_orb_range_pct': config.min_orb_range_pct,
            'max_orb_range_pct': config.max_orb_range_pct,
            'latest_trade_time': config.latest_trade_time,
            'require_candle_close': config.require_candle_close,
            'min_sentiment_long': config.min_sentiment_long,
            'max_sentiment_short': config.max_sentiment_short,
            'rsi_overbought': config.rsi_overbought,
            'rsi_oversold': config.rsi_oversold,
        }


# Global strategy instance
orb_strategy = ORBStrategy()
