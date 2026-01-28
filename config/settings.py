"""
Configuration settings for Alpaca ORB Trading Bot
"""
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class SignalLevel(Enum):
    """Signal sensitivity levels for trading"""
    STRICT = "STRICT"      # Conservative: High confidence signals only
    MODERATE = "MODERATE"  # Balanced: Default setting
    RELAXED = "RELAXED"    # Aggressive: More signals, higher risk


class TradingMode(Enum):
    """Trading session modes"""
    REGULAR = "regular"        # Regular hours only (9:30-16:00 EST)
    PREMARKET = "premarket"    # Premarket only (8:00-9:25 EST)
    POSTMARKET = "postmarket"  # Postmarket only (16:05-18:00 EST)
    ALL_SESSIONS = "all"       # All sessions (8:00-18:00 EST)


@dataclass
class SignalLevelConfig:
    """Configuration parameters that vary by signal sensitivity level"""
    # Signal quality
    min_signal_score: float
    min_relative_volume: float

    # ORB range filters
    min_orb_range_pct: float
    max_orb_range_pct: float

    # Trading window
    latest_trade_time: str

    # Confirmation requirements
    require_candle_close: bool

    # Sentiment thresholds
    min_sentiment_long: float
    max_sentiment_short: float

    # RSI thresholds
    rsi_overbought: int
    rsi_oversold: int


@dataclass
class ExtendedHoursConfig:
    """Extended hours trading configuration"""
    # Session enablement
    premarket_enabled: bool = False
    postmarket_enabled: bool = False

    # Premarket trading window (4:00 AM - 9:30 AM, but trade 8:00 - 9:25)
    premarket_trade_start: str = "08:00"
    premarket_trade_end: str = "09:25"

    # Postmarket trading window (4:00 PM - 8:00 PM, but trade 16:05 - 18:00)
    postmarket_trade_start: str = "16:05"
    postmarket_trade_end: str = "18:00"
    postmarket_force_close: str = "19:30"  # Force close all positions

    # Position sizing (multiplier vs regular hours)
    premarket_position_size_mult: float = 0.50   # 50% of regular size
    postmarket_position_size_mult: float = 0.25  # 25% of regular size

    # Stop loss (ATR multiplier - wider stops for extended hours)
    premarket_stop_atr_mult: float = 2.5   # vs 1.5x regular
    postmarket_stop_atr_mult: float = 3.0  # vs 1.5x regular

    # Volume requirements (higher for extended hours)
    premarket_min_rvol: float = 2.5   # vs 1.2x regular
    postmarket_min_rvol: float = 2.5  # vs 1.2x regular

    # Trade limits
    premarket_max_trades: int = 2
    postmarket_max_trades: int = 2

    # Spread limits (max bid-ask spread as %)
    premarket_max_spread_pct: float = 0.005   # 0.5%
    postmarket_max_spread_pct: float = 0.004  # 0.4%

    # Premarket: Gap & Go specific
    premarket_min_gap_pct: float = 3.0  # Higher gap requirement for premarket

    # Postmarket: News/earnings specific
    postmarket_min_move_pct: float = 5.0  # Min movement to consider entry
    postmarket_require_catalyst: bool = True  # Require news catalyst

    # Gap scoring thresholds (premarket)
    # Justification: Higher gaps indicate stronger catalyst/momentum
    gap_score_10pct: int = 30    # 10%+ gap = max points (strong catalyst)
    gap_score_7pct: int = 25     # 7-10% gap = high conviction
    gap_score_5pct: int = 20     # 5-7% gap = moderate conviction
    gap_score_3pct: int = 15     # 3-5% gap = minimum threshold

    # Volume scoring thresholds
    # Justification: Higher RVOL = better liquidity and institutional interest
    rvol_score_5x: int = 30      # 5x+ RVOL = exceptional volume
    rvol_score_3_5x: int = 25    # 3.5-5x RVOL = high volume
    rvol_score_2_5x: int = 20    # 2.5-3.5x RVOL = good volume
    rvol_score_2x: int = 10      # 2-2.5x RVOL = minimum acceptable

    # Spread scoring thresholds
    # Justification: Tighter spreads = lower trading costs
    spread_score_0_1pct: int = 20   # 0.1% spread = excellent liquidity
    spread_score_0_2pct: int = 15   # 0.2% spread = good liquidity
    spread_score_0_3pct: int = 10   # 0.3% spread = acceptable
    spread_score_0_5pct: int = 5    # 0.5% spread = marginal

    # Time to open scoring (premarket only)
    # Justification: Signals closer to open have better follow-through
    time_score_15min: int = 20     # Within 15 min of open
    time_score_30min: int = 15     # Within 30 min of open
    time_score_60min: int = 10     # Within 60 min of open

    # News/earnings scoring (postmarket)
    # Justification: Catalyst type affects reliability of move
    move_score_15pct: int = 40     # 15%+ move = extreme reaction
    move_score_10pct: int = 30     # 10-15% move = strong reaction
    move_score_7pct: int = 20      # 7-10% move = moderate reaction
    move_score_5pct: int = 10      # 5-7% move = minimum threshold

    def __post_init__(self):
        """Validate configuration parameters are within acceptable ranges."""
        # Position size multipliers: must be between 0 and 1 (0-100%)
        if not 0 < self.premarket_position_size_mult <= 1.0:
            raise ValueError(
                f"premarket_position_size_mult must be between 0 and 1.0, got {self.premarket_position_size_mult}"
            )
        if not 0 < self.postmarket_position_size_mult <= 1.0:
            raise ValueError(
                f"postmarket_position_size_mult must be between 0 and 1.0, got {self.postmarket_position_size_mult}"
            )

        # Stop ATR multipliers: reasonable range is 1.0 to 5.0
        if not 1.0 <= self.premarket_stop_atr_mult <= 5.0:
            raise ValueError(
                f"premarket_stop_atr_mult must be between 1.0 and 5.0, got {self.premarket_stop_atr_mult}"
            )
        if not 1.0 <= self.postmarket_stop_atr_mult <= 5.0:
            raise ValueError(
                f"postmarket_stop_atr_mult must be between 1.0 and 5.0, got {self.postmarket_stop_atr_mult}"
            )

        # Trade limits: must be positive
        if self.premarket_max_trades <= 0:
            raise ValueError(
                f"premarket_max_trades must be positive, got {self.premarket_max_trades}"
            )
        if self.postmarket_max_trades <= 0:
            raise ValueError(
                f"postmarket_max_trades must be positive, got {self.postmarket_max_trades}"
            )

        # Spread limits: reasonable range is 0.001 to 0.02 (0.1% to 2%)
        if not 0.001 <= self.premarket_max_spread_pct <= 0.02:
            raise ValueError(
                f"premarket_max_spread_pct should be between 0.001 and 0.02, got {self.premarket_max_spread_pct}. "
                f"Note: 0.5% spread = high trading cost"
            )
        if not 0.001 <= self.postmarket_max_spread_pct <= 0.02:
            raise ValueError(
                f"postmarket_max_spread_pct should be between 0.001 and 0.02, got {self.postmarket_max_spread_pct}. "
                f"Note: 0.5% spread = high trading cost"
            )

        # RVOL requirements: minimum 1.0
        if self.premarket_min_rvol < 1.0:
            raise ValueError(
                f"premarket_min_rvol must be at least 1.0, got {self.premarket_min_rvol}"
            )
        if self.postmarket_min_rvol < 1.0:
            raise ValueError(
                f"postmarket_min_rvol must be at least 1.0, got {self.postmarket_min_rvol}"
            )


# Session-specific parameter lookup
@dataclass
class SessionParams:
    """Parameters for a specific trading session"""
    position_size_mult: float
    stop_atr_mult: float
    min_rvol: float
    max_trades: int
    max_spread_pct: Optional[float]
    trade_start: str
    trade_end: str


def get_session_params(mode: TradingMode, extended_config: 'ExtendedHoursConfig') -> SessionParams:
    """Get trading parameters for a specific session"""
    if mode == TradingMode.PREMARKET:
        return SessionParams(
            position_size_mult=extended_config.premarket_position_size_mult,
            stop_atr_mult=extended_config.premarket_stop_atr_mult,
            min_rvol=extended_config.premarket_min_rvol,
            max_trades=extended_config.premarket_max_trades,
            max_spread_pct=extended_config.premarket_max_spread_pct,
            trade_start=extended_config.premarket_trade_start,
            trade_end=extended_config.premarket_trade_end,
        )
    elif mode == TradingMode.POSTMARKET:
        return SessionParams(
            position_size_mult=extended_config.postmarket_position_size_mult,
            stop_atr_mult=extended_config.postmarket_stop_atr_mult,
            min_rvol=extended_config.postmarket_min_rvol,
            max_trades=extended_config.postmarket_max_trades,
            max_spread_pct=extended_config.postmarket_max_spread_pct,
            trade_start=extended_config.postmarket_trade_start,
            trade_end=extended_config.postmarket_trade_end,
        )
    else:  # REGULAR or ALL_SESSIONS (use regular params)
        return SessionParams(
            position_size_mult=1.0,
            stop_atr_mult=1.5,  # Default from TradingConfig
            min_rvol=1.2,  # Default from MODERATE
            max_trades=3,
            max_spread_pct=None,  # No spread check for regular hours
            trade_start="09:30",
            trade_end="16:00",
        )


# Pre-defined signal level configurations
SIGNAL_LEVEL_CONFIGS: dict[SignalLevel, SignalLevelConfig] = {
    SignalLevel.STRICT: SignalLevelConfig(
        min_signal_score=70.0,
        min_relative_volume=1.5,
        min_orb_range_pct=0.3,
        max_orb_range_pct=2.0,
        latest_trade_time="11:30",
        require_candle_close=True,
        min_sentiment_long=-0.3,
        max_sentiment_short=0.3,
        rsi_overbought=70,
        rsi_oversold=30,
    ),
    SignalLevel.MODERATE: SignalLevelConfig(
        min_signal_score=55.0,
        min_relative_volume=1.2,
        min_orb_range_pct=0.2,
        max_orb_range_pct=2.5,
        latest_trade_time="14:30",
        require_candle_close=True,
        min_sentiment_long=-0.5,
        max_sentiment_short=0.5,
        rsi_overbought=75,
        rsi_oversold=25,
    ),
    SignalLevel.RELAXED: SignalLevelConfig(
        min_signal_score=40.0,
        min_relative_volume=1.0,
        min_orb_range_pct=0.15,
        max_orb_range_pct=3.0,
        latest_trade_time="15:30",
        require_candle_close=False,
        min_sentiment_long=-0.7,
        max_sentiment_short=0.7,
        rsi_overbought=80,
        rsi_oversold=20,
    ),
}


@dataclass
class AlpacaConfig:
    """Alpaca API configuration"""
    api_key: str = os.getenv("ALPACA_API_KEY", "")
    secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    paper: bool = os.getenv("ALPACA_PAPER", "true").lower() == "true"

    @property
    def base_url(self) -> str:
        if self.paper:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"


@dataclass
class TelegramConfig:
    """Telegram bot configuration"""
    bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")


@dataclass
class SentimentConfig:
    """Sentiment analysis configuration"""
    finnhub_api_key: str = os.getenv("FINNHUB_API_KEY", "")
    enabled: bool = os.getenv("SENTIMENT_ENABLED", "true").lower() == "true"
    cache_minutes: int = 30


@dataclass
class TradingConfig:
    """Trading parameters"""
    # Capital
    max_capital: float = 25000.0

    # Risk management
    risk_per_trade: float = 0.02  # 2%
    max_trades_per_day: int = 3
    reward_risk_ratio: float = 2.0  # 2:1

    # Trading hours (EST)
    premarket_start: str = "06:00"
    market_open: str = "09:30"
    orb_end: str = "09:35"  # Opening Range period ends (5 min after open)
    trading_end: str = "16:00"  # Stop trading, close positions (market close)

    # Opening Range
    orb_period_minutes: int = 5  # Reduced from 15 for faster signal detection

    # Scanner filters
    min_gap_percent: float = 2.0
    min_premarket_volume: int = 500_000
    min_price: float = 10.0
    max_price: float = 500.0
    min_avg_volume: int = 1_000_000

    # Signal level - determines dynamic parameters below
    # Load from environment, default to MODERATE
    signal_level: SignalLevel = field(
        default_factory=lambda: SignalLevel(
            os.getenv("SIGNAL_LEVEL", "MODERATE").upper()
        ) if os.getenv("SIGNAL_LEVEL", "MODERATE").upper() in [e.value for e in SignalLevel]
        else SignalLevel.MODERATE
    )

    # Trading mode - determines which sessions are active
    # Load from environment, default to REGULAR
    trading_mode: TradingMode = field(
        default_factory=lambda: TradingMode(
            os.getenv("TRADING_MODE", "regular").lower()
        ) if os.getenv("TRADING_MODE", "regular").lower() in [e.value for e in TradingMode]
        else TradingMode.REGULAR
    )

    # Static signal filters (not affected by signal level)
    rsi_period: int = 14

    # Breakout confirmation (Phase 1)
    breakout_buffer_pct: float = 0.001  # 0.1% buffer above/below ORB levels

    # Execution (Phase 5)
    use_limit_entry: bool = True
    limit_entry_buffer_pct: float = 0.001  # 0.1%
    limit_order_fill_timeout: int = 30  # seconds to wait for limit order fill
    stop_atr_multiplier: float = 1.5

    # Risk management (Phase 6)
    max_daily_loss: float = 750.0    # 3% of $25k - circuit breaker
    max_consecutive_losses: int = 2  # Cooldown after consecutive losses

    # Watchlist
    max_watchlist_size: int = 10

    # Position Management - Partial Profit Taking
    partial_close_enabled: bool = True
    partial_close_at_r: float = 1.0       # Close 50% at 1R profit
    partial_close_percent: float = 0.50    # Percentage to close
    trailing_stop_enabled: bool = True
    trailing_ema_period: int = 9           # EMA9 for trailing
    trailing_bar_timeframe: int = 5        # 5-minute bars
    position_check_interval: int = 5       # Check positions every 5 seconds

    # Monitoring intervals (for adaptive polling)
    base_monitoring_interval: int = 5      # Base interval in seconds (reduced from 10)
    min_monitoring_interval: int = 2       # Minimum when active (reduced from 5)
    max_monitoring_interval: int = 10      # Maximum when quiet (reduced from 30)

    # Show all signals mode (default: True - show all signals with classification)
    # Set to False to restore previous behavior (only show signals meeting threshold)
    show_all_signals: bool = True

    @property
    def signal_config(self) -> SignalLevelConfig:
        """Get the active signal level configuration"""
        return SIGNAL_LEVEL_CONFIGS[self.signal_level]

    # Convenience properties that delegate to signal_config
    @property
    def min_signal_score(self) -> float:
        return self.signal_config.min_signal_score

    @property
    def min_relative_volume(self) -> float:
        return self.signal_config.min_relative_volume

    @property
    def min_orb_range_pct(self) -> float:
        return self.signal_config.min_orb_range_pct

    @property
    def max_orb_range_pct(self) -> float:
        return self.signal_config.max_orb_range_pct

    @property
    def latest_trade_time(self) -> str:
        return self.signal_config.latest_trade_time

    @property
    def require_candle_close(self) -> bool:
        return self.signal_config.require_candle_close

    @property
    def rsi_overbought(self) -> int:
        return self.signal_config.rsi_overbought

    @property
    def rsi_oversold(self) -> int:
        return self.signal_config.rsi_oversold


@dataclass
class OneOffStrategiesConfig:
    """Configuration for one-off strategies (on-demand, not automated)"""
    # VWAP Mean Reversion
    vwap_min_distance_pct: float = 0.02    # 2% minimum distance from VWAP
    vwap_max_distance_pct: float = 0.03    # 3% maximum
    vwap_rsi_oversold: int = 35
    vwap_rsi_overbought: int = 65
    vwap_min_rel_volume: float = 1.2
    vwap_atr_max_pct: float = 0.025
    vwap_stop_atr_mult: float = 1.5
    vwap_time_stop_minutes: int = 45
    vwap_min_score: float = 60.0


@dataclass
class Settings:
    """Main settings container"""
    alpaca: AlpacaConfig
    telegram: TelegramConfig
    trading: TradingConfig
    sentiment: SentimentConfig
    oneoff: OneOffStrategiesConfig
    extended_hours: ExtendedHoursConfig

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            alpaca=AlpacaConfig(),
            telegram=TelegramConfig(),
            trading=TradingConfig(),
            sentiment=SentimentConfig(),
            oneoff=OneOffStrategiesConfig(),
            extended_hours=ExtendedHoursConfig()
        )


# Global settings instance
settings = Settings.load()
