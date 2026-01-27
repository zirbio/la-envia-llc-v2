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
    orb_end: str = "09:45"  # Opening Range period ends
    trading_end: str = "16:00"  # Stop trading, close positions (market close)

    # Opening Range
    orb_period_minutes: int = 15

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

    # Static signal filters (not affected by signal level)
    rsi_period: int = 14

    # Breakout confirmation (Phase 1)
    breakout_buffer_pct: float = 0.001  # 0.1% buffer above/below ORB levels

    # Execution (Phase 5)
    use_limit_entry: bool = True
    limit_entry_buffer_pct: float = 0.001  # 0.1%
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
    base_monitoring_interval: int = 10     # Base interval in seconds
    min_monitoring_interval: int = 5       # Minimum when active
    max_monitoring_interval: int = 30      # Maximum when quiet

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

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            alpaca=AlpacaConfig(),
            telegram=TelegramConfig(),
            trading=TradingConfig(),
            sentiment=SentimentConfig(),
            oneoff=OneOffStrategiesConfig()
        )


# Global settings instance
settings = Settings.load()
