"""
Configuration settings for Alpaca ORB Trading Bot
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


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
    min_score_long: float = -0.3  # Minimum sentiment for long trades
    max_score_short: float = 0.3  # Maximum sentiment for short trades
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

    # Signal filters
    min_relative_volume: float = 1.5
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    rsi_period: int = 14

    # Breakout confirmation (Phase 1)
    breakout_buffer_pct: float = 0.001  # 0.1% buffer above/below ORB levels
    require_candle_close: bool = True   # Require candle close for confirmation

    # ORB range filter (Phase 3)
    min_orb_range_pct: float = 0.3   # Min 0.3% range (tradeable movement)
    max_orb_range_pct: float = 2.0   # Max 2.0% range (achievable 2R target)

    # Scoring system (Phase 4)
    min_signal_score: float = 70.0   # Minimum score to trigger signal (0-100)

    # Execution (Phase 5)
    use_limit_entry: bool = True
    limit_entry_buffer_pct: float = 0.001  # 0.1%
    stop_atr_multiplier: float = 1.5

    # Risk management (Phase 6)
    max_daily_loss: float = 750.0    # 3% of $25k - circuit breaker
    max_consecutive_losses: int = 2  # Cooldown after consecutive losses
    latest_trade_time: str = "11:30" # No new trades after this time

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
