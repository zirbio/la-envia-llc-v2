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
    enabled: bool = True


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
    trading_end: str = "11:30"  # Stop trading, close positions

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

    # Watchlist
    max_watchlist_size: int = 10


@dataclass
class Settings:
    """Main settings container"""
    alpaca: AlpacaConfig
    telegram: TelegramConfig
    trading: TradingConfig

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            alpaca=AlpacaConfig(),
            telegram=TelegramConfig(),
            trading=TradingConfig()
        )


# Global settings instance
settings = Settings.load()
