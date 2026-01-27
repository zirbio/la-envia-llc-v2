"""
One-Off Strategy Registry

Central registry for all one-off (on-demand) trading strategies.
Use list_strategies() to see available strategies and get_strategy()
to instantiate a specific one.

Usage:
    from strategy.oneoff import list_strategies, get_strategy

    # See what's available
    print(list_strategies())

    # Get and use a strategy
    strategy = get_strategy('vwap_reversion')
    signals = await strategy.scan_opportunities(['AAPL', 'TSLA', 'NVDA'])
"""
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from strategy.oneoff.base import OneOffStrategy


# Registry of all available one-off strategies
# Each entry contains metadata about the strategy
ONEOFF_STRATEGIES: Dict[str, Dict[str, Any]] = {
    'vwap_reversion': {
        'name': 'VWAP Mean Reversion',
        'description': 'Reverses to VWAP when price deviates 2%+ from it',
        'probability': '65-70%',
        'risk': 'bajo',
        'best_conditions': 'Range days, after 10:30 AM ET',
        'module': 'strategy.oneoff.vwap_reversion',
        'class': 'VWAPReversionStrategy',
        'doc': 'docs/oneoff/VWAP_REVERSION.md'
    },
    'rsi_extremes': {
        'name': 'RSI Extremos + VWAP',
        'description': 'Reversals on RSI <25 or >75 with VWAP confirmation',
        'probability': '60-65%',
        'risk': 'bajo',
        'best_conditions': 'Moderate volatility, volume climax',
        'module': 'strategy.oneoff.rsi_extremes',
        'class': 'RSIExtremesStrategy',
        'doc': 'docs/oneoff/RSI_EXTREMES.md'
    },
    'pullback': {
        'name': 'Momentum Pullback',
        'description': 'First pullback to EMA9 in established trend',
        'probability': '60-65%',
        'risk': 'medio',
        'best_conditions': 'Clear trends, first hour',
        'module': 'strategy.oneoff.pullback',
        'class': 'PullbackStrategy',
        'doc': 'docs/oneoff/PULLBACK.md'
    },
    'gap_fill': {
        'name': 'Gap Fill',
        'description': 'Fade 2-5% gaps without strong catalyst',
        'probability': '55-60%',
        'risk': 'medio',
        'best_conditions': 'Moderate gaps, no earnings/FDA',
        'module': 'strategy.oneoff.gap_fill',
        'class': 'GapFillStrategy',
        'doc': 'docs/oneoff/GAP_FILL.md'
    },
    'bollinger_squeeze': {
        'name': 'Bollinger Squeeze Breakout',
        'description': 'Breakout after band contraction',
        'probability': '50-55%',
        'risk': 'alto',
        'best_conditions': 'Pre-volatility expansion',
        'module': 'strategy.oneoff.bollinger_squeeze',
        'class': 'BollingerSqueezeStrategy',
        'doc': 'docs/oneoff/BOLLINGER_SQUEEZE.md'
    }
}


def list_strategies() -> Dict[str, Dict[str, Any]]:
    """
    List all available one-off strategies with their metadata.

    Returns:
        Dictionary mapping strategy IDs to their metadata
    """
    return ONEOFF_STRATEGIES


def get_strategy_info(name: str) -> Dict[str, Any]:
    """
    Get metadata for a specific strategy.

    Args:
        name: Strategy identifier (e.g., 'vwap_reversion')

    Returns:
        Strategy metadata dictionary

    Raises:
        ValueError: If strategy not found
    """
    if name not in ONEOFF_STRATEGIES:
        available = list(ONEOFF_STRATEGIES.keys())
        raise ValueError(
            f"Strategy '{name}' not found. "
            f"Available strategies: {available}"
        )

    return ONEOFF_STRATEGIES[name]


def get_strategy(name: str) -> "OneOffStrategy":
    """
    Load and instantiate a strategy by name.

    Args:
        name: Strategy identifier (e.g., 'vwap_reversion')

    Returns:
        Instantiated strategy object

    Raises:
        ValueError: If strategy not found
        ImportError: If strategy module cannot be loaded
    """
    if name not in ONEOFF_STRATEGIES:
        available = list(ONEOFF_STRATEGIES.keys())
        raise ValueError(
            f"Strategy '{name}' not found. "
            f"Available strategies: {available}"
        )

    config = ONEOFF_STRATEGIES[name]

    try:
        # Dynamically import the strategy module
        module = __import__(config['module'], fromlist=[config['class']])
        strategy_class = getattr(module, config['class'])
        return strategy_class()
    except ImportError as e:
        raise ImportError(
            f"Could not import strategy '{name}' from "
            f"{config['module']}: {e}"
        )
    except AttributeError as e:
        raise ImportError(
            f"Strategy class '{config['class']}' not found in "
            f"{config['module']}: {e}"
        )


def is_strategy_implemented(name: str) -> bool:
    """
    Check if a strategy is actually implemented (not just registered).

    Args:
        name: Strategy identifier

    Returns:
        True if strategy module can be imported
    """
    if name not in ONEOFF_STRATEGIES:
        return False

    config = ONEOFF_STRATEGIES[name]
    try:
        __import__(config['module'], fromlist=[config['class']])
        return True
    except (ImportError, AttributeError):
        return False


def list_implemented_strategies() -> Dict[str, Dict[str, Any]]:
    """
    List only strategies that are actually implemented.

    Returns:
        Dictionary of implemented strategies
    """
    return {
        name: info for name, info in ONEOFF_STRATEGIES.items()
        if is_strategy_implemented(name)
    }


# Convenience function to print strategy summary
def print_strategies_summary() -> None:
    """Print a formatted summary of all strategies"""
    print("\n" + "=" * 60)
    print("ONE-OFF STRATEGIES AVAILABLE")
    print("=" * 60)

    for name, info in ONEOFF_STRATEGIES.items():
        implemented = "✓" if is_strategy_implemented(name) else "✗"
        print(f"\n[{implemented}] {name}")
        print(f"    Name: {info['name']}")
        print(f"    Description: {info['description']}")
        print(f"    Win Rate: {info['probability']}")
        print(f"    Risk: {info['risk']}")
        print(f"    Best Conditions: {info['best_conditions']}")

    print("\n" + "=" * 60)
    print("Use: get_strategy('strategy_name') to instantiate")
    print("=" * 60 + "\n")
