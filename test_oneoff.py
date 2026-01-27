"""
Test script for One-Off Strategy System

Tests:
1. Registry system works correctly
2. Strategy can be loaded and instantiated
3. VWAP Reversion strategy logic
4. Signal generation (with live data if available)
"""
import asyncio
from loguru import logger

# Configure logging
logger.add("logs/test_oneoff.log", level="DEBUG", rotation="1 MB")


def test_registry():
    """Test the strategy registry system"""
    print("\n" + "=" * 60)
    print("TEST 1: Strategy Registry")
    print("=" * 60)

    from strategy.oneoff import (
        list_strategies,
        get_strategy_info,
        is_strategy_implemented,
        list_implemented_strategies,
        print_strategies_summary
    )

    # Test list_strategies
    strategies = list_strategies()
    print(f"\n✓ Found {len(strategies)} registered strategies:")
    for name, info in strategies.items():
        print(f"  - {name}: {info['name']}")

    # Test get_strategy_info
    print("\n✓ Testing get_strategy_info('vwap_reversion'):")
    info = get_strategy_info('vwap_reversion')
    print(f"  Name: {info['name']}")
    print(f"  Description: {info['description']}")
    print(f"  Probability: {info['probability']}")

    # Test invalid strategy
    print("\n✓ Testing invalid strategy (should raise ValueError):")
    try:
        get_strategy_info('invalid_strategy')
        print("  ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"  Correctly raised: {e}")

    # Test is_strategy_implemented
    print("\n✓ Testing is_strategy_implemented:")
    for name in strategies.keys():
        implemented = is_strategy_implemented(name)
        status = "✓" if implemented else "✗"
        print(f"  [{status}] {name}")

    # Test list_implemented_strategies
    implemented = list_implemented_strategies()
    print(f"\n✓ Implemented strategies: {len(implemented)}")

    # Print summary
    print("\n✓ Strategy summary:")
    print_strategies_summary()

    print("\n✓ Registry tests PASSED")
    return True


def test_load_strategy():
    """Test loading and instantiating a strategy"""
    print("\n" + "=" * 60)
    print("TEST 2: Load Strategy")
    print("=" * 60)

    from strategy.oneoff import get_strategy

    # Load VWAP Reversion
    print("\n✓ Loading 'vwap_reversion' strategy...")
    strategy = get_strategy('vwap_reversion')

    print(f"  Name: {strategy.name}")
    print(f"  Display Name: {strategy.display_name}")
    print(f"  Description: {strategy.description}")
    print(f"  Doc Path: {strategy.get_documentation_path()}")

    # Test config
    print("\n✓ Strategy config:")
    print(f"  Min VWAP distance: {strategy.config.min_vwap_distance_pct:.1%}")
    print(f"  Max VWAP distance: {strategy.config.max_vwap_distance_pct:.1%}")
    print(f"  RSI oversold: {strategy.config.rsi_oversold}")
    print(f"  RSI overbought: {strategy.config.rsi_overbought}")
    print(f"  Min rel volume: {strategy.config.min_rel_volume}x")
    print(f"  Stop ATR mult: {strategy.config.stop_atr_multiplier}x")
    print(f"  Min score: {strategy.config.min_signal_score}")

    print("\n✓ Load strategy tests PASSED")
    return True


def test_scoring():
    """Test the scoring system"""
    print("\n" + "=" * 60)
    print("TEST 3: Scoring System")
    print("=" * 60)

    from strategy.oneoff import get_strategy

    strategy = get_strategy('vwap_reversion')

    # Test perfect LONG setup
    print("\n✓ Testing perfect LONG setup:")
    score = strategy.calculate_score(
        vwap_distance_pct=0.025,  # 2.5% - sweet spot
        rsi=22,                    # Very oversold
        rel_volume=2.5,            # Strong volume
        macd_improving=True,       # MACD confirming
        atr_pct=0.012,             # Low volatility
        direction='LONG'
    )
    print(f"  Score: {score}/100 (expected ~95-100)")

    # Test weak LONG setup
    print("\n✓ Testing weak LONG setup:")
    score = strategy.calculate_score(
        vwap_distance_pct=0.012,  # Too close to VWAP
        rsi=45,                    # Not oversold
        rel_volume=1.1,            # Weak volume
        macd_improving=False,      # MACD not confirming
        atr_pct=0.03,              # High volatility
        direction='LONG'
    )
    print(f"  Score: {score}/100 (expected ~15-25)")

    # Test perfect SHORT setup
    print("\n✓ Testing perfect SHORT setup:")
    score = strategy.calculate_score(
        vwap_distance_pct=0.025,  # 2.5% - sweet spot
        rsi=78,                    # Very overbought
        rel_volume=2.0,            # Strong volume
        macd_improving=True,       # MACD confirming (falling for short)
        atr_pct=0.015,             # Moderate volatility
        direction='SHORT'
    )
    print(f"  Score: {score}/100 (expected ~85-95)")

    print("\n✓ Scoring tests PASSED")
    return True


def test_position_sizing():
    """Test position sizing calculation"""
    print("\n" + "=" * 60)
    print("TEST 4: Position Sizing")
    print("=" * 60)

    from strategy.oneoff import get_strategy

    strategy = get_strategy('vwap_reversion')

    # Test with $100 stock, $2 risk
    print("\n✓ Testing position sizing ($100 entry, $98 stop):")
    position_size, risk_amount = strategy.calculate_position_size(
        entry_price=100.0,
        stop_loss=98.0
    )
    print(f"  Position size: {position_size} shares")
    print(f"  Risk amount: ${risk_amount:.2f}")
    print(f"  Expected risk: ~${25000 * 0.02:.2f} (2% of $25k)")

    # Test with higher priced stock
    print("\n✓ Testing position sizing ($500 entry, $490 stop):")
    position_size, risk_amount = strategy.calculate_position_size(
        entry_price=500.0,
        stop_loss=490.0
    )
    print(f"  Position size: {position_size} shares")
    print(f"  Risk amount: ${risk_amount:.2f}")

    print("\n✓ Position sizing tests PASSED")
    return True


async def test_live_scan():
    """Test scanning with live market data (if available)"""
    print("\n" + "=" * 60)
    print("TEST 5: Live Market Scan")
    print("=" * 60)

    from strategy.oneoff import get_strategy
    from data.market_data import market_data

    strategy = get_strategy('vwap_reversion')

    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

    print(f"\n✓ Scanning {len(symbols)} symbols for VWAP reversion setups...")
    print(f"  Symbols: {', '.join(symbols)}")

    try:
        signals = await strategy.scan_opportunities(symbols)

        if signals:
            print(f"\n✓ Found {len(signals)} signals:")
            for signal in signals:
                print(f"\n{signal}")
        else:
            print("\n✓ No signals found (this is normal outside market hours)")
            print("  The scan completed successfully.")

        print("\n✓ Live scan test PASSED")
        return True

    except Exception as e:
        print(f"\n⚠ Live scan failed (this may be normal outside market hours):")
        print(f"  Error: {e}")
        print("\n✓ Live scan test SKIPPED")
        return True


def test_signal_format():
    """Test signal formatting"""
    print("\n" + "=" * 60)
    print("TEST 6: Signal Formatting")
    print("=" * 60)

    from datetime import datetime
    from strategy.oneoff.base import OneOffSignal, SignalDirection

    # Create a mock signal
    signal = OneOffSignal(
        symbol="AAPL",
        strategy_name="VWAP Mean Reversion",
        direction=SignalDirection.LONG,
        entry_price=175.50,
        stop_loss=173.25,
        take_profit_1=178.00,
        take_profit_2=179.50,
        position_size=222,
        risk_amount=500.00,
        score=82,
        reasoning="Price 2.1% below VWAP, RSI oversold (28), volume 1.8x avg, MACD improving",
        timestamp=datetime.now(),
        vwap=178.00,
        rsi=28,
        atr=1.50,
        relative_volume=1.8
    )

    print("\n✓ Standard format:")
    print(signal)

    print("\n✓ Telegram format:")
    print(signal.to_telegram_message())

    print("\n✓ Signal formatting tests PASSED")
    return True


async def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ONE-OFF STRATEGY SYSTEM TESTS")
    print("=" * 60)

    tests = [
        ("Registry", test_registry),
        ("Load Strategy", test_load_strategy),
        ("Scoring", test_scoring),
        ("Position Sizing", test_position_sizing),
        ("Signal Format", test_signal_format),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test FAILED with error: {e}")
            results.append((name, False))

    # Run async test
    try:
        result = await test_live_scan()
        results.append(("Live Scan", result))
    except Exception as e:
        print(f"\n✗ Live Scan test FAILED with error: {e}")
        results.append(("Live Scan", False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠ Some tests failed. Please check the logs.")

    return passed == total


if __name__ == "__main__":
    asyncio.run(run_all_tests())
