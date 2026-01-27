"""
Strategy Configuration Screen

Display and configure trading strategy parameters.
"""
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from prompt_toolkit import prompt
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

from config.settings import settings, SignalLevel, SIGNAL_LEVEL_CONFIGS


class StrategyScreen:
    """Strategy configuration and display screen"""

    def __init__(self, console: Console = None):
        self.console = console

    def display_current_config(self):
        """Display current strategy configuration"""
        config = settings.trading.signal_config
        level = settings.trading.signal_level

        if not RICH_AVAILABLE or self.console is None:
            self._display_basic(config, level)
        else:
            self._display_rich(config, level)

    def _display_basic(self, config, level: SignalLevel):
        """Display without Rich"""
        print(f"\nCONFIGURACIÓN DE ESTRATEGIA (Nivel: {level.value})")
        print("=" * 50)
        print(f"Min Signal Score: {config.min_signal_score}")
        print(f"Min RVOL: {config.min_relative_volume}x")
        print(f"ORB Range: {config.min_orb_range_pct}% - {config.max_orb_range_pct}%")
        print(f"Latest Trade: {config.latest_trade_time}")
        print(f"RSI Overbought: {config.rsi_overbought}")
        print(f"RSI Oversold: {config.rsi_oversold}")
        print(f"Sentiment Long >=: {config.min_sentiment_long}")
        print(f"Sentiment Short <=: {config.max_sentiment_short}")
        print(f"Require Candle Close: {config.require_candle_close}")

    def _display_rich(self, config, level: SignalLevel):
        """Display with Rich"""
        table = Table(show_header=True, box=box.ROUNDED, border_style="magenta")
        table.add_column("Parámetro", style="dim")
        table.add_column("Valor", style="bold", justify="right")
        table.add_column("Descripción", style="dim")

        table.add_row("Min Signal Score", str(config.min_signal_score), "Puntuación mínima para señal")
        table.add_row("Min RVOL", f"{config.min_relative_volume}x", "Volumen relativo mínimo")
        table.add_row("ORB Range Min", f"{config.min_orb_range_pct}%", "Rango ORB mínimo")
        table.add_row("ORB Range Max", f"{config.max_orb_range_pct}%", "Rango ORB máximo")
        table.add_row("Latest Trade", config.latest_trade_time, "Hora límite para nuevas entradas")
        table.add_row("RSI Overbought", str(config.rsi_overbought), "Límite sobrecompra")
        table.add_row("RSI Oversold", str(config.rsi_oversold), "Límite sobreventa")
        table.add_row("Sentiment Long", f">= {config.min_sentiment_long}", "Sentimiento mín para long")
        table.add_row("Sentiment Short", f"<= {config.max_sentiment_short}", "Sentimiento máx para short")
        table.add_row("Candle Close", str(config.require_candle_close), "Requiere cierre de vela")

        panel = Panel(
            table,
            title=f"[bold]CONFIGURACIÓN ({level.value})[/bold]",
            border_style="magenta",
            padding=(1, 1)
        )
        self.console.print(panel)

    def display_level_comparison(self):
        """Display comparison of all signal levels"""
        if not RICH_AVAILABLE or self.console is None:
            self._display_comparison_basic()
        else:
            self._display_comparison_rich()

    def _display_comparison_basic(self):
        """Display comparison without Rich"""
        print("\nCOMPARACIÓN DE NIVELES")
        print("=" * 70)
        print(f"{'Parámetro':<25} {'STRICT':<15} {'MODERATE':<15} {'RELAXED':<15}")
        print("-" * 70)

        strict = SIGNAL_LEVEL_CONFIGS[SignalLevel.STRICT]
        moderate = SIGNAL_LEVEL_CONFIGS[SignalLevel.MODERATE]
        relaxed = SIGNAL_LEVEL_CONFIGS[SignalLevel.RELAXED]

        print(f"{'Min Signal Score':<25} {strict.min_signal_score:<15} {moderate.min_signal_score:<15} {relaxed.min_signal_score:<15}")
        print(f"{'Min RVOL':<25} {strict.min_relative_volume}x{'':<12} {moderate.min_relative_volume}x{'':<12} {relaxed.min_relative_volume}x")
        print(f"{'ORB Range Min':<25} {strict.min_orb_range_pct}%{'':<13} {moderate.min_orb_range_pct}%{'':<13} {relaxed.min_orb_range_pct}%")
        print(f"{'ORB Range Max':<25} {strict.max_orb_range_pct}%{'':<13} {moderate.max_orb_range_pct}%{'':<13} {relaxed.max_orb_range_pct}%")
        print(f"{'Latest Trade':<25} {strict.latest_trade_time:<15} {moderate.latest_trade_time:<15} {relaxed.latest_trade_time:<15}")
        print(f"{'RSI Overbought':<25} {strict.rsi_overbought:<15} {moderate.rsi_overbought:<15} {relaxed.rsi_overbought:<15}")
        print(f"{'RSI Oversold':<25} {strict.rsi_oversold:<15} {moderate.rsi_oversold:<15} {relaxed.rsi_oversold:<15}")

    def _display_comparison_rich(self):
        """Display comparison with Rich"""
        table = Table(box=box.ROUNDED, border_style="cyan")
        table.add_column("Parámetro", style="dim")
        table.add_column("STRICT", style="red", justify="center")
        table.add_column("MODERATE", style="yellow", justify="center")
        table.add_column("RELAXED", style="green", justify="center")

        strict = SIGNAL_LEVEL_CONFIGS[SignalLevel.STRICT]
        moderate = SIGNAL_LEVEL_CONFIGS[SignalLevel.MODERATE]
        relaxed = SIGNAL_LEVEL_CONFIGS[SignalLevel.RELAXED]

        table.add_row("Min Signal Score", str(strict.min_signal_score), str(moderate.min_signal_score), str(relaxed.min_signal_score))
        table.add_row("Min RVOL", f"{strict.min_relative_volume}x", f"{moderate.min_relative_volume}x", f"{relaxed.min_relative_volume}x")
        table.add_row("ORB Range Min", f"{strict.min_orb_range_pct}%", f"{moderate.min_orb_range_pct}%", f"{relaxed.min_orb_range_pct}%")
        table.add_row("ORB Range Max", f"{strict.max_orb_range_pct}%", f"{moderate.max_orb_range_pct}%", f"{relaxed.max_orb_range_pct}%")
        table.add_row("Latest Trade", strict.latest_trade_time, moderate.latest_trade_time, relaxed.latest_trade_time)
        table.add_row("RSI Overbought", str(strict.rsi_overbought), str(moderate.rsi_overbought), str(relaxed.rsi_overbought))
        table.add_row("RSI Oversold", str(strict.rsi_oversold), str(moderate.rsi_oversold), str(relaxed.rsi_oversold))
        table.add_row("Sentiment Long", str(strict.min_sentiment_long), str(moderate.min_sentiment_long), str(relaxed.min_sentiment_long))
        table.add_row("Sentiment Short", str(strict.max_sentiment_short), str(moderate.max_sentiment_short), str(relaxed.max_sentiment_short))
        table.add_row("Candle Close", str(strict.require_candle_close), str(moderate.require_candle_close), str(relaxed.require_candle_close))

        panel = Panel(
            table,
            title="[bold]COMPARACIÓN DE NIVELES[/bold]",
            border_style="cyan",
            padding=(1, 1)
        )
        self.console.print(panel)

    def select_signal_level(self) -> Optional[SignalLevel]:
        """
        Display signal level selection and return choice.

        Returns:
            Selected SignalLevel or None if cancelled
        """
        current = settings.trading.signal_level

        if not RICH_AVAILABLE or self.console is None:
            print(f"\nSELECCIONE NIVEL DE SEÑAL (Actual: {current.value})")
            print("1. STRICT - Conservador, alta confianza")
            print("2. MODERATE - Balanceado (default)")
            print("3. RELAXED - Agresivo, más señales")
            print("4. Cancelar")
            choice = input("\nSeleccione [1-4]: ").strip()
        else:
            self.display_level_comparison()
            self.console.print()
            self.console.print(f"[dim]Nivel actual: {current.value}[/dim]")
            self.console.print()

            table = Table(show_header=False, box=box.ROUNDED, border_style="green")
            table.add_column("", style="cyan", width=4)
            table.add_column("Nivel", style="bold")
            table.add_column("Descripción", style="dim")

            table.add_row("1", "[red]STRICT[/red]", "Conservador, alta confianza")
            table.add_row("2", "[yellow]MODERATE[/yellow]", "Balanceado (default)")
            table.add_row("3", "[green]RELAXED[/green]", "Agresivo, más señales")
            table.add_row("4", "Cancelar", "")

            self.console.print(Panel(table, title="[bold]SELECCIONE NIVEL[/bold]", border_style="green"))
            self.console.print()

            if PROMPT_TOOLKIT_AVAILABLE:
                choice = prompt("Seleccione [1-4]: ").strip()
            else:
                choice = input("Seleccione [1-4]: ").strip()

        level_map = {
            '1': SignalLevel.STRICT,
            '2': SignalLevel.MODERATE,
            '3': SignalLevel.RELAXED
        }

        return level_map.get(choice)

    def wait_for_input(self):
        """Wait for user input to continue"""
        if RICH_AVAILABLE and self.console:
            self.console.print()

        if PROMPT_TOOLKIT_AVAILABLE:
            prompt("Presione Enter para continuar...")
        else:
            input("\nPresione Enter para continuar...")
