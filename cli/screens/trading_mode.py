"""
Trading Mode Selection Screen

Detailed trading mode selection and confirmation.
"""
from typing import Optional
from datetime import datetime
import pytz

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

from config.settings import TradingMode, settings


EST = pytz.timezone('US/Eastern')
MADRID = pytz.timezone('Europe/Madrid')


def convert_est_to_madrid(est_time_str: str) -> str:
    """
    Convert an EST time string to Madrid time.

    Args:
        est_time_str: Time string like "9:30 AM" or "8:00"

    Returns:
        Time string in Madrid timezone
    """
    # Clean up the time string
    time_str = est_time_str.strip().upper()
    is_pm = 'PM' in time_str
    is_am = 'AM' in time_str
    time_str = time_str.replace('AM', '').replace('PM', '').strip()

    # Parse hour and minute
    parts = time_str.split(':')
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0

    # Convert 12-hour to 24-hour if needed
    if is_pm and hour != 12:
        hour += 12
    elif is_am and hour == 12:
        hour = 0

    # Create EST datetime and convert to Madrid
    now = datetime.now(EST)
    est_dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    madrid_dt = est_dt.astimezone(MADRID)

    return madrid_dt.strftime('%H:%M')


def format_dual_hours(est_hours: str) -> str:
    """
    Format hours string with both EST and Madrid times.

    Args:
        est_hours: Hours string like "9:30 AM - 4:00 PM EST"

    Returns:
        String like "9:30 - 16:00 EST (15:30 - 22:00 Madrid)"
    """
    # Remove "EST" suffix if present
    hours = est_hours.replace('EST', '').strip()

    # Split start and end times
    if ' - ' in hours:
        start, end = hours.split(' - ')
        start_madrid = convert_est_to_madrid(start)
        end_madrid = convert_est_to_madrid(end)
        return f"{hours} EST ({start_madrid} - {end_madrid} Madrid)"

    return est_hours


class TradingModeScreen:
    """Trading mode selection and info screen"""

    def __init__(self, console: Console = None):
        self.console = console

    def display_mode_details(self, mode: TradingMode):
        """Display detailed information about a trading mode"""
        mode_info = self._get_mode_info(mode)

        if not RICH_AVAILABLE or self.console is None:
            self._display_basic(mode_info)
        else:
            self._display_rich(mode_info)

    def _get_mode_info(self, mode: TradingMode) -> dict:
        """Get detailed info for a trading mode"""
        ext_config = settings.extended_hours

        mode_configs = {
            TradingMode.REGULAR: {
                'name': 'Regular Hours',
                'emoji': '📊',
                'hours': '9:30 - 16:00 EST (15:30 - 22:00 Madrid)',
                'strategy': 'Opening Range Breakout (ORB)',
                'position_size': '100%',
                'stop_mult': '1.5x ATR',
                'min_rvol': '1.2x',
                'max_trades': '3',
                'description': 'Trading durante horas regulares del mercado. '
                             'Máxima liquidez y spreads más estrechos.'
            },
            TradingMode.PREMARKET: {
                'name': 'Premarket',
                'emoji': '🌅',
                'hours': f'{ext_config.premarket_trade_start} - {ext_config.premarket_trade_end} EST (14:00 - 15:25 Madrid)',
                'strategy': 'Gap & Go (Momentum)',
                'position_size': f'{ext_config.premarket_position_size_mult*100:.0f}%',
                'stop_mult': f'{ext_config.premarket_stop_atr_mult}x ATR',
                'min_rvol': f'{ext_config.premarket_min_rvol}x',
                'max_trades': str(ext_config.premarket_max_trades),
                'spread_max': f'{ext_config.premarket_max_spread_pct*100:.1f}%',
                'description': 'Trading pre-apertura. Enfoque en gaps significativos (3%+). '
                             'Solo órdenes límite. Menor liquidez requiere tamaños reducidos.'
            },
            TradingMode.POSTMARKET: {
                'name': 'Postmarket',
                'emoji': '🌙',
                'hours': f'{ext_config.postmarket_trade_start} - {ext_config.postmarket_trade_end} EST (22:05 - 00:00 Madrid)',
                'strategy': 'Earnings & News Reactions',
                'position_size': f'{ext_config.postmarket_position_size_mult*100:.0f}%',
                'stop_mult': f'{ext_config.postmarket_stop_atr_mult}x ATR',
                'min_rvol': f'{ext_config.postmarket_min_rvol}x',
                'max_trades': str(ext_config.postmarket_max_trades),
                'spread_max': f'{ext_config.postmarket_max_spread_pct*100:.1f}%',
                'force_close': ext_config.postmarket_force_close,
                'description': 'Trading post-cierre. Reacciones a earnings y noticias. '
                             'Requiere movimiento mínimo 5%. Cierre forzado a las 19:30 EST (01:30 Madrid).'
            },
            TradingMode.ALL_SESSIONS: {
                'name': 'Todas las Sesiones',
                'emoji': '🔄',
                'hours': f'{ext_config.premarket_trade_start} - {ext_config.postmarket_trade_end} EST (14:00 - 00:00 Madrid)',
                'strategy': 'Gap & Go → ORB → Earnings',
                'position_size': 'Variable por sesión',
                'description': 'Trading en todas las sesiones. Cambia automáticamente de '
                             'estrategia según la hora. Máxima oportunidad, mayor gestión.'
            }
        }

        return mode_configs.get(mode, {})

    def _display_basic(self, info: dict):
        """Display mode details without Rich"""
        print(f"\n{info.get('emoji', '')} {info.get('name', 'Unknown')}")
        print("=" * 50)
        print(f"Horario: {info.get('hours', 'N/A')}")
        print(f"Estrategia: {info.get('strategy', 'N/A')}")
        print(f"Tamaño posición: {info.get('position_size', 'N/A')}")
        print(f"Stop multiplier: {info.get('stop_mult', 'N/A')}")
        print(f"Min RVOL: {info.get('min_rvol', 'N/A')}")
        print(f"Max trades: {info.get('max_trades', 'N/A')}")
        if 'spread_max' in info:
            print(f"Spread máximo: {info['spread_max']}")
        if 'force_close' in info:
            print(f"Cierre forzado: {info['force_close']}")
        print(f"\n{info.get('description', '')}")

    def _display_rich(self, info: dict):
        """Display mode details with Rich"""
        # Create details table
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Parámetro", style="dim", width=20)
        table.add_column("Valor", style="bold")

        table.add_row("Horario", info.get('hours', 'N/A'))
        table.add_row("Estrategia", info.get('strategy', 'N/A'))
        table.add_row("Tamaño posición", info.get('position_size', 'N/A'))

        if 'stop_mult' in info:
            table.add_row("Stop multiplier", info.get('stop_mult', 'N/A'))
        if 'min_rvol' in info:
            table.add_row("Min RVOL", info.get('min_rvol', 'N/A'))
        if 'max_trades' in info:
            table.add_row("Max trades", info.get('max_trades', 'N/A'))
        if 'spread_max' in info:
            table.add_row("Spread máximo", info['spread_max'])
        if 'force_close' in info:
            table.add_row("Cierre forzado", info['force_close'])

        # Create panel with emoji and name
        title = f"{info.get('emoji', '')} {info.get('name', 'Unknown')}"

        panel = Panel(
            table,
            title=f"[bold]{title}[/bold]",
            subtitle=f"[dim]{info.get('description', '')}[/dim]",
            border_style="cyan",
            padding=(1, 2)
        )

        self.console.print(panel)

    def confirm_selection(self, mode: TradingMode) -> bool:
        """
        Show mode details and confirm selection.

        Args:
            mode: Selected trading mode

        Returns:
            True if confirmed, False otherwise
        """
        self.display_mode_details(mode)

        if RICH_AVAILABLE and self.console:
            self.console.print()
            self.console.print("[yellow]¿Iniciar este modo?[/yellow]")
            self.console.print("[dim](S)í / (N)o[/dim]")
            self.console.print()

        if PROMPT_TOOLKIT_AVAILABLE:
            response = prompt("Confirmar [S/N]: ").strip().upper()
        else:
            response = input("\nConfirmar [S/N]: ").strip().upper()

        return response in ('S', 'SI', 'Y', 'YES', '')
