# VWAP Mean Reversion Strategy

## Resumen

| Atributo | Valor |
|----------|-------|
| **ID** | `vwap_reversion` |
| **Probabilidad** | 65-70% |
| **Riesgo** | Bajo |
| **Mejor Momento** | Después de 10:30 AM ET |
| **Condiciones Ideales** | Días de rango, sin tendencia clara |

## Concepto

La estrategia aprovecha la tendencia natural del precio a regresar al VWAP (Volume Weighted Average Price). Cuando el precio se aleja significativamente del VWAP (2-3%), hay alta probabilidad de que revierta hacia él.

## Condiciones de Entrada

### LONG (Compra)

```
✓ Precio < VWAP * 0.98 (2%+ por debajo)
✓ RSI < 35 (sobreventa)
✓ Volumen relativo > 1.2x
✓ MACD histogram mejorando (subiendo)
✓ ATR / precio < 2.5% (volatilidad moderada)
```

### SHORT (Venta)

```
✓ Precio > VWAP * 1.02 (2%+ por encima)
✓ RSI > 65 (sobrecompra)
✓ Volumen relativo > 1.2x
✓ MACD histogram empeorando (cayendo)
✓ ATR / precio < 2.5% (volatilidad moderada)
```

## Gestión de Riesgo

| Parámetro | Valor |
|-----------|-------|
| **Stop Loss** | 1.5x ATR desde entrada |
| **Take Profit 1** | VWAP (cerrar 50%) |
| **Take Profit 2** | 1% pasado VWAP (cerrar resto) |
| **Time Stop** | 45 minutos máximo |

### Ejemplo LONG

```
Entrada: $98.00 (2% bajo VWAP de $100)
ATR: $1.50
Stop Loss: $98.00 - ($1.50 * 1.5) = $95.75
Take Profit 1: $100.00 (VWAP)
Take Profit 2: $101.00 (1% sobre VWAP)
```

## Sistema de Scoring (0-100)

| Factor | Puntos | Criterio |
|--------|--------|----------|
| Distancia VWAP | 0-30 | 2-3% = 30pts (sweet spot) |
| RSI extremo | 0-25 | <25 o >75 = 25pts |
| Volumen | 0-20 | >2.0x = 20pts |
| MACD dirección | 0-15 | Confirmando = 15pts |
| ATR bajo | 0-10 | <1.5% = 10pts |

**Mínimo para señal**: 60/100

## Cuándo Usar

### Condiciones Ideales

- Días de rango (no tendencia)
- Después de 10:30 AM ET (evitar volatilidad apertura)
- ATR < 2.5% del precio
- Sin noticias pendientes del activo

### Cuándo Evitar

- Primeros 30 minutos de mercado
- Días de tendencia fuerte
- Antes de earnings/noticias
- ATR > 3% del precio
- Gap > 5%

## Uso

```python
from strategy.oneoff import get_strategy

# Instanciar estrategia
strategy = get_strategy('vwap_reversion')

# Escanear símbolos
signals = await strategy.scan_opportunities(['AAPL', 'MSFT', 'GOOGL'])

# Procesar señales
for signal in signals:
    print(signal)
    # Enviar a Telegram para confirmación
    # await telegram_bot.send_signal(signal)
```

## Configuración

Los parámetros se pueden ajustar en `config/settings.py`:

```python
@dataclass
class OneOffStrategiesConfig:
    # VWAP Mean Reversion
    vwap_min_distance_pct: float = 0.02    # 2% mínimo
    vwap_max_distance_pct: float = 0.03    # 3% máximo
    vwap_rsi_oversold: int = 35
    vwap_rsi_overbought: int = 65
    vwap_min_rel_volume: float = 1.2
    vwap_atr_max_pct: float = 0.025
    vwap_stop_atr_mult: float = 1.5
    vwap_time_stop_minutes: int = 45
    vwap_min_score: float = 60.0
```

O directamente al instanciar:

```python
from strategy.oneoff.vwap_reversion import VWAPReversionStrategy, VWAPReversionConfig

config = VWAPReversionConfig(
    min_vwap_distance_pct=0.015,  # Más agresivo
    rsi_oversold=40,
    min_signal_score=55.0
)

strategy = VWAPReversionStrategy(config=config)
```

## Indicadores Utilizados

| Indicador | Fuente | Uso |
|-----------|--------|-----|
| VWAP | `data/indicators.py` | Nivel de reversión |
| RSI (14) | `data/indicators.py` | Confirmación sobreventa/sobrecompra |
| MACD | `data/indicators.py` | Dirección del momentum |
| ATR (14) | `data/indicators.py` | Stop loss y filtro volatilidad |
| Volumen Relativo | `data/market_data.py` | Confirmación de interés |

## Backtest Esperado

| Métrica | Objetivo |
|---------|----------|
| Win Rate | > 55% |
| Profit Factor | > 1.3 |
| Avg Win / Avg Loss | > 1.0 |
| Max Drawdown | < 10% |

## Notas

1. **No perseguir**: Si el precio ya está volviendo al VWAP, no entrar
2. **Respetar time stop**: Salir a los 45 min aunque no haya tocado stop
3. **Partial profit**: Cerrar 50% en VWAP, dejar correr el resto
4. **Correlación**: Evitar múltiples posiciones en activos correlacionados
