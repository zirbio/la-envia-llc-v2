# Estrategias One-Off

Estrategias de trading que se ejecutan **bajo demanda** (no automáticas como ORB).

## Concepto

Las estrategias one-off son complementarias a la estrategia principal ORB:
- **Se invocan manualmente** cuando el trader lo decide
- **No corren en loop** automáticamente
- **Cada una tiene condiciones específicas** de entrada/salida
- **Registro centralizado** para fácil descubrimiento

## Uso

```python
from strategy.oneoff import get_strategy, list_strategies

# Ver estrategias disponibles
list_strategies()

# Ejecutar una estrategia específica
strategy = get_strategy('vwap_reversion')
signals = await strategy.scan_opportunities(['AAPL', 'TSLA', 'NVDA'])

for signal in signals:
    print(signal)
```

## Estrategias Disponibles

| ID | Nombre | Probabilidad | Riesgo | Estado |
|----|--------|--------------|--------|--------|
| `vwap_reversion` | VWAP Mean Reversion | 65-70% | Bajo | ✅ Implementada |
| `rsi_extremes` | RSI Extremos + VWAP | 60-65% | Bajo | ⏳ Pendiente |
| `pullback` | Momentum Pullback | 60-65% | Medio | ⏳ Pendiente |
| `gap_fill` | Gap Fill | 55-60% | Medio | ⏳ Pendiente |
| `bollinger_squeeze` | Bollinger Squeeze | 50-55% | Alto | ⏳ Pendiente |

## Documentación Detallada

- [VWAP Mean Reversion](./VWAP_REVERSION.md)
- RSI Extremos (próximamente)
- Momentum Pullback (próximamente)
- Gap Fill (próximamente)
- Bollinger Squeeze (próximamente)

## Arquitectura

```
strategy/oneoff/
├── __init__.py           # Registro de estrategias
├── base.py               # Clase abstracta base
├── vwap_reversion.py     # VWAP Mean Reversion
├── rsi_extremes.py       # RSI Extremos (futuro)
├── pullback.py           # Momentum Pullback (futuro)
├── gap_fill.py           # Gap Fill (futuro)
└── bollinger_squeeze.py  # Bollinger Squeeze (futuro)
```

## Señales

Todas las estrategias generan `OneOffSignal` con:

- `symbol`: Símbolo del activo
- `strategy_name`: Nombre de la estrategia
- `direction`: LONG o SHORT
- `entry_price`: Precio de entrada
- `stop_loss`: Stop loss
- `take_profit_1`: Primer objetivo (50% posición)
- `take_profit_2`: Segundo objetivo (resto)
- `position_size`: Tamaño de posición
- `risk_amount`: Riesgo en dólares
- `score`: Puntuación de calidad (0-100)
- `reasoning`: Explicación del setup

## Crear Nueva Estrategia

1. Crear archivo en `strategy/oneoff/nueva_estrategia.py`
2. Heredar de `OneOffStrategy`
3. Implementar métodos abstractos:
   - `name`, `display_name`, `description`
   - `scan_opportunities()`
   - `calculate_score()`
4. Registrar en `__init__.py` en `ONEOFF_STRATEGIES`
5. Crear documentación en `docs/oneoff/NUEVA_ESTRATEGIA.md`

## Integración con Telegram

Las señales one-off pueden enviarse a Telegram usando:

```python
signal.to_telegram_message()
```

Esto genera un mensaje formateado con HTML para Telegram.
