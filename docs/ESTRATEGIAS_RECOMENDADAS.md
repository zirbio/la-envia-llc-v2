# Estrategias de Trading Recomendadas para Alpaca

Este documento presenta las **mejores estrategias de trading intradía** que puedes implementar usando tu infraestructura actual de Alpaca API. Todas utilizan los indicadores y datos que ya tienes disponibles.

---

## Resumen Ejecutivo

| Estrategia | Probabilidad Éxito | Complejidad | Mejor Condición Mercado |
|------------|-------------------|-------------|------------------------|
| **VWAP Mean Reversion** | Alta (65-70%) | Media | Días de rango/consolidación |
| **Momentum Pullback** | Alta (60-65%) | Baja | Tendencias claras |
| **Gap Fill** | Media-Alta (55-60%) | Baja | Gaps sin catalizador fuerte |
| **Bollinger Squeeze Breakout** | Media (50-55%) | Media | Baja volatilidad pre-expansión |
| **RSI Extremos + VWAP** | Alta (60-65%) | Baja | Sobreventa/sobrecompra extrema |

---

## 1. VWAP Mean Reversion Strategy

### Teoría
El VWAP actúa como un "imán" de precio institucional. Cuando el precio se aleja significativamente del VWAP, tiende a regresar a él. Esta estrategia capitaliza esos retornos.

### Condiciones de Entrada LONG
```python
# Usar indicadores existentes de data/indicators.py
conditions_long = {
    'precio_bajo_vwap': price < vwap * 0.98,      # Precio 2%+ bajo VWAP
    'rsi_oversold': rsi < 35,                      # RSI indica sobreventa
    'vwap_cerca': abs(price - vwap) / vwap < 0.03, # No más de 3% de VWAP
    'volumen_confirmacion': rel_volume > 1.2,      # Volumen elevado
    'macd_girando': macd_histogram > prev_histogram, # MACD mejorando
    'no_tendencia_fuerte': atr / price < 0.025,    # ATR bajo (no tendencia fuerte)
}
```

### Condiciones de Entrada SHORT
```python
conditions_short = {
    'precio_sobre_vwap': price > vwap * 1.02,      # Precio 2%+ sobre VWAP
    'rsi_overbought': rsi > 65,                    # RSI indica sobrecompra
    'vwap_cerca': abs(price - vwap) / vwap < 0.03,
    'volumen_confirmacion': rel_volume > 1.2,
    'macd_girando': macd_histogram < prev_histogram, # MACD empeorando
    'no_tendencia_fuerte': atr / price < 0.025,
}
```

### Gestión de Posición
- **Stop Loss**: 1.5x ATR desde entrada
- **Take Profit**: VWAP (objetivo primario)
- **TP Extendido**: Lado opuesto del VWAP (para 50% restante)
- **Time Stop**: 45-60 minutos máximo

### Configuración Recomendada
```python
@dataclass
class VWAPReversionConfig:
    min_vwap_distance_pct: float = 0.02    # 2% mínimo alejado de VWAP
    max_vwap_distance_pct: float = 0.03    # 3% máximo (evita cuchillos)
    rsi_oversold: int = 35
    rsi_overbought: int = 65
    min_rel_volume: float = 1.2
    atr_max_pct: float = 0.025             # Evitar días muy volátiles
    time_stop_minutes: int = 45
    partial_at_vwap: float = 0.50          # Cerrar 50% en VWAP
```

### Mejores Condiciones
- Días de rango (no tendenciales)
- Después de las 10:30 AM ET (menos volatilidad inicial)
- Cuando VIX está estable o bajando

### Score de Calidad (0-100)
```python
def calculate_vwap_reversion_score(price, vwap, rsi, rel_volume, macd_hist, atr):
    score = 0

    # Distancia VWAP (0-30 pts)
    dist_pct = abs(price - vwap) / vwap
    if 0.02 <= dist_pct <= 0.03:
        score += 30  # Sweet spot
    elif 0.015 <= dist_pct <= 0.035:
        score += 20

    # RSI extremo (0-25 pts)
    if rsi < 30 or rsi > 70:
        score += 25
    elif rsi < 35 or rsi > 65:
        score += 15

    # Volumen (0-20 pts)
    if rel_volume >= 1.5:
        score += 20
    elif rel_volume >= 1.2:
        score += 10

    # MACD girando (0-15 pts)
    # Si es long y histogram mejorando, o short y empeorando
    score += 15  # Simplificado

    # ATR bajo = mejor (0-10 pts)
    if atr / price < 0.02:
        score += 10
    elif atr / price < 0.025:
        score += 5

    return score
```

---

## 2. Momentum Pullback Strategy (First Pullback)

### Teoría
En una tendencia establecida, el primer pullback al EMA de corto plazo (9 o 20) ofrece una entrada de bajo riesgo con la tendencia. Es más conservadora que "comprar el breakout".

### Condiciones de Entrada LONG
```python
conditions_long = {
    'tendencia_alcista': price > ema_20 and ema_9 > ema_20,
    'pullback_a_ema': low <= ema_9 * 1.005,        # Toca o cerca de EMA9
    'rebote_confirmado': close > open,              # Vela verde en pullback
    'rsi_no_extremo': 40 < rsi < 65,               # RSI saludable
    'volumen_en_pullback': rel_volume < 1.0,       # Vol bajo en pullback (ideal)
    'vwap_soporte': price > vwap,                  # VWAP como soporte
    'macd_positivo': macd > 0,                     # MACD sobre cero
}
```

### Condiciones de Entrada SHORT
```python
conditions_short = {
    'tendencia_bajista': price < ema_20 and ema_9 < ema_20,
    'pullback_a_ema': high >= ema_9 * 0.995,
    'rechazo_confirmado': close < open,            # Vela roja en pullback
    'rsi_no_extremo': 35 < rsi < 60,
    'volumen_en_pullback': rel_volume < 1.0,
    'vwap_resistencia': price < vwap,
    'macd_negativo': macd < 0,
}
```

### Gestión de Posición
- **Stop Loss**: Bajo el low de la vela de pullback (LONG) o sobre el high (SHORT)
- **Stop Alternativo**: 1.2x ATR desde entrada
- **TP1**: Nuevo High/Low del día (HOD/LOD)
- **TP2**: 2x el riesgo inicial
- **Trailing**: EMA9 en timeframe de 5 minutos

### Configuración Recomendada
```python
@dataclass
class PullbackConfig:
    ema_fast: int = 9
    ema_slow: int = 20
    pullback_tolerance_pct: float = 0.005   # 0.5% de EMA
    rsi_min_long: int = 40
    rsi_max_long: int = 65
    rsi_min_short: int = 35
    rsi_max_short: int = 60
    max_pullback_volume: float = 1.0        # Volumen bajo en pullback
    stop_atr_multiplier: float = 1.2
```

### Filtros Adicionales
```python
# Evitar si:
avoid_conditions = {
    'gap_muy_grande': gap_pct > 5.0,           # Gap extremo = riesgo de reversión
    'despues_de_11_30': current_time > time(11, 30),  # Momentum suele agotarse
    'earnings_hoy': has_earnings_today,
    'cerca_de_resistencia': price > pdh * 0.99,  # PDH como resistencia
}
```

---

## 3. Gap Fill Strategy

### Teoría
Los gaps tienden a "llenarse" (regresar al precio de cierre anterior) cuando no hay un catalizador fundamental fuerte. Esta estrategia opera contra el gap esperando el fill.

### Condiciones de Entrada (Fade Gap Up)
```python
conditions_fade_gap_up = {
    'gap_moderado': 2.0 <= gap_pct <= 5.0,        # Gap 2-5%
    'sin_catalizador_fuerte': sentiment_score < 0.5,  # No muy alcista
    'rsi_overbought': rsi > 65,
    'precio_bajo_premarket_high': price < pmh,    # No hacer nuevos highs
    'volumen_decreciente': rel_volume < rel_volume_5min_ago,
    'vwap_perdido': price < vwap,                 # Pierde VWAP
    'primera_hora': minutes_since_open < 60,
}
```

### Condiciones de Entrada (Fade Gap Down)
```python
conditions_fade_gap_down = {
    'gap_moderado': -5.0 <= gap_pct <= -2.0,
    'sin_catalizador_fuerte': sentiment_score > -0.5,
    'rsi_oversold': rsi < 35,
    'precio_sobre_premarket_low': price > pml,
    'volumen_decreciente': rel_volume < rel_volume_5min_ago,
    'vwap_recuperado': price > vwap,
    'primera_hora': minutes_since_open < 60,
}
```

### Gestión de Posición
- **Stop Loss**: PMH + 0.5% (gap up fade) o PML - 0.5% (gap down fade)
- **TP1**: 50% del gap (fill parcial)
- **TP2**: Cierre del día anterior (100% fill)
- **Time Stop**: Si no llena 50% en 90 minutos, salir

### Configuración Recomendada
```python
@dataclass
class GapFillConfig:
    min_gap_pct: float = 2.0
    max_gap_pct: float = 5.0           # Gaps muy grandes no suelen llenar
    max_sentiment_for_fade: float = 0.5
    min_sentiment_for_fade: float = -0.5
    rsi_overbought: int = 65
    rsi_oversold: int = 35
    partial_fill_target: float = 0.50  # Objetivo de 50% del gap
    full_fill_target: float = 1.00
    time_stop_minutes: int = 90
```

### Filtros Críticos (NO operar si)
```python
avoid_gap_fade = {
    'catalizador_fuerte': has_earnings or has_fda or has_merger,
    'gap_muy_grande': abs(gap_pct) > 7.0,
    'tendencia_sector': sector_etf_gap_same_direction > 2.0,
    'volumen_premarket_extremo': premarket_volume > avg_daily_volume * 0.5,
    'sentiment_extremo': abs(sentiment_score) > 0.7,
}
```

---

## 4. Bollinger Band Squeeze Breakout

### Teoría
Cuando las Bandas de Bollinger se contraen (baja volatilidad), frecuentemente precede una expansión de volatilidad. El squeeze indica "carga" de energía lista para explotar en dirección.

### Detección de Squeeze
```python
def detect_bollinger_squeeze(df, threshold_pct=0.03):
    """
    Detecta cuando el ancho de las bandas es menor al threshold
    """
    upper, middle, lower = calculate_bollinger_bands(df['close'])
    band_width = (upper - lower) / middle

    # Squeeze si el ancho está en el 20% inferior de los últimos 50 períodos
    band_width_percentile = band_width.rolling(50).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )

    is_squeeze = band_width_percentile < 0.20
    return is_squeeze, band_width
```

### Condiciones de Entrada LONG
```python
conditions_long = {
    'squeeze_activo': is_squeeze,
    'breakout_superior': close > bb_upper,          # Rompe banda superior
    'volumen_spike': rel_volume > 1.8,              # Volumen confirma
    'macd_positivo': macd_histogram > 0,
    'ema_alineadas': ema_9 > ema_20,
    'vwap_soporte': price > vwap,
}
```

### Condiciones de Entrada SHORT
```python
conditions_short = {
    'squeeze_activo': is_squeeze,
    'breakdown_inferior': close < bb_lower,
    'volumen_spike': rel_volume > 1.8,
    'macd_negativo': macd_histogram < 0,
    'ema_alineadas': ema_9 < ema_20,
    'vwap_resistencia': price < vwap,
}
```

### Gestión de Posición
- **Stop Loss**: Banda media (SMA 20)
- **TP1**: 1.5x el ancho de la banda
- **TP2**: 2.5x el ancho de la banda (trailing)

### Configuración Recomendada
```python
@dataclass
class BollingerSqueezeConfig:
    bb_period: int = 20
    bb_std: float = 2.0
    squeeze_percentile: float = 0.20     # Top 20% más estrecho
    squeeze_lookback: int = 50
    min_breakout_volume: float = 1.8
    stop_to_middle_band: bool = True
    tp1_band_width_mult: float = 1.5
    tp2_band_width_mult: float = 2.5
```

### Mejor Momento
- Después de períodos de consolidación (30+ minutos de rango)
- Pre-noticias o eventos conocidos
- Primeras horas de la sesión (9:45-11:00 AM)

---

## 5. RSI Extremos + Confirmación VWAP

### Teoría
RSI en niveles extremos (<25 o >75) combinado con confirmación de VWAP ofrece entradas de alta probabilidad para reversiones de corto plazo.

### Condiciones de Entrada LONG (RSI Oversold)
```python
conditions_long = {
    'rsi_extremo': rsi < 25,                       # RSI muy bajo
    'rsi_girando': rsi > prev_rsi,                 # RSI empezando a subir
    'vwap_cerca': abs(price - vwap) / vwap < 0.02, # Cerca de VWAP
    'stoch_oversold': stoch_k < 20 and stoch_d < 20,
    'stoch_cruzando': stoch_k > stoch_d,          # %K cruza sobre %D
    'no_breakdown': price > bb_lower,              # No rompió BB inferior
    'volumen_climax': rel_volume > 2.0,           # Posible capitulación
}
```

### Condiciones de Entrada SHORT (RSI Overbought)
```python
conditions_short = {
    'rsi_extremo': rsi > 75,
    'rsi_girando': rsi < prev_rsi,
    'vwap_cerca': abs(price - vwap) / vwap < 0.02,
    'stoch_overbought': stoch_k > 80 and stoch_d > 80,
    'stoch_cruzando': stoch_k < stoch_d,
    'no_breakout': price < bb_upper,
    'volumen_climax': rel_volume > 2.0,
}
```

### Gestión de Posición
- **Stop Loss**: Low de la vela de entrada (LONG) o High (SHORT)
- **TP1**: VWAP
- **TP2**: EMA20
- **TP3**: Lado opuesto de Bollinger

### Configuración Recomendada
```python
@dataclass
class RSIExtremesConfig:
    rsi_extreme_oversold: int = 25
    rsi_extreme_overbought: int = 75
    stoch_oversold: int = 20
    stoch_overbought: int = 80
    max_vwap_distance: float = 0.02
    min_climax_volume: float = 2.0
    require_stoch_cross: bool = True
```

---

## Implementación Sugerida

### Estructura de Archivos
```
strategy/
├── orb.py              # Existente
├── vwap_reversion.py   # Nueva
├── pullback.py         # Nueva
├── gap_fill.py         # Nueva
├── bollinger_squeeze.py # Nueva
├── rsi_extremes.py     # Nueva
└── strategy_selector.py # Selector inteligente
```

### Selector de Estrategia Inteligente
```python
class StrategySelector:
    """
    Selecciona la mejor estrategia basada en condiciones de mercado
    """

    def select_strategy(self, market_data: dict) -> str:
        """
        Retorna el nombre de la estrategia más apropiada
        """
        vix = market_data.get('vix', 20)
        market_trend = market_data.get('spy_trend')  # 'up', 'down', 'range'
        time_of_day = market_data.get('minutes_since_open', 0)

        # Primera hora: ORB y Gap Fill
        if time_of_day < 60:
            if abs(market_data.get('gap_pct', 0)) >= 2.0:
                if market_data.get('has_catalyst'):
                    return 'ORB'  # Breakout con catalizador
                else:
                    return 'GAP_FILL'  # Fade sin catalizador
            return 'ORB'

        # Media mañana: Tendencia o Reversión
        if 60 <= time_of_day < 180:
            if market_trend in ['up', 'down']:
                return 'PULLBACK'  # Seguir tendencia
            else:
                return 'VWAP_REVERSION'  # Mercado en rango

        # Tarde: Más conservador
        if market_data.get('bollinger_squeeze'):
            return 'BOLLINGER_SQUEEZE'

        if market_data.get('rsi_extreme'):
            return 'RSI_EXTREMES'

        return 'VWAP_REVERSION'  # Default conservador
```

---

## Backtesting y Validación

### Métricas Clave a Monitorear
```python
@dataclass
class StrategyMetrics:
    win_rate: float          # Objetivo: > 50%
    profit_factor: float     # Objetivo: > 1.5
    avg_win_loss_ratio: float # Objetivo: > 1.2
    max_drawdown: float      # Límite: < 10%
    sharpe_ratio: float      # Objetivo: > 1.0
    trades_per_day: float    # Límite: 3-5
    avg_hold_time: float     # Objetivo: < 60 min
```

### Reglas de Rotación de Estrategias
1. Si win_rate < 40% en últimos 10 trades: pausar estrategia 1 día
2. Si profit_factor < 1.0 en última semana: revisar parámetros
3. Si max_drawdown > 5% en un día: circuit breaker

---

## Configuración Global Recomendada

```python
# En config/settings.py, añadir:

@dataclass
class MultiStrategyConfig:
    # Capital allocation por estrategia
    max_capital_per_strategy: float = 0.30   # 30% máx por estrategia

    # Correlación
    max_correlated_positions: int = 2        # No más de 2 del mismo sector

    # Tiempo
    min_time_between_signals: int = 300      # 5 min entre señales

    # Estrategias activas
    enabled_strategies: list = field(default_factory=lambda: [
        'ORB',
        'VWAP_REVERSION',
        'PULLBACK',
        'GAP_FILL'
    ])
```

---

## Resumen de Prioridades

### Para Empezar (Bajo Riesgo)
1. **VWAP Mean Reversion** - Alta probabilidad, bajo riesgo
2. **RSI Extremos** - Setups claros, fácil de identificar

### Para Avanzados (Mayor Potencial)
3. **Momentum Pullback** - Requiere identificar tendencia
4. **Gap Fill** - Necesita filtrar catalizadores

### Para Expertos (Más Complejo)
5. **Bollinger Squeeze** - Timing es crítico

---

## Próximos Pasos

1. Implementar `VWAPReversionStrategy` como segunda estrategia
2. Crear `StrategySelector` para rotación automática
3. Añadir backtesting para cada estrategia
4. Integrar con sistema de alertas Telegram existente

Todas estas estrategias usan los indicadores ya implementados en `data/indicators.py` y los datos de Alpaca en `data/market_data.py`.
