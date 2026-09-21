# High Garden Coffee — mercados de exportación potenciales

Reto técnico de ML Engineering resuelto con metodología **CRISP-DM**: identifica países productores de café
cuyo consumo interno ya supera (o está por superar) su propia producción, es decir, futuros mercados de destino.

🔗 **App en vivo:** https://prueba-nttdata-nd5h4hecz7yc9mlqqzc4mw.streamlit.app/

## Resultado principal

Para el año cafetero 2022/23, cuatro países ya consumen más café del que producen:

| Mercado | Necesidad esperada (sacos de 60 kg) | Probabilidad de déficit |
|---|---:|---:|
| Filipinas | 4.088.042 | 100 % |
| Tailandia | 1.270.967 | 100 % |
| Venezuela | 173.014 | 79 % |
| Cuba | 82.582 | 100 % |

El ranking es estable (Spearman 0,97 en backtesting). La app incluye un asistente de IA (text-to-SQL) para
consultar los resultados en lenguaje natural.

## Estructura

```
.
├── app.py                # aplicación de Streamlit
├── asistente_ia.py       # asistente text-to-SQL
├── notebooks/            # 1_negocio → 5_evaluación (CRISP-DM)
└── data/                 # datasets fuente y generados
```

## Ejecutar en local

```bash
git clone https://github.com/davidalejoagudelo/prueba-nttdata
cd <carpeta-del-repositorio>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Para el asistente de IA, copia `.env.example` a `.env` y agrega tu clave de Gemini (plan gratuito, sin
tarjeta). El resto de la app funciona sin clave.

## Limitaciones

- No considera precios ni rentabilidad, solo volumen de importación esperado.
- No descuenta inventarios ni re-exportaciones.
- Datos históricos hasta 2019/20 (ICO); no contrastados con años posteriores.

## Fuente

Organización Internacional del Café (ICO), series 1990/91–2019/20 de consumo doméstico y producción por país.
