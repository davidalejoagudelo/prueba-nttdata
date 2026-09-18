# High Garden Coffee — mercados de exportación potenciales

Reto técnico de ML Engineering resuelto con la metodología **CRISP-DM**. El proyecto identifica y prioriza
países **productores** de café que se están volviendo **mercados de destino**: aquellos cuyo consumo interno
ya supera, o está por superar, su propia producción.

La entrega son cinco notebooks encadenados y una aplicación de Streamlit que deja los resultados consultables,
incluso preguntando en lenguaje natural.

## Resultado principal

Cuatro países productores ya consumen más café del que cosechan. Para el año cafetero 2022/23, la proyección
estima la siguiente necesidad de importación:

| Mercado | Necesidad esperada (sacos de 60 kg) | Probabilidad de déficit | Tipo de café |
|---|---:|---:|---|
| Filipinas | 4.088.042 | 100 % | Robusta/Arábica |
| Tailandia | 1.270.967 | 100 % | Robusta/Arábica |
| Venezuela | 173.014 | 79 % | Arábica |
| Cuba | 82.582 | 100 % | Arábica |

Las cifras se acompañan de una marca de confiabilidad: Filipinas se modela con una serie corta posterior a un
quiebre, y el consumo de Cuba viene de una serie sin variación.

## Estructura

```
.
├── app.py                  # aplicación de Streamlit
├── asistente_ia.py         # asistente text-to-SQL sobre la tabla analítica
├── config.py               # rutas y carga de credenciales
├── requirements.txt
├── .env.example            # plantilla de credenciales (el .env real no se sube)
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── data/
│   ├── coffee_db.parquet                 # consumo doméstico (dataset del reto)
│   ├── coffee-production-...-ico.csv     # producción (fuente pública, datos ICO)
│   ├── coffee_balance_long.parquet       # tabla analítica (genera el notebook 3)
│   └── modelado_*.parquet                # resultados del modelado (genera el notebook 4)
└── notebooks/
    ├── 1_entendimiento_negocio.ipynb
    ├── 2_entendimiento_datos_eda.ipynb
    ├── 3_preparacion_datos.ipynb
    ├── 4_modelado.ipynb
    └── 5_evaluacion.ipynb
```

## Cómo ejecutarlo

```bash
git clone <URL-DEL-REPOSITORIO>
cd <carpeta-del-repositorio>

python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Los notebooks se ejecutan en orden 1 → 5 y regeneran todo el contenido de `data/`.

## Credenciales del asistente de IA

La sección *Consultar con IA* usa la API de Gemini. **La clave nunca va en el código.** Se busca en este orden:

1. `st.secrets["GEMINI_API_KEY"]` (Streamlit Community Cloud)
2. variable de entorno `GEMINI_API_KEY`
3. archivo `.env` en la raíz del repositorio
4. lo que se escriba en la barra lateral de la aplicación

Para trabajar en local:

```bash
cp .env.example .env
# edita .env y pega tu clave de Google AI Studio
```

`.env` y `.streamlit/secrets.toml` están en `.gitignore`. Si alguna vez subes una clave por error, bórrala en
AI Studio y genera otra: reescribir el commit no basta.

El plan gratuito de AI Studio no pide tarjeta, así que no genera cobros; si se agota el cupo, la API responde
con error 429 y la aplicación lo explica. El resto de la aplicación funciona sin clave.

## Despliegue en Streamlit Community Cloud

1. Sube el repositorio a GitHub.
2. En [share.streamlit.io](https://share.streamlit.io), crea una app apuntando a `app.py`.
3. En *Settings → Secrets*, pega el contenido de `.streamlit/secrets.toml.example` con tu clave real.

## Metodología

| Fase | Qué contiene | Hallazgo o entregable |
|---|---|---|
| 1. Negocio | Objetivo, preguntas, hipótesis y criterios de éxito | Priorizar mercados a 3 años cafeteros |
| 2. Datos | Procedencia, unidades, calidad | El dataset está en kg, no en tazas; 32 de 53 series de consumo son planas |
| 3. Preparación | Integración con producción, limpieza, indicadores | `coffee_balance_long.parquet` y segmentación de países |
| 4. Modelado | Torneo de modelos con backtesting de origen móvil y simulación | Probabilidad de déficit y necesidad esperada de importación |
| 5. Evaluación | Criterios de éxito, estabilidad y revisión del proceso | El ranking es estable (Spearman 0,97; top 5 idéntico) |

## Cómo funciona el asistente de IA

1. El modelo recibe la pregunta y una descripción semántica de la tabla, y devuelve un JSON con la consulta SQL
   y la especificación de la gráfica.
2. `validar_sql()` verifica que sea una sola sentencia de solo lectura sobre la tabla `balance`, sin funciones
   de acceso a archivos; si falta `LIMIT`, se agrega.
3. DuckDB ejecuta la consulta sobre una copia en memoria, con el acceso a archivos externos desactivado.
4. Un segundo llamado resume el resultado usando solo las filas devueltas.

## Limitaciones

- **Sin precios.** El resultado dice cuánto café necesitará importar un mercado, no si venderle será rentable.
- **Balance aparente.** No descuenta existencias ni re-exportaciones.
- **Datos hasta 2019/20.** Las proyecciones no se han contrastado con los años posteriores.
- **Calibración parcial.** El indicador acierta en los casos extremos, pero sobreestima la probabilidad en la
  zona intermedia: sirve para ordenar mercados, no como probabilidad literal.

## Fuentes

Organización Internacional del Café (ICO), series históricas 1990/91–2019/20 de consumo doméstico y producción
por país, obtenidas de datasets públicos derivados de los archivos de la ICO.
