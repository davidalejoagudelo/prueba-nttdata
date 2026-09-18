"""
High Garden Coffee: tablero de avance (CRISP-DM, fases 1 a 3).

Ejecutar:
    pip install "streamlit>=1.50" pandas pyarrow altair google-genai duckdb
    streamlit run app.py

Asistente IA (opcional): pega tu API key de Google AI Studio en la barra lateral.
También puedes definirla fuera del código, en .streamlit/secrets.toml (GEMINI_API_KEY = "...")
o en la variable de entorno GEMINI_API_KEY. Nunca la escribas en este archivo ni la subas a GitHub.

Requiere en la misma carpeta (o en DATA_DIR) el archivo generado por el notebook:
    coffee_balance_long.parquet
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import config

DATA_DIR = config.DATA_DIR
RUTA = config.RUTA_BALANCE

VERDE, ROJO, AZUL, OCRE, GRIS = "#3F7A4A", "#A3243B", "#2D4F7C", "#9A6A12", "#8A958F"
SEG_LABEL = {
    "modelable": "Modelable",
    "consumo plano: proyección constante": "Consumo plano",
    "excluido: datos insuficientes": "Excluido: datos insuficientes",
    "excluido: baja materialidad": "Excluido: volumen marginal",
}
DEFICIT4 = ["Philippines", "Thailand", "Venezuela", "Cuba"]

st.set_page_config(page_title="High Garden Coffee", page_icon="☕", layout="wide")

try:
    import asistente_ia as ia
    IA_DISPONIBLE = True
except ImportError:
    IA_DISPONIBLE = False


# ---------------------------------------------------------------- datos
@st.cache_data
def cargar(ruta: Path) -> pd.DataFrame:
    df = pd.read_parquet(ruta).sort_values(["country", "anio_inicio"])
    df["segmento_label"] = df["segmento"].map(SEG_LABEL)
    return df


@st.cache_data
def perfil(df: pd.DataFrame) -> pd.DataFrame:
    def f(g):
        c = g["consumo_sacos"]
        return pd.Series({
            "coffee_type": g["coffee_type"].iloc[0],
            "segmento": g["segmento"].iloc[0],
            "anios_deficit": int(g["flag_deficit"].sum()),
            "prop_sin_cambio": (c.diff().dropna() == 0).mean() if c.notna().sum() > 1 else np.nan,
            "ceros_consumo": int(g["flag_consumo_cero"].sum()),
            "ceros_produccion": int(g["flag_produccion_cero"].sum()),
        })
    return df.groupby("country").apply(f)


def cagr(a, b, n):
    return (b / a) ** (1 / n) - 1


if not RUTA.exists():
    st.error(f"No se encontró {RUTA.name}. Ejecuta el notebook CRISP-DM (fase 3) para generarlo "
             "y ponlo en la misma carpeta que app.py.")
    st.stop()

df = cargar(RUTA)
paises = perfil(df)
glob = df.groupby("crop_year")[["consumo_sacos", "produccion_sacos"]].sum() / 1e6
glob["anio"] = glob.index.str[:4].astype(int)


def linea_ratio(data, titulo="", log=False, alto=180):
    escala = alt.Scale(type="log") if log else alt.Scale(zero=True)
    base = alt.Chart(data).encode(x=alt.X("anio_inicio:Q", title=None, axis=alt.Axis(format="d")))
    linea = base.mark_line(color=ROJO, strokeWidth=2).encode(
        y=alt.Y("ratio_consumo_produccion:Q", title="Consumo / producción", scale=escala),
        tooltip=["crop_year", alt.Tooltip("ratio_consumo_produccion:Q", format=".2f")])
    regla = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(color=GRIS, strokeDash=[5, 4]).encode(y="y:Q")
    return (linea + regla).properties(height=alto, title=titulo)


# ---------------------------------------------------------------- navegación
st.sidebar.title("High Garden Coffee")
st.sidebar.caption("Productores de café que se están volviendo mercados de destino")
vista = st.sidebar.radio("Sección", [
    "Resumen", "Explorar un país", "Consultar con IA 🤖",
    "1 · Negocio", "2 · Datos", "3 · Preparación",
    "4 · Modelado", "5 · Evaluación", "6 · Despliegue",
])


@st.cache_data
def cargar_modelado() -> dict | None:
    """Artefactos que exporta el notebook 4. Si faltan, las fases 4 y 5 muestran solo el diseño."""
    if not all(p.exists() for p in config.ARTEFACTOS_MODELADO.values()):
        return None
    return {n: pd.read_parquet(p) for n, p in config.ARTEFACTOS_MODELADO.items()}


modelado = cargar_modelado()



# ---------------------------------------------------------------- asistente IA: clave y cliente
if IA_DISPONIBLE:
    MODELOS_POR_DEFECTO = ia.MODELOS_PREFERIDOS
    st.session_state.setdefault("api_key", config.obtener_api_key())
    with st.sidebar.expander("🔑 Asistente IA (Gemini)"):
        st.caption(f"Clave detectada en: **{config.origen_api_key()}**")
        st.text_input("API key de Google AI Studio", type="password", key="api_key",
                      help="Se guarda solo en la memoria de esta sesión. Para no pegarla cada vez, "
                           "ponla en el archivo .env (ignorado por git) o en los secrets de Streamlit.")
        opciones = st.session_state.get("modelos") or MODELOS_POR_DEFECTO
        st.selectbox("Modelo", opciones, key="modelo")
        st.caption("Plan gratuito: mientras no actives la facturación en AI Studio, no hay cobros. "
                   "Si se agota el cupo, la API solo responde con error 429.")


def obtener_cliente():
    """Un cliente por sesión. Conserva el contador de llamadas aunque cambies de modelo."""
    clave = st.session_state.get("api_key", "").strip()
    if not IA_DISPONIBLE or not clave:
        return None
    cliente = st.session_state.get("cliente")
    if cliente is None or cliente.api_key != clave:
        previas = cliente.llamadas if cliente else 0
        cliente = ia.ClienteGemini(api_key=clave, llamadas=previas)
        st.session_state["cliente"] = cliente
    cliente.modelo = st.session_state.get("modelo", MODELOS_POR_DEFECTO[0])
    return cliente


@st.cache_resource
def conexion_sql(ruta: Path):
    return ia.conexion_duckdb(ruta)


def dibujar(resultado: pd.DataFrame, g: dict):
    """Dibuja solo si la especificación del modelo usa columnas que existen en el resultado."""
    tipo, x, y, color = g.get("tipo"), g.get("x"), g.get("y"), g.get("color")
    columnas = set(resultado.columns)
    if tipo not in ("linea", "barra") or x not in columnas or y not in columnas:
        return
    base = alt.Chart(resultado)
    marca = base.mark_line(strokeWidth=2) if tipo == "linea" else base.mark_bar(color=VERDE)
    enc = {"x": alt.X(x, sort="-y") if tipo == "barra" else alt.X(x),
           "y": alt.Y(y),
           "tooltip": list(resultado.columns)[:6]}
    if color in columnas:
        enc["color"] = alt.Color(color, legend=alt.Legend(orient="top", title=None))
    st.altair_chart(marca.encode(**enc).properties(height=360, title=g.get("titulo", "")), width="stretch")


def responder(cliente, pregunta: str):
    """Flujo completo. Las piezas marcadas como TODO viven en asistente_ia.py."""
    system, prompt = ia.construir_prompt(pregunta)
    plan = ia.interpretar_respuesta(cliente.generar(prompt, system=system, json_mode=True))
    if plan.get("fuera_de_alcance"):
        st.warning(plan.get("motivo") or "La pregunta está fuera del alcance de los datos.")
        return
    sql = plan.get("sql", "")
    with st.expander("SQL generado"):
        st.code(sql, language="sql")
    ok, motivo = ia.validar_sql(sql)
    if not ok:
        st.error(f"Consulta bloqueada por la validación: {motivo}")
        return
    sql = ia.agregar_limit_si_falta(sql)
    resultado = ia.ejecutar_sql(conexion_sql(RUTA).cursor(), sql)
    dibujar(resultado, plan.get("grafica") or {})
    st.dataframe(resultado, hide_index=True, width="stretch")
    try:
        st.markdown(ia.resumir_resultado(cliente, pregunta, resultado))
    except NotImplementedError:
        st.caption("Resumen pendiente: completa resumir_resultado() en asistente_ia.py.")


# ---------------------------------------------------------------- resumen
if vista == "Resumen":
    st.title("Cuatro países productores ya consumen más café del que cosechan")
    st.write("Filipinas, Tailandia, Venezuela y Cuba importan para cubrir su propio consumo. "
             "La línea punteada marca el punto en que el consumo iguala a la producción.")
    cols = st.columns(4)
    for col, k in zip(cols, DEFICIT4):
        s = df[df["country"] == k]
        ult = s[s["crop_year"] == "2019/20"].iloc[0]
        with col:
            st.subheader(k)
            st.metric("Ratio 2019/20", f"{ult['ratio_consumo_produccion']:.1f}×",
                      f"{int(paises.loc[k, 'anios_deficit'])} años en déficit", delta_color="off")
            st.altair_chart(linea_ratio(s, log=True), width="stretch")
            st.caption(f"Déficit 2019/20: {-ult['excedente_sacos']:,.0f} sacos. {ult['coffee_type']}.".replace(",", "."))

    st.header("Lo que sabemos hasta ahora")
    st.markdown(f"""
- **El consumo interno de los productores crece más rápido que su producción:** {cagr(glob.consumo_sacos.iloc[0], glob.consumo_sacos.iloc[-1], 29):.1%} anual frente a {cagr(glob.produccion_sacos.iloc[0], glob.produccion_sacos.iloc[-1], 29):.1%}.
- **El dataset no está en tazas, está en kilogramos.** El 99,9% de los valores son múltiplos de 60.
- **En 32 de 53 países el consumo casi no cambia de un año a otro.** Solo 18 países tienen series útiles para modelar.
- **La producción es unas 3 veces más volátil que el consumo.** Conviene proyectar cada serie por separado.
""")

# ---------------------------------------------------------------- explorar
elif vista == "Explorar un país":
    st.title("Explorar un país")
    c1, c2 = st.columns(2)
    seg = c1.selectbox("Segmento", ["Todos"] + list(SEG_LABEL.values()))
    lista = sorted(paises.index if seg == "Todos" else paises.index[paises["segmento"].map(SEG_LABEL) == seg])
    k = c2.selectbox("País", lista, index=lista.index("Philippines") if "Philippines" in lista else 0)
    s = df[df["country"] == k]
    p = paises.loc[k]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tipo de café", p["coffee_type"])
    m2.metric("Segmento", SEG_LABEL[p["segmento"]])
    m3.metric("Años en déficit", int(p["anios_deficit"]))
    exc = s["excedente_sacos"].dropna()
    m4.metric("Excedente último año", f"{exc.iloc[-1]:,.0f}".replace(",", ".") if len(exc) else "sin dato")

    g1, g2 = st.columns(2)
    largo = s.melt(id_vars=["anio_inicio", "crop_year"], value_vars=["produccion_sacos", "consumo_sacos"],
                   var_name="serie", value_name="sacos")
    largo["serie"] = largo["serie"].map({"produccion_sacos": "Producción", "consumo_sacos": "Consumo doméstico"})
    g1.altair_chart(alt.Chart(largo).mark_line(strokeWidth=2).encode(
        x=alt.X("anio_inicio:Q", title=None, axis=alt.Axis(format="d")),
        y=alt.Y("sacos:Q", title="Sacos de 60 kg"),
        color=alt.Color("serie:N", scale=alt.Scale(domain=["Producción", "Consumo doméstico"], range=[VERDE, ROJO]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["crop_year", "serie", alt.Tooltip("sacos:Q", format=",.0f")],
    ).properties(height=340, title="Producción y consumo"), width="stretch")
    usar_log = s["ratio_consumo_produccion"].max() > 3
    g2.altair_chart(linea_ratio(s, "Ratio consumo / producción", log=usar_log, alto=340), width="stretch")

    if k == "Philippines":
        st.warning("Posible quiebre de serie: la producción cae de ~730 mil a ~189 mil sacos entre 2009/10 y 2010/11 "
                   "y se queda en ese nivel. El modelado usará datos desde 2010/11.")
    if p["prop_sin_cambio"] >= 0.5:
        st.warning(f"Consumo con escalones: no cambia en {p['prop_sin_cambio']:.0%} de los años. "
                   "Su proyección será constante y de menor certeza.")
    if p["ceros_consumo"] or p["ceros_produccion"]:
        st.info(f"Años sin dato reportado: {p['ceros_consumo']} en consumo y {p['ceros_produccion']} en producción.")

# ---------------------------------------------------------------- asistente IA
elif vista.startswith("Consultar"):
    st.title("Consultar con IA (beta)")
    st.caption("Preguntas en lenguaje natural sobre la tabla analítica. El modelo propone SQL, "
               "la app lo valida y lo ejecuta en DuckDB, y luego dibuja el resultado.")
    cliente = obtener_cliente()
    if not IA_DISPONIBLE:
        st.info("Instala las dependencias: pip install google-genai duckdb")
    elif cliente is None:
        st.info("Ingresa tu API key en la barra lateral, en «🔑 Asistente IA (Gemini)».")
    else:
        c1, c2 = st.columns([1, 3])
        if c1.button("Probar conexión"):
            try:
                eco = cliente.probar_conexion()
                modelos = cliente.listar_modelos_flash() or MODELOS_POR_DEFECTO
                if cliente.modelo in modelos:
                    modelos = [cliente.modelo] + [m for m in modelos if m != cliente.modelo]
                st.session_state["modelos"] = modelos
                if st.session_state.get("modelo") != cliente.modelo:
                    st.session_state["modelo"] = cliente.modelo
                    st.rerun()
                st.success(f"Conexión correcta (respuesta: «{eco}»). Modelo confirmado: {cliente.modelo}. "
                           "Otros modelos Flash disponibles: " + ", ".join(modelos))
            except Exception as e:
                st.error(ia.explicar_error(e))
        c2.caption(f"Llamadas en esta sesión: {cliente.llamadas} de {cliente.limite} · modelo: {cliente.modelo}")

        pregunta = st.text_input("Tu pregunta", placeholder="¿Qué países tuvieron déficit en 2019/20?")
        if st.button("Consultar", type="primary", disabled=not pregunta.strip()):
            try:
                responder(cliente, pregunta.strip())
            except NotImplementedError as e:
                st.info(f"Pieza pendiente: {e}.")
            except Exception as e:
                st.error(ia.explicar_error(e))

# ---------------------------------------------------------------- fase 1
elif vista.startswith("1"):
    st.title("Fase 1 · Entendimiento del negocio")
    st.info("**Objetivo:** identificar y priorizar, a 3 años cafeteros, los países productores cuyo balance "
            "producción − consumo se está cerrando o ya es negativo.")
    a, b = st.columns(2)
    a.subheader("Preguntas de negocio")
    a.markdown("""
1. ¿Qué productores ya consumen más de lo que producen?
2. ¿En cuáles el consumo crece más rápido que la producción?
3. ¿Qué tamaño tendrá el déficit en 3 años?
4. ¿Qué tipo de café predomina en esos mercados?
""")
    b.subheader("Restricciones")
    b.markdown("""
- Sin precios: se proyectan volúmenes.
- Datos hasta 2019/20: validación con backtesting.
- Series cortas (30 puntos anuales).
- Plazo de 2 días.
""")
    st.subheader("Hipótesis")
    st.dataframe(pd.DataFrame({
        "Hipótesis": ["H1: el consumo crece más rápido que la producción",
                      "H2: algunos productores ya son importadores netos",
                      "H3: la producción es más volátil que el consumo",
                      "H4: un modelo simple supera al ingenuo a 3 años"],
        "Estado": ["✅ Confirmada", "✅ Confirmada", "✅ Confirmada", "⏳ Pendiente"],
    }), hide_index=True, width="stretch")

# ---------------------------------------------------------------- fase 2
elif vista.startswith("2"):
    st.title("Fase 2 · Entendimiento de los datos")
    st.error("**El enunciado dice \"tazas\", pero los datos están en kilogramos.** El 99,9% de los valores son "
             "múltiplos exactos de 60. La producción viene en toneladas métricas.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Crecimiento anual del consumo", f"{cagr(glob.consumo_sacos.iloc[0], glob.consumo_sacos.iloc[-1], 29):.1%}")
    k2.metric("Crecimiento anual de la producción", f"{cagr(glob.produccion_sacos.iloc[0], glob.produccion_sacos.iloc[-1], 29):.1%}")
    k3.metric("Cosecha consumida en casa",
              f"{glob.consumo_sacos.iloc[-1] / glob.produccion_sacos.iloc[-1]:.0%}",
              f"desde {glob.consumo_sacos.iloc[0] / glob.produccion_sacos.iloc[0]:.0%}")
    gl = glob.reset_index().melt(id_vars=["crop_year", "anio"], var_name="serie", value_name="millones")
    gl["serie"] = gl["serie"].map({"produccion_sacos": "Producción", "consumo_sacos": "Consumo doméstico"})
    st.altair_chart(alt.Chart(gl).mark_line(strokeWidth=2).encode(
        x=alt.X("anio:Q", title=None, axis=alt.Axis(format="d")),
        y=alt.Y("millones:Q", title="Millones de sacos"),
        color=alt.Color("serie:N", scale=alt.Scale(domain=["Producción", "Consumo doméstico"], range=[VERDE, ROJO]),
                        legend=alt.Legend(orient="top", title=None)),
    ).properties(height=320), width="stretch")

    a, b = st.columns(2)
    ult = df[df["crop_year"] == "2019/20"].set_index("country")["consumo_sacos"].dropna()
    top = (ult / ult.sum()).nlargest(10).rename("participacion").reset_index()
    a.altair_chart(alt.Chart(top).mark_bar(color=ROJO).encode(
        x=alt.X("participacion:Q", axis=alt.Axis(format="%"), title=None),
        y=alt.Y("country:N", sort="-x", title=None)).properties(title="Top 10 consumidores 2019/20", height=300),
        width="stretch")
    piv = df.pivot(index="country", columns="crop_year", values="consumo_sacos")
    base = piv["2009/10"]
    crec = cagr(base, piv["2019/20"], 10)[base >= 10_000].dropna().nlargest(10).rename("cagr").reset_index()
    b.altair_chart(alt.Chart(crec).mark_bar(color=VERDE).encode(
        x=alt.X("cagr:Q", axis=alt.Axis(format="%"), title=None),
        y=alt.Y("country:N", sort="-x", title=None)).properties(title="Mayor crecimiento del consumo, 2009/10–2019/20", height=300),
        width="stretch")

    st.subheader("Calidad de los datos")
    st.dataframe(pd.DataFrame([
        ["Consumo en kg, no en tazas", "kg ÷ 60 = sacos"],
        ["Producción en toneladas", "t ÷ 0,06 = sacos"],
        ["Etiquetas de año distintas", "Mapeo 2020 → 2019/20 validado"],
        ["Paraguay con tipo 0", "Usar el tipo del archivo de consumo"],
        ["Ceros en ambas fuentes", "Tratados como dato no reportado"],
        ["32 de 53 series de consumo planas", "Proyección constante declarada"],
        ["Quiebre en Filipinas (2010/11)", "Modelar desde 2010/11"],
        ["CSV en latin-1", "encoding='latin-1'"],
    ], columns=["Hallazgo", "Decisión"]), hide_index=True, width="stretch")

# ---------------------------------------------------------------- fase 3
elif vista.startswith("3"):
    st.title("Fase 3 · Preparación de los datos")
    st.write("`coffee_balance_long.parquet`: 55 países × 30 años con consumo, producción, excedente, ratio y marcas de calidad.")
    conteo = paises["segmento"].map(SEG_LABEL).value_counts().rename("países").reset_index()
    st.altair_chart(alt.Chart(conteo).mark_bar().encode(
        x=alt.X("países:Q"), y=alt.Y("segmento:N", sort="-x", title=None),
        color=alt.Color("segmento:N", legend=None,
                        scale=alt.Scale(domain=list(SEG_LABEL.values()), range=[VERDE, OCRE, GRIS, "#5E6D65"]))),
        width="stretch")
    for s_key, label in SEG_LABEL.items():
        with st.expander(f"{label} ({(paises['segmento'] == s_key).sum()})"):
            st.write(", ".join(sorted(paises.index[paises["segmento"] == s_key])))

    st.subheader("Productores con al menos un año en déficit")
    ult = df[df["crop_year"] == "2019/20"].set_index("country")
    tabla = paises[(paises["anios_deficit"] > 0) & ~paises["segmento"].str.startswith("excluido")].copy()
    tabla["primer_deficit"] = [df[(df.country == k) & df.flag_deficit]["crop_year"].min() for k in tabla.index]
    tabla["ratio_2019_20"] = ult.loc[tabla.index, "ratio_consumo_produccion"]
    tabla["excedente_2019_20"] = ult.loc[tabla.index, "excedente_sacos"]
    tabla["segmento"] = tabla["segmento"].map(SEG_LABEL)
    st.dataframe(tabla.sort_values("anios_deficit", ascending=False)[
        ["anios_deficit", "primer_deficit", "coffee_type", "segmento", "ratio_2019_20", "excedente_2019_20"]
    ].style.format({"ratio_2019_20": "{:.2f}", "excedente_2019_20": "{:,.0f}"}), width="stretch")

    exp = df[df["country"].isin(["Brazil", "Ethiopia", "Indonesia", "Mexico", "Colombia"])]
    st.altair_chart(alt.Chart(exp).mark_line(strokeWidth=2).encode(
        x=alt.X("anio_inicio:Q", title=None, axis=alt.Axis(format="d")),
        y=alt.Y("ratio_consumo_produccion:Q", title="Consumo / producción", axis=alt.Axis(format="%")),
        color=alt.Color("country:N", legend=alt.Legend(orient="top", title=None)),
    ).properties(height=320, title="El mercado interno gana peso en los grandes productores"), width="stretch")

# ---------------------------------------------------------------- fase 4
elif vista.startswith("4"):
    st.title("Fase 4 · Modelado")
    if modelado is None:
        st.info("Ejecuta `notebooks/4_modelado.ipynb` para generar los artefactos de esta fase.")
        st.stop()

    st.write("Consumo y producción se proyectan por separado a 3 años cafeteros. Cada serie se queda con el modelo "
             "que mejor pronostica en backtesting de origen móvil, y la incertidumbre sale de los errores reales "
             "de ese backtesting.")

    bt, elegidos = modelado["backtesting"], modelado["modelos"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Pronósticos evaluados", f"{len(bt):,}".replace(",", "."))
    k2.metric("Series modeladas", f"{len(elegidos)}")
    k3.metric("Escenarios por país y año", "1.000")

    a, b = st.columns(2)
    a.subheader("MASE promedio por modelo")
    a.dataframe(bt.pivot_table(index="modelo", columns="variable", values="mase", aggfunc="mean").round(3),
                width="stretch")
    b.subheader("Modelo elegido por serie")
    b.dataframe(elegidos.pivot_table(index="variable", columns="modelo", aggfunc="size", fill_value=0),
                width="stretch")

    st.subheader("MASE por horizonte")
    por_h = bt.pivot_table(index="h", columns="modelo", values="mase", aggfunc="mean").reset_index()
    st.altair_chart(alt.Chart(por_h.melt(id_vars="h", var_name="modelo", value_name="mase")).mark_line(
        strokeWidth=2, point=True).encode(
        x=alt.X("h:O", title="Años hacia adelante"),
        y=alt.Y("mase:Q", title="MASE promedio"),
        color=alt.Color("modelo:N", legend=alt.Legend(orient="top", title=None)),
    ).properties(height=300), width="stretch")
    st.caption("El error crece con el horizonte: a tres años ningún modelo es preciso, por eso el resultado "
               "se entrega como probabilidad y no como número único.")

# ---------------------------------------------------------------- fase 5
elif vista.startswith("5"):
    st.title("Fase 5 · Evaluación")
    if modelado is None:
        st.info("Ejecuta `notebooks/4_modelado.ipynb` y `notebooks/5_evaluacion.ipynb` para ver esta fase.")
        st.stop()

    ranking = modelado["ranking"].set_index("country")
    calibracion = modelado["calibracion"]
    anio = ranking["crop_year"].iloc[0]

    st.subheader(f"Mercados priorizados para {anio}")
    top = ranking.sort_values("necesidad_esperada_sacos", ascending=False).head(8).reset_index()
    st.altair_chart(alt.Chart(top).mark_bar().encode(
        x=alt.X("necesidad_esperada_sacos:Q", title="Necesidad esperada de importación (sacos)"),
        y=alt.Y("country:N", sort="-x", title=None),
        color=alt.Color("confiabilidad:N", legend=alt.Legend(orient="top", title=None),
                        scale=alt.Scale(range=[OCRE, ROJO])),
        tooltip=["country", "prob_deficit", "necesidad_esperada_sacos", "coffee_type"],
    ).properties(height=320), width="stretch")

    st.dataframe(top[["country", "coffee_type", "prob_deficit", "necesidad_esperada_sacos", "confiabilidad"]]
                 .style.format({"prob_deficit": "{:.0%}", "necesidad_esperada_sacos": "{:,.0f}"}),
                 hide_index=True, width="stretch")

    st.subheader("¿Se puede confiar en esas probabilidades?")
    brier = ((calibracion["prob_pronosticada"] - calibracion["deficit_real"]) ** 2).mean()
    base = ((calibracion["prob_climatologica"] - calibracion["deficit_real"]) ** 2).mean()
    relevantes = calibracion[calibracion["prob_climatologica"] > 0]
    brier_rel = ((relevantes["prob_pronosticada"] - relevantes["deficit_real"]) ** 2).mean()
    base_rel = ((relevantes["prob_climatologica"] - relevantes["deficit_real"]) ** 2).mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Brier (todos los casos)", f"{brier:.3f}", f"línea base {base:.3f}", delta_color="off")
    c2.metric("Brier (países con historia de déficit)", f"{brier_rel:.3f}", f"línea base {base_rel:.3f}",
              delta_color="off")
    c3.metric("Casos evaluados", f"{len(calibracion)}")

    cortes = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    tabla = (calibracion.assign(rango=pd.cut(calibracion["prob_pronosticada"], cortes, include_lowest=True))
             .groupby("rango", observed=True)
             .agg(casos=("deficit_real", "size"), probabilidad_media=("prob_pronosticada", "mean"),
                  frecuencia_real=("deficit_real", "mean")).reset_index())
    tabla["rango"] = tabla["rango"].astype(str)
    st.dataframe(tabla.style.format({"probabilidad_media": "{:.0%}", "frecuencia_real": "{:.0%}"}),
                 hide_index=True, width="stretch")
    st.warning("El método acierta en los extremos, pero sobreestima en la zona intermedia. "
               "Sirve para **ordenar** mercados e identificar casos claros, no como probabilidad literal.")

# ---------------------------------------------------------------- fase 6
else:
    st.title("Fase 6 · Despliegue")
    st.write("Esta aplicación es el despliegue: lee los artefactos que generan los notebooks y los deja "
             "consultables, incluida la pregunta en lenguaje natural.")
    a, b = st.columns(2)
    a.subheader("Cómo se ejecuta")
    a.markdown("""
1. `pip install -r requirements.txt`
2. Copia `.env.example` a `.env` y pon tu `GEMINI_API_KEY` (opcional, solo para el asistente).
3. `streamlit run app.py`

Los notebooks se corren en orden 1 → 5 y regeneran todo lo que hay en `data/`.
""")
    b.subheader("Próxima iteración")
    b.markdown("""
- Modelar consumo y producción de forma conjunta para corregir la sobreestimación.
- Recalibrar las probabilidades antes de publicarlas.
- Incorporar exportaciones de la ICO y variables de población y PIB.
- Reentrenamiento anual cuando la ICO publique datos nuevos.
""")
    st.subheader("Limitaciones que deben acompañar cualquier decisión")
    st.markdown("""
- **Sin precios:** el resultado dice cuánto café necesitará importar un mercado, no si venderle será rentable.
- **Balance aparente:** no incluye existencias ni re-exportaciones.
- **Datos hasta 2019/20:** las proyecciones no se han podido contrastar con los años posteriores.
""")
