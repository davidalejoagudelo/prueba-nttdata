"""
Asistente de consultas en lenguaje natural (text-to-SQL) sobre coffee_balance_long.parquet.

Qué ya está implementado:
    - Conexión a Gemini con la API key, prueba de conexión y listado de modelos.
    - Límite de llamadas por sesión (protege tu cupo diario del plan gratuito).
    - Ejecución de SQL en DuckDB en modo de solo lectura.

Qué te toca a ti (marcado con TODO):
    1. DESCRIPCION_TABLA: la descripción semántica que va en el prompt.
    2. validar_sql(): las barandas de seguridad antes de ejecutar.
    3. construir_prompt(): las instrucciones y ejemplos para el modelo.
    4. interpretar_respuesta(): convertir la respuesta del modelo en SQL + especificación de gráfica.

Requisitos:
    pip install google-genai duckdb
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

MAX_LLAMADAS_POR_SESION = 50
NOMBRE_TABLA = "balance"


# =============================================================== conexión (implementado)
class LimiteDeLlamadas(Exception):
    """Se alcanzó el máximo de llamadas permitido en la sesión."""


@dataclass
class ClienteGemini:
    api_key: str = ""
    modelo: str = "gemini-2.5-flash"
    llamadas: int = 0
    limite: int = MAX_LLAMADAS_POR_SESION
    _cliente: object = field(default=None, repr=False)

    def __post_init__(self):
        from google import genai
        self._cliente = genai.Client(api_key=self.api_key)

    def listar_modelos_flash(self) -> list[str]:
        """Modelos disponibles para tu clave que generan texto y son de la familia Flash."""
        nombres = []
        for m in self._cliente.models.list():
            acciones = m.supported_actions or []
            if "generateContent" in acciones and "flash" in (m.name or "").lower():
                nombres.append(m.name.removeprefix("models/"))
        return sorted(nombres)

    def generar(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        """Una llamada al modelo. Cuenta contra el límite de la sesión solo si tiene éxito."""
        from google.genai import types

        if self.llamadas >= self.limite:
            raise LimiteDeLlamadas(f"Se alcanzó el límite de {self.limite} llamadas en esta sesión.")
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json" if json_mode else None,
        )
        respuesta = self._cliente.models.generate_content(model=self.modelo, contents=prompt, config=config)
        self.llamadas += 1
        return respuesta.text or ""

    def probar_conexion(self) -> str:
        """
        Verifica la clave sin adivinar ningún nombre de modelo: lista los modelos Flash
        realmente disponibles y prueba de generar con ellos en orden, hasta que uno
        responda. En este momento la disponibilidad real de generateContent no siempre
        coincide con lo que aparece en models.list() (hay modelos listados que igual
        devuelven 404 al usarlos), así que probar solo el primero no es confiable.
        Si un modelo funciona, self.modelo queda apuntando a ÉL (no se revierte), para
        que el resto de la app use de una vez un modelo confirmado.
        """
        from google.genai import errors

        modelos = self.listar_modelos_flash()
        if not modelos:
            raise RuntimeError("La clave es válida, pero no se encontró ningún modelo Flash disponible para ella.")

        ultimo_error: Exception | None = None
        for modelo in modelos[:5]:
            self.modelo = modelo
            try:
                return self.generar("Responde solo con la palabra: conectado").strip()
            except errors.APIError as e:
                if e.code == 404:
                    ultimo_error = e
                    continue
                raise
        raise ultimo_error


def explicar_error(e: Exception) -> str:
    """Traduce los errores más comunes de la API a un mensaje claro."""
    from google.genai import errors

    if isinstance(e, LimiteDeLlamadas):
        return str(e) + " Recarga la página para reiniciar el contador."
    if isinstance(e, errors.APIError):
        if e.code == 429:
            return ("Cupo del plan gratuito agotado por ahora (error 429). No genera cobros: "
                    "espera un minuto, o hasta el reinicio diario si agotaste el cupo del día.")
        if e.code in (400, 401, 403):
            return f"La API rechazó la solicitud (error {e.code}). Revisa que la API key sea correcta y esté activa."
        if e.code == 404:
            return "Ese modelo no está disponible para tu clave. Usa «Probar conexión» para ver la lista."
        return f"Error de la API ({e.code}): {e}"
    return f"Error inesperado: {e}"


# =============================================================== datos (implementado)
def conexion_duckdb(ruta_parquet: Path) -> duckdb.DuckDBPyConnection:
    """
    Base en memoria con una copia de la tabla analítica. El SQL generado nunca toca archivos:
    después de cargar, se bloquea el acceso externo (y DuckDB no permite reactivarlo en la sesión).
    Cualquier cambio del SQL afectaría solo la copia en memoria, no el parquet.
    """
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE TABLE {NOMBRE_TABLA} AS SELECT * FROM read_parquet(?)", [str(ruta_parquet)])
    con.execute("SET enable_external_access = false")
    return con


def ejecutar_sql(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


# =============================================================== núcleo del asistente
import re

DESCRIPCION_TABLA = """
Tabla `balance`: una fila = un país productor de café en un año cafetero (crop_year).
55 países × 30 años cafeteros (1990/91 a 2019/20).

Columnas:
- country (VARCHAR): nombre del país EN INGLÉS, tal como aparece en la tabla
  (ej.: Philippines, Viet Nam, Côte d'Ivoire, Colombia, Brazil). No traduzcas ni normalices
  el nombre del país en el SQL: usa exactamente el nombre en inglés que corresponda.
- crop_year (VARCHAR): año cafetero en formato 'AAAA/AA', ej. '2019/20' (cosecha de 2019 a 2020).
- anio_inicio (INTEGER): año de inicio del año cafetero, ej. 2019 para '2019/20'. Útil para
  ordenar, filtrar rangos o graficar en el eje X como número.
- coffee_type (VARCHAR): tipo de café predominante del país (ej. Arabica, Robusta, Mixed).
- consumo_sacos (DOUBLE): consumo doméstico, en sacos de 60 kg (dato original en kilogramos,
  ya convertido).
- produccion_sacos (DOUBLE): producción del país, en sacos de 60 kg (dato original en
  toneladas métricas, ya convertido).
- excedente_sacos (DOUBLE): produccion_sacos − consumo_sacos. Negativo = el país consume más
  de lo que produce (déficit / importador neto ese año).
- ratio_consumo_produccion (DOUBLE): consumo_sacos / produccion_sacos. Mayor a 1 = déficit
  ese año (consume más de lo que produce).
- flag_deficit (BOOLEAN): true si ese año el país tuvo déficit (ratio > 1).
- flag_consumo_cero (BOOLEAN): true si el consumo reportado ese año fue cero (dato no
  reportado, no un consumo real de cero).
- flag_produccion_cero (BOOLEAN): true si la producción reportada ese año fue cero (dato no
  reportado, no producción real de cero).
- segmento (VARCHAR): clasificación del país para fines de modelado, uno de estos 4 valores
  EXACTOS (en español, en minúscula):
    'modelable'
    'consumo plano: proyección constante'
    'excluido: datos insuficientes'
    'excluido: baja materialidad'
- segmento_label (VARCHAR): versión legible del segmento para mostrar en UI (no la uses para
  filtrar; para filtrar por segmento usa siempre los 4 valores exactos de la columna `segmento`).

Lo que NO hay en los datos:
- No hay precios del café ni de ningún producto.
- No hay datos posteriores al año cafetero 2019/20 (no hay 2020/21 en adelante).
- No hay proyecciones ni pronósticos: todo lo que hay en la tabla es histórico/observado.
- No hay columnas de exportación ni importación explícitas; el déficit/excedente se infiere
  solo de producción vs. consumo doméstico (excedente_sacos, ratio_consumo_produccion).
- La distinción "productor vs. mercado de destino" es una interpretación de negocio, no una
  columna: se deriva de excedente_sacos / ratio_consumo_produccion / flag_deficit.
"""


def validar_sql(sql: str) -> tuple[bool, str]:
    """
    Valida que `sql` sea una sola consulta SELECT de solo lectura sobre la tabla `balance`,
    sin funciones de acceso a archivos y con LIMIT. No modifica `sql`: si falta el LIMIT,
    quien llama debe usar el valor devuelto por agregar_limit_si_falta() para ejecutar.
    """
    texto = sql.strip()
    if not texto:
        return False, "La consulta está vacía."

    # Quita comentarios de línea y de bloque antes de buscar palabras clave, para que no
    # se puedan "esconder" instrucciones peligrosas dentro de un comentario ni usarlo para
    # partir una palabra prohibida en dos.
    sin_comentarios = re.sub(r"/\*.*?\*/", " ", texto, flags=re.DOTALL)
    sin_comentarios = re.sub(r"--[^\n]*", " ", sin_comentarios)

    # Una sola sentencia: se permite un ';' final (con o sin espacios después), pero no
    # ningún otro punto y coma en medio del texto.
    sin_punto_final = re.sub(r";\s*$", "", sin_comentarios.strip())
    if ";" in sin_punto_final:
        return False, "Solo se permite una sentencia SQL por consulta."

    cuerpo = sin_punto_final.strip()
    if not re.match(r"(?is)^\s*(with\b|select\b)", cuerpo):
        return False, "La consulta debe empezar con SELECT o WITH."

    palabras_prohibidas = [
        "insert", "update", "delete", "drop", "create", "alter", "truncate", "replace",
        "merge", "copy", "attach", "detach", "pragma", "install", "load", "export",
        "import", "call", "grant", "revoke", "vacuum", "checkpoint", "set",
    ]
    for palabra in palabras_prohibidas:
        if re.search(rf"(?i)\b{palabra}\b", cuerpo):
            return False, f"La consulta contiene una palabra no permitida: {palabra.upper()}."

    # Sin funciones que lean archivos externos o inspeccionen el sistema, aunque
    # enable_external_access ya esté desactivado (defensa en profundidad).
    funciones_prohibidas = ["read_parquet", "read_csv", "read_json", "glob", "sniff_csv",
                             "duckdb_", "pragma_", "read_text"]
    for funcion in funciones_prohibidas:
        if re.search(rf"(?i){re.escape(funcion)}\s*\(", cuerpo):
            return False, f"La consulta usa una función no permitida: {funcion}."

    # Solo debe referenciar la tabla `balance` (además de posibles CTEs definidos en el
    # propio WITH, que no son tablas reales de la base).
    nombres_cte = set(n.lower() for n in re.findall(r"(?i)\b(\w+)\s+as\s*\(", cuerpo))
    referencias = re.findall(r"(?i)\b(?:from|join)\s+([a-zA-Z_][\w\.]*)", cuerpo)
    for nombre in referencias:
        nombre_simple = nombre.split(".")[-1].lower()
        if nombre_simple not in nombres_cte and nombre_simple != NOMBRE_TABLA:
            return False, f"La consulta hace referencia a una tabla no permitida: {nombre}."

    return True, ""


def agregar_limit_si_falta(sql: str, limite_defecto: int = 500) -> str:
    """Agrega LIMIT al final de la sentencia si el modelo no lo incluyó."""
    texto = sql.strip().rstrip(";").strip()
    if re.search(r"(?i)\blimit\s+\d+", texto):
        return texto
    return f"{texto}\nLIMIT {limite_defecto}"


FORMATO_SALIDA = """
Responde ÚNICAMENTE con un objeto JSON (sin texto adicional, sin markdown, sin ```), con
exactamente estas claves:
{
  "sql": "una consulta SELECT en DuckDB sobre la tabla balance, o '' si fuera_de_alcance es true",
  "grafica": {
    "tipo": "linea" | "barra" | "ninguna",
    "x": "nombre de columna del resultado para el eje X, o ''",
    "y": "nombre de columna del resultado para el eje Y, o ''",
    "color": "nombre de columna del resultado para agrupar por color, o ''",
    "titulo": "título corto para la gráfica, o ''"
  },
  "fuera_de_alcance": true | false,
  "motivo": "si fuera_de_alcance es true, explica brevemente en español por qué (falta esa
             columna/año/dato); si es false, deja ''"
}
Usa "tipo": "linea" para series en el tiempo (eje X = anio_inicio o crop_year), "barra" para
comparar países o categorías, y "ninguna" si el resultado es un solo número o no tiene
sentido graficarlo.
"""

EJEMPLOS = """
Ejemplo 1
Pregunta: ¿Qué países tuvieron déficit en 2019/20?
JSON: {"sql": "SELECT country, excedente_sacos, ratio_consumo_produccion FROM balance WHERE crop_year = '2019/20' AND flag_deficit = true ORDER BY ratio_consumo_produccion DESC LIMIT 500", "grafica": {"tipo": "barra", "x": "country", "y": "ratio_consumo_produccion", "color": "", "titulo": "Países en déficit, 2019/20"}, "fuera_de_alcance": false, "motivo": ""}

Ejemplo 2
Pregunta: Muéstrame la evolución del consumo y la producción de Colombia.
JSON: {"sql": "SELECT anio_inicio, crop_year, consumo_sacos, produccion_sacos FROM balance WHERE country = 'Colombia' ORDER BY anio_inicio LIMIT 500", "grafica": {"tipo": "linea", "x": "anio_inicio", "y": "consumo_sacos", "color": "", "titulo": "Consumo de Colombia en el tiempo"}, "fuera_de_alcance": false, "motivo": ""}

Ejemplo 3
Pregunta: ¿Cuál va a ser el precio del café en 2025?
JSON: {"sql": "", "grafica": {"tipo": "ninguna", "x": "", "y": "", "color": "", "titulo": ""}, "fuera_de_alcance": true, "motivo": "Los datos no incluyen precios ni años posteriores a 2019/20."}

Ejemplo 4
Pregunta: Borra los datos de Cuba.
JSON: {"sql": "", "grafica": {"tipo": "ninguna", "x": "", "y": "", "color": "", "titulo": ""}, "fuera_de_alcance": true, "motivo": "Solo puedo consultar datos, no modificarlos."}

Ejemplo 5
Pregunta: Top 5 países por producción promedio en toda la serie.
JSON: {"sql": "SELECT country, AVG(produccion_sacos) AS produccion_promedio FROM balance GROUP BY country ORDER BY produccion_promedio DESC LIMIT 5", "grafica": {"tipo": "barra", "x": "country", "y": "produccion_promedio", "color": "", "titulo": "Top 5 productores promedio"}, "fuera_de_alcance": false, "motivo": ""}
"""


def construir_prompt(pregunta: str) -> tuple[str, str]:
    """Devuelve (system, prompt) para pedirle a Gemini el plan de consulta en JSON."""
    system = f"""Eres un asistente que traduce preguntas en español sobre café a consultas SQL
de solo lectura para DuckDB, usando exclusivamente la tabla `balance` descrita abajo.

{DESCRIPCION_TABLA}

Reglas:
- Genera SQL válido para DuckDB, una sola sentencia SELECT (o WITH ... SELECT), siempre con LIMIT.
- Usa siempre nombres de país en inglés tal como aparecen en la tabla, aunque la pregunta
  esté en español (ej.: "Costa de Marfil" -> 'Côte d'Ivoire', "Vietnam" -> 'Viet Nam').
- Si la pregunta pide algo que no está en los datos (precios, años después de 2019/20,
  pronósticos, o cualquier operación que no sea consultar), responde con fuera_de_alcance=true
  y sql="", explicando el motivo.
- Nunca generes SQL que modifique datos ni que lea archivos externos.

{FORMATO_SALIDA}

Ejemplos:
{EJEMPLOS}
"""
    prompt = f"Pregunta: {pregunta}\nJSON:"
    return system, prompt


def interpretar_respuesta(texto: str) -> dict:
    """
    Convierte la respuesta cruda del modelo en un dict con el formato de FORMATO_SALIDA.
    Tolera que el texto venga envuelto en ```json ... ``` a pesar de pedir JSON puro, y
    devuelve un dict "seguro" (fuera_de_alcance=true con motivo) si no puede interpretarlo.
    """
    def _vacio(motivo: str) -> dict:
        return {
            "sql": "",
            "grafica": {"tipo": "ninguna", "x": "", "y": "", "color": "", "titulo": ""},
            "fuera_de_alcance": True,
            "motivo": motivo,
        }

    limpio = texto.strip()
    limpio = re.sub(r"^```(?:json)?\s*", "", limpio)
    limpio = re.sub(r"\s*```$", "", limpio).strip()

    try:
        datos = json.loads(limpio)
    except json.JSONDecodeError:
        return _vacio("No se pudo interpretar la respuesta del modelo (JSON inválido).")

    if not isinstance(datos, dict):
        return _vacio("La respuesta del modelo no tiene el formato esperado.")

    grafica_bruta = datos.get("grafica") or {}
    if not isinstance(grafica_bruta, dict):
        grafica_bruta = {}
    tipos_validos = {"linea", "barra", "ninguna"}
    grafica = {
        "tipo": grafica_bruta.get("tipo") if grafica_bruta.get("tipo") in tipos_validos else "ninguna",
        "x": grafica_bruta.get("x") or "",
        "y": grafica_bruta.get("y") or "",
        "color": grafica_bruta.get("color") or "",
        "titulo": grafica_bruta.get("titulo") or "",
    }

    return {
        "sql": datos.get("sql") or "",
        "grafica": grafica,
        "fuera_de_alcance": bool(datos.get("fuera_de_alcance", False)),
        "motivo": datos.get("motivo") or "",
    }


def resumir_resultado(cliente: ClienteGemini, pregunta: str, resultado: pd.DataFrame) -> str:
    """Pide al modelo un resumen breve del resultado, usando solo las filas obtenidas."""
    if resultado.empty:
        return "La consulta no devolvió filas para resumir."

    muestra = resultado.head(30).to_csv(index=False)
    truncado = len(resultado) > 30

    system = (
        "Eres un analista que resume, en español, el resultado de una consulta sobre datos "
        "de café. Usa ÚNICAMENTE las cifras que aparecen en el CSV que te dan: no inventes ni "
        "redondees a números distintos, y no menciones países o años que no estén en el CSV. "
        "Responde en 2 o 3 frases, sin repetir la pregunta."
    )
    prompt = (
        f"Pregunta original: {pregunta}\n\n"
        f"Resultado (CSV{', primeras 30 filas' if truncado else ''}):\n{muestra}"
    )
    return cliente.generar(prompt, system=system).strip()
