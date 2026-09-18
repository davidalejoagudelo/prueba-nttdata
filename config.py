"""
Configuración y credenciales del proyecto.

La API key de Gemini NUNCA va en el código. Se busca en este orden:
    1. st.secrets["GEMINI_API_KEY"]  → Streamlit Community Cloud
    2. variable de entorno GEMINI_API_KEY
    3. archivo .env en la raíz del repo (ignorado por git)
    4. lo que el usuario escriba en la barra lateral de la app

Copia .env.example a .env y pon ahí tu clave para trabajar en local.
"""
from __future__ import annotations

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DATA_DIR = RAIZ / "data"
NOTEBOOKS_DIR = RAIZ / "notebooks"

RUTA_BALANCE = DATA_DIR / "coffee_balance_long.parquet"
ARTEFACTOS_MODELADO = {
    "ranking": DATA_DIR / "modelado_ranking.parquet",
    "ranking_ingenuo": DATA_DIR / "modelado_ranking_ingenuo.parquet",
    "proyeccion": DATA_DIR / "modelado_proyeccion.parquet",
    "backtesting": DATA_DIR / "modelado_backtesting.parquet",
    "calibracion": DATA_DIR / "modelado_calibracion.parquet",
    "modelos": DATA_DIR / "modelado_modelos_elegidos.parquet",
}

NOMBRE_VARIABLE = "GEMINI_API_KEY"


def _leer_env_file(ruta: Path = RAIZ / ".env") -> str:
    """Lee GEMINI_API_KEY de un .env sin dependencias externas."""
    if not ruta.exists():
        return ""
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        if clave.strip() == NOMBRE_VARIABLE:
            return valor.strip().strip('"').strip("'")
    return ""


def obtener_api_key() -> str:
    """Devuelve la clave encontrada, o cadena vacía si no hay ninguna configurada."""
    try:
        import streamlit as st

        valor = st.secrets.get(NOMBRE_VARIABLE, "")
        if valor:
            return str(valor)
    except Exception:
        pass
    return os.environ.get(NOMBRE_VARIABLE, "") or _leer_env_file()


def origen_api_key() -> str:
    """Para mostrar en la UI de dónde salió la clave, sin revelarla."""
    try:
        import streamlit as st

        if st.secrets.get(NOMBRE_VARIABLE, ""):
            return "secrets de Streamlit"
    except Exception:
        pass
    if os.environ.get(NOMBRE_VARIABLE, ""):
        return "variable de entorno"
    if _leer_env_file():
        return "archivo .env"
    return "sin configurar"
