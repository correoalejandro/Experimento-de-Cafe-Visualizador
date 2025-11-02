# app_experimento_cafe.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import io
import re
from pathlib import Path
import altair as alt

st.set_page_config(page_title="Experimento Sensorial de Café", layout="wide")

# =============================
# 🧭 Sidebar
# =============================
st.sidebar.title("☕ Experimento de Café")
st.sidebar.caption("Carga de datos, opciones y navegación")

# Archivo por defecto y carga
BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_POR_DEFECTO = BASE_DIR / "MuestreoCafe.csv"
archivo = st.sidebar.file_uploader("Sube tu CSV", type=["csv"])
ruta_manual = st.sidebar.text_input("...o escribe la ruta del CSV", value="")

# Mapeo de encabezados → nombres canónicos
# Ajusta a tus nombres reales de columnas (ya configurado para tu CSV)
MAPEO_COLUMNAS = {
    "ID_Participante": "participante_id",
    "Edad": "grupo_edad",
    "Sexo": "sexo",
    "Marca": "tipo_cafe",
    "Olor": "olor",
    "Sabor": "sabor",
    "Acidez": "acidez",
    # "Orden": "orden_presentacion",  # opcional
}

ALIAS_PATTERNS = {
    "participante_id": [r"(?i)^id$", r"(?i)participante", r"(?i)id_?participante"],
    "grupo_edad":      [r"(?i)edad", r"(?i)rango_?edad"],
    "sexo":            [r"(?i)^sexo$", r"(?i)g[eé]nero"],
    "tipo_cafe":       [r"(?i)tipo.*caf[eé]", r"(?i)^caf[eé]$", r"(?i)marca"],
    "orden_presentacion": [r"(?i)orden", r"(?i)orden_?presentaci[oó]n", r"(?i)posici[oó]n"],
    "olor":            [r"(?i)^olor$"],
    "sabor":           [r"(?i)^sabor$"],
    "acidez":          [r"(?i)^acidez$"],
}

ATRIBUTOS = ["olor", "sabor", "acidez"]


def _leer_csv_desde_path(path: Path) -> pd.DataFrame:
    """Lee un CSV intentando diferentes configuraciones de codificación."""
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")


def leer_csv_flexible(file_like_or_path):
    path_candidate = None

    if isinstance(file_like_or_path, Path):
        path_candidate = file_like_or_path
    elif isinstance(file_like_or_path, str):
        candidate = file_like_or_path.strip()
        if candidate:
            path_candidate = Path(candidate).expanduser()

    if path_candidate is not None:
        posibles_rutas = []
        if path_candidate.is_absolute():
            posibles_rutas.append(path_candidate)
        else:
            posibles_rutas.append(Path.cwd() / path_candidate)
            posibles_rutas.append(BASE_DIR / path_candidate)

        for ruta in posibles_rutas:
            if ruta.is_file():
                return _leer_csv_desde_path(ruta)

        raise FileNotFoundError(f"No se encontró el archivo especificado: {path_candidate}")

    elif file_like_or_path is not None:
        # UploadedFile → buffer
        data = file_like_or_path.read()
        buf = io.BytesIO(data)
        try:
            return pd.read_csv(buf, sep=None, engine="python", encoding="utf-8-sig")
        except Exception:
            buf.seek(0)
            return pd.read_csv(buf, sep=None, engine="python", encoding="utf-8")
    else:
        # Intentar por defecto
        if not ARCHIVO_POR_DEFECTO.is_file():
            raise FileNotFoundError(
                f"No se encontró el archivo por defecto en {ARCHIVO_POR_DEFECTO}."
            )
        return _leer_csv_desde_path(ARCHIVO_POR_DEFECTO)

def aplicar_mapeo(df: pd.DataFrame) -> pd.DataFrame:
    ren = {k: v for k, v in MAPEO_COLUMNAS.items() if k in df.columns}
    df = df.rename(columns=ren)
    # alias si falta alguna
    faltan = {"participante_id", "grupo_edad", "sexo", "tipo_cafe", "olor", "sabor", "acidez"} - set(df.columns)
    if faltan:
        alias = {}
        for canonico, patrones in ALIAS_PATTERNS.items():
            if canonico in df.columns:
                continue
            for col in df.columns:
                for pat in patrones:
                    if re.search(pat, str(col)):
                        alias[col] = canonico
                        break
        if alias:
            df = df.rename(columns=alias)
    return df

def convertir_ordinal_a_likert(df: pd.DataFrame) -> pd.DataFrame:
    mapas = {
        "olor":   {"malo": 1, "regular": 3, "bueno": 5, "excelente": 7},
        "sabor":  {"malo": 1, "regular": 3, "bueno": 5, "excelente": 7},
        "acidez": {"baja": 2, "media": 4, "alta": 6},
    }
    for col, mapa in mapas.items():
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.lower().map(mapa)
    return df

def validar_columnas(df: pd.DataFrame):
    requeridas = {"participante_id", "grupo_edad", "sexo", "tipo_cafe", "olor", "sabor", "acidez"}
    faltan = requeridas - set(df.columns)
    if faltan:
        st.error(f"Faltan columnas requeridas: {faltan}")
        st.stop()

# ===== Navegación
pagina = st.sidebar.radio(
    "Ir a:",
    ["🏠 Inicio", "📊 Exploración", "🧪 Pruebas", "⚙️ Ayuda"]
)

# ===== Carga de datos
entrada_usuario = archivo if archivo is not None else (ruta_manual.strip() or None)

if entrada_usuario is None:
    st.sidebar.caption(
        f"Usando archivo por defecto: `{ARCHIVO_POR_DEFECTO.name}` incluido en la app."
    )

try:
    df = leer_csv_flexible(entrada_usuario)
except Exception as e:
    st.error(f"No se pudo leer el CSV: {e}")
    st.stop()

df = aplicar_mapeo(df)
df = convertir_ordinal_a_likert(df)
validar_columnas(df)

# =============================
# 🏠 Inicio
# =============================
if pagina == "🏠 Inicio":
    st.title("🏠 Experimento Sensorial de Café")
    st.markdown("""
    App compacta para **EDA** y **pruebas de hipótesis** en un test sensorial de café.
    - Diseños: **entre-sujetos (Welch)** o **intra-sujetos (apareado)**.
    - Atributos: **olor, sabor, acidez** (Likert 1–7 o convertidos desde texto).
    """)

    st.subheader("Vista previa de datos")
    st.dataframe(df.head(25), use_container_width=True)
    st.caption(f"Filas: {len(df):,} — Columnas: {', '.join(df.columns)}")

# =============================
# 📊 Exploración
# =============================
elif pagina == "📊 Exploración":
    st.title("📊 Exploración de datos")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Descriptivos por marca y atributo")
        piezas = []
        for atr in ATRIBUTOS:
            g = df.groupby("tipo_cafe")[atr].agg(["count", "mean", "std", "median"])
            g["atributo"] = atr
            piezas.append(g.reset_index())
        desc = pd.concat(piezas, ignore_index=True)
        desc = desc[["atributo","tipo_cafe","count","mean","std","median"]].sort_values(["atributo","tipo_cafe"])
        st.dataframe(desc, use_container_width=True)
    with col2:
        st.subheader("Filtros rápidos")
        marcas = sorted(df["tipo_cafe"].dropna().unique().tolist())
        atr = st.selectbox("Atributo", ATRIBUTOS, index=1)
        st.bar_chart(df.groupby("tipo_cafe")[atr].mean())

    st.markdown("---")
    st.subheader("Distribuciones por marca")
    seleccion = st.multiselect("Marcas a comparar", marcas, default=marcas[:2])
    

    if seleccion:
        subset = df[df["tipo_cafe"].isin(seleccion)][["tipo_cafe", atr]].dropna()
        chart = (
            alt.Chart(subset)
            .mark_boxplot(size=40)
            .encode(
                x=alt.X("tipo_cafe:N", title="Marca de café"),
                y=alt.Y(f"{atr}:Q", title=f"Puntuación de {atr}"),
                color="tipo_cafe:N"
            )
            .properties(width=600, height=400)
        )
        st.altair_chart(chart, use_container_width=True)


# =============================
# 🧪 Pruebas
# =============================
elif pagina == "🧪 Pruebas":
    st.title("🧪 Pruebas de hipótesis")
    diseño = st.radio("Selecciona diseño", ["Entre-sujetos (Welch)", "Intra-sujetos (apareado)"], horizontal=True)
    marcas = sorted(df["tipo_cafe"].dropna().unique().tolist())
    ATR = st.multiselect("Atributos a probar", ATRIBUTOS, default=ATRIBUTOS)

    def holm(pvals: np.ndarray) -> np.ndarray:
        orden = np.argsort(pvals)
        m = len(pvals)
        ajust = np.empty_like(pvals, dtype=float)
        for rank, idx in enumerate(orden, start=1):
            ajust[idx] = min((m - rank + 1) * pvals[idx], 1.0)
        return ajust

    resultados = []
    if diseño.startswith("Entre"):
        # t de Welch (grupos independientes)
        for atr in ATR:
            for i in range(len(marcas)):
                for j in range(i+1, len(marcas)):
                    a, b = marcas[i], marcas[j]
                    A = df.loc[df["tipo_cafe"]==a, atr].dropna()
                    B = df.loc[df["tipo_cafe"]==b, atr].dropna()
                    if len(A) < 2 or len(B) < 2:
                        continue
                    tval, pval = stats.ttest_ind(A, B, equal_var=False, nan_policy="omit")
                    resultados.append([atr, a, b, len(A), len(B), float(A.mean()-B.mean()), float(tval), float(pval)])
        cols = ["atributo","cafe_a","cafe_b","n_a","n_b","dif_media_a_menos_b","t","p"]
    else:
        # apareado por participante
        pivots = {atr: df.pivot_table(index="participante_id", columns="tipo_cafe", values=atr, aggfunc="first") for atr in ATR}
        for atr in ATR:
            P = pivots[atr]
            marcas_loc = [m for m in marcas if m in P.columns]
            for i in range(len(marcas_loc)):
                for j in range(i+1, len(marcas_loc)):
                    a, b = marcas_loc[i], marcas_loc[j]
                    sub = P[[a,b]].dropna()
                    if len(sub) < 2:
                        continue
                    tval, pval = stats.ttest_rel(sub[a].values, sub[b].values, nan_policy="omit")
                    diff = float(np.nanmean(sub[a].values - sub[b].values))
                    resultados.append([atr, a, b, int(len(sub)), diff, float(tval), float(pval)])
        cols = ["atributo","cafe_a","cafe_b","n_parejas","dif_media_a_menos_b","t","p"]

    if resultados:
        tabla = pd.DataFrame(resultados, columns=cols)
        # Holm por atributo
        tablas = []
        for atr, sub in tabla.groupby("atributo"):
            sub = sub.copy()
            sub["p_holm"] = holm(sub["p"].values)
            sub["sig_0_05_holm"] = sub["p_holm"] < 0.05
            tablas.append(sub)
        tabla = pd.concat(tablas, ignore_index=True).sort_values(["atributo","p_holm","p"])
        st.subheader("Resultados")
        st.dataframe(tabla, use_container_width=True)
    else:
        st.info("No hay pares comparables suficientes con la configuración actual.")

# =============================
# ⚙️ Ayuda
# =============================
else:
    st.title("⚙️ Ayuda y notas")
    st.markdown("""
- **Columnas requeridas** (nombres canónicos): `participante_id, grupo_edad, sexo, tipo_cafe, olor, sabor, acidez`.
- Si tus encabezados difieren, ajusta el diccionario **MAPEO_COLUMNAS** en el código (lado izquierdo = nombre real; derecho = canónico).
- **Diseño**: usa "Entre-sujetos (Welch)" cuando cada persona prueba solo una marca; "Intra-sujetos (apareado)" cuando cada persona prueba varias marcas.
- Escalas en texto (malo/bueno/alta...) se convierten automáticamente a **Likert** (1–7).
- Las pruebas múltiples se corrigen con **Holm** por atributo.
""")
