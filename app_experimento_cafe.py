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
ARCHIVO_POR_DEFECTO = BASE_DIR / "MuestreoCafe_merged.csv"
archivo = st.sidebar.file_uploader("Sube tu CSV", type=["csv"])
ruta_manual = st.sidebar.text_input("...o escribe la ruta del CSV", value="")

if ARCHIVO_POR_DEFECTO.is_file():
    st.sidebar.download_button(
        "Descargar CSV de ejemplo",
        data=ARCHIVO_POR_DEFECTO.read_bytes(),
        file_name=ARCHIVO_POR_DEFECTO.name,
        mime="text/csv",
        help="Útil para validar el formato esperado antes de subir tus propios datos."
    )
    st.sidebar.info(
        "Si no subes un archivo propio se cargará automáticamente "
        f"`{ARCHIVO_POR_DEFECTO.name}`. Puedes reemplazarlo subiendo un CSV "
        "con el botón anterior o escribiendo una ruta personalizada."
    )

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
usa_archivo_default = entrada_usuario is None

if usa_archivo_default:
    st.sidebar.caption(
        f"Usando archivo por defecto: `{ARCHIVO_POR_DEFECTO.name}` incluido en la app."
    )

try:
    df = leer_csv_flexible(entrada_usuario)
except Exception as e:
    st.error(f"No se pudo leer el CSV: {e}")
    st.stop()

if usa_archivo_default:
    fuente_datos = f"{ARCHIVO_POR_DEFECTO.name} (predefinido)"
elif archivo is not None:
    fuente_datos = f"{archivo.name} (archivo subido)"
else:
    fuente_datos = f"{Path(ruta_manual.strip()).name or ruta_manual.strip()} (ruta manual)"

st.sidebar.caption(f"Fuente actual: {fuente_datos}")

df = aplicar_mapeo(df)
df = convertir_ordinal_a_likert(df)
validar_columnas(df)


# 🏠 Inicio
# =============================
if pagina == "🏠 Inicio":
    st.title("🏠 Experimento Sensorial de Café")
    st.markdown("""
    App compacta para **EDA** y **pruebas de hipótesis** en un test sensorial de café.
    - Diseños: **entre-sujetos (Welch)** o **intra-sujetos (apareado)**.
    - Atributos: **olor, sabor, acidez** (Likert 1–7 o convertidos desde texto).
    """)

    st.subheader("Vista completa de datos")
    st.caption(f"Fuente de datos actual: {fuente_datos}")
    st.dataframe(df, use_container_width=True)
    participantes = df["participante_id"].nunique() if "participante_id" in df.columns else "?"
    st.caption(
        f"Filas: {len(df):,} — Columnas: {', '.join(df.columns)} — "
        f"Participantes únicos: {participantes}"
    )

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
    seleccion = st.multiselect("Marcas a comparar", marcas, default=marcas)
    

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
    import altair as alt
    st.title("🧪 Pruebas de hipótesis")
    diseño = st.radio("Selecciona diseño", ["Entre-sujetos (Welch)", "Intra-sujetos (apareado)"], horizontal=True)
    marcas = sorted(df["tipo_cafe"].dropna().unique().tolist())
    ATR = st.multiselect("Atributos a probar", ATRIBUTOS, default=ATRIBUTOS)

    # ---- utilidades efecto y CI ----
    import numpy as np
    from scipy import stats

    def hedges_g_ind(a, b):
        """Hedges g para grupos independientes (aprox. aun con varianzas desiguales)."""
        na, nb = len(a), len(b)
        sa2, sb2 = np.var(a, ddof=1), np.var(b, ddof=1)
        sp = np.sqrt(((na-1)*sa2 + (nb-1)*sb2) / (na+nb-2)) if (na+nb-2)>0 else np.nan
        d = (np.mean(a) - np.mean(b)) / sp if sp>0 else np.nan
        # corrección de Hedges
        J = 1 - 3/(4*(na+nb)-9) if (na+nb)>2 else 1.0
        return d*J

    def welch_df(sa2, na, sb2, nb):
        num = (sa2/na + sb2/nb)**2
        den = (sa2**2 / (na**2*(na-1))) + (sb2**2 / (nb**2*(nb-1)))
        return num/den

    def welch_ci(a, b, alpha=0.05):
        """IC para diferencia de medias (A-B) con Welch."""
        na, nb = len(a), len(b)
        ma, mb = np.mean(a), np.mean(b)
        sa2, sb2 = np.var(a, ddof=1), np.var(b, ddof=1)
        se = np.sqrt(sa2/na + sb2/nb)
        dfw = welch_df(sa2, na, sb2, nb)
        tcrit = stats.t.ppf(1 - alpha/2, dfw)
        diff = ma - mb
        return diff, (diff - tcrit*se, diff + tcrit*se), dfw, se

    def holm(pvals: np.ndarray) -> np.ndarray:
        orden = np.argsort(pvals)
        m = len(pvals)
        ajust = np.empty_like(pvals, dtype=float)
        for rank, idx in enumerate(orden, start=1):
            ajust[idx] = min((m - rank + 1) * pvals[idx], 1.0)
        return ajust

    # ---- construir resultados ----
    resultados = []
    if diseño.startswith("Entre"):
        for atr in ATR:
            for i in range(len(marcas)):
                for j in range(i+1, len(marcas)):
                    a, b = marcas[i], marcas[j]
                    A = df.loc[df["tipo_cafe"]==a, atr].dropna()
                    B = df.loc[df["tipo_cafe"]==b, atr].dropna()
                    if len(A) < 2 or len(B) < 2:
                        continue
                    tval, pval = stats.ttest_ind(A, B, equal_var=False, nan_policy="omit")
                    diff, ci, dfw, se = welch_ci(A, B)
                    g = hedges_g_ind(A, B)
                    resultados.append({
                        "atributo": atr, "cafe_a": a, "cafe_b": b,
                        "n_a": int(len(A)), "n_b": int(len(B)),
                        "dif_media_a_menos_b": float(diff),
                        "ci95_inf": float(ci[0]), "ci95_sup": float(ci[1]),
                        "t": float(tval), "gl_welch": float(dfw), "p": float(pval),
                        "hedges_g": float(g)
                    })
        cols = ["atributo","cafe_a","cafe_b","n_a","n_b","dif_media_a_menos_b","ci95_inf","ci95_sup","t","gl_welch","p","hedges_g"]
    else:
        # apareado por participante
        for atr in ATR:
            P = df.pivot_table(index="participante_id", columns="tipo_cafe", values=atr, aggfunc="first")
            marcas_loc = [m for m in marcas if m in P.columns]
            for i in range(len(marcas_loc)):
                for j in range(i+1, len(marcas_loc)):
                    a, b = marcas_loc[i], marcas_loc[j]
                    sub = P[[a,b]].dropna()
                    if len(sub) < 2:
                        continue
                    tval, pval = stats.ttest_rel(sub[a].values, sub[b].values, nan_policy="omit")
                    diff = float(np.nanmean(sub[a].values - sub[b].values))
                    # IC pareado
                    d = sub[a].values - sub[b].values
                    se = stats.sem(d, nan_policy="omit")
                    tcrit = stats.t.ppf(0.975, df=len(d)-1)
                    ci = (diff - tcrit*se, diff + tcrit*se)
                    resultados.append({
                        "atributo": atr, "cafe_a": a, "cafe_b": b,
                        "n_parejas": int(len(sub)),
                        "dif_media_a_menos_b": float(diff),
                        "ci95_inf": float(ci[0]), "ci95_sup": float(ci[1]),
                        "t": float(tval), "gl": int(len(sub)-1), "p": float(pval),
                        "hedges_g": np.nan  # opcional en apareado
                    })
        cols = ["atributo","cafe_a","cafe_b","n_parejas","dif_media_a_menos_b","ci95_inf","ci95_sup","t","gl","p","hedges_g"]

    if not resultados:
        st.info("No hay comparaciones posibles con la configuración actual.")
        st.stop()

    tabla = pd.DataFrame(resultados, columns=cols)

    # Holm por atributo
    tablas = []
    for atr, sub in tabla.groupby("atributo", as_index=False):
        sub = sub.copy()
        sub["p_holm"] = holm(sub["p"].values)
        sub["sig_0_05_holm"] = sub["p_holm"] < 0.05
        tablas.append(sub)
    tabla = pd.concat(tablas, ignore_index=True).sort_values(["atributo","p_holm","p"])

    # ===== Layout compacto y legible =====
    st.subheader("Resultados (resumen)")
    st.dataframe(
        tabla[["atributo","cafe_a","cafe_b",
            *(["n_a","n_b"] if diseño.startswith("Entre") else ["n_parejas"]),
            "dif_media_a_menos_b","ci95_inf","ci95_sup","p_holm","sig_0_05_holm","hedges_g"]],
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Interpretación")

    atr_sel = st.selectbox("Atributo", ATR, index=0, key="atr_interp")

    # 1) Ranking de medias (más fácil de leer)
    medias = (
        df.groupby("tipo_cafe")[atr_sel]
        .mean()
        .sort_values(ascending=False)
    )
    st.markdown("**🏆 Ranking de medias**")
    for i, (marca, val) in enumerate(medias.items(), start=1):
        st.markdown(f"{i}. **{marca}** — {val:.2f}")

    st.markdown("---")

    # 2) Solo comparaciones significativas (Holm < 0.05), ordenadas por p_holm
    sig = (
        tabla[(tabla["atributo"] == atr_sel) & (tabla["p_holm"] < 0.05)]
        .copy()
        .sort_values("p_holm")
    )

    st.markdown("**✅ Diferencias significativas (Holm < 0.05)**")
    if sig.empty:
        st.info("No se detectaron diferencias significativas para este atributo.")
    else:
        for _, r in sig.iterrows():
            a, b = r["cafe_a"], r["cafe_b"]
            diff  = r["dif_media_a_menos_b"]
            ci_lo, ci_hi = r["ci95_inf"], r["ci95_sup"]
            arrow = "→" if diff > 0 else "←"
            st.markdown(
                f"- **{a} {arrow} {b}**: Δ = **{abs(diff):.2f}** "
                f"(IC95% [{ci_lo:.2f}, {ci_hi:.2f}]; p(Holm) = {r['p_holm']:.4f})"
            )

    # (Opcional) botón para ver todas las comparaciones en bruto
    with st.expander("Ver todas las comparaciones (completo)"):
        todas = tabla[tabla["atributo"] == atr_sel].copy()
        todas["Δ_abs"] = todas["dif_media_a_menos_b"].abs()
        st.dataframe(
            todas.sort_values(["p_holm","Δ_abs"]),
            use_container_width=True
        )



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
