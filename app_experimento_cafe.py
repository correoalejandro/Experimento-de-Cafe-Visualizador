# app_experimento_cafe.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import io
import re
from pathlib import Path
import altair as alt
from scipy.stats import levene, shapiro

st.set_page_config(page_title="Experimento Sensorial de Café", layout="wide")

# =============================
# 🧭 Sidebar
# =============================
st.sidebar.title("☕ Experimento de Café")
st.sidebar.caption("Opciones y navegación")

# ===== Navegación
pagina = st.sidebar.radio(
    "Ir a:",
    ["🏠 Inicio", " Exploración", " Pruebas", " Ayuda"]
)


# Archivo por defecto y carga
BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_POR_DEFECTO = BASE_DIR / "MuestreoCafe_merged.csv"
if ARCHIVO_POR_DEFECTO.is_file():
    st.sidebar.download_button(
        "Descargar datos",
        data=ARCHIVO_POR_DEFECTO.read_bytes(),
        file_name=ARCHIVO_POR_DEFECTO.name,
        mime="text/csv",
        help="Útil para validar el formato esperado antes de subir tus propios datos."
    )
    st.sidebar.info(
        ""
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



# ===== Carga de datos
entrada_usuario = ARCHIVO_POR_DEFECTO
usa_archivo_default = True



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
#  Exploración (tabs dinámicos)
# =============================
elif pagina == " Exploración":
    import altair as alt
    import numpy as np
    import re
    import streamlit as st
    import pandas as pd

    st.title(" Exploración de datos")

    # -------------------------
    # 1) Chequeos de columnas
    # -------------------------
    atributos_presentes = [a for a in ATRIBUTOS if a in df.columns]
    tiene_tipo_cafe = "tipo_cafe" in df.columns
    tiene_demografia = ("edad_num" in df.columns) or ("grupo_edad" in df.columns)
    tiene_sexo = "sexo" in df.columns

    print(f"[DEBUG] atributos_presentes={atributos_presentes}")
    print(f"[DEBUG] tiene_tipo_cafe={tiene_tipo_cafe}, tiene_demografia={tiene_demografia}, tiene_sexo={tiene_sexo}")

    # -----------------------------------
    # 2) Registrar qué tabs sí se pueden
    #    (orden requerido por el usuario)
    # -----------------------------------
    tabs_definicion = []

    # 1) Marcas por atributo (antes "Distribuciones por marca")
    if tiene_tipo_cafe and atributos_presentes:
        tabs_definicion.append(("Marcas por atributo", "render_marcas_por_atributo"))

    # 2) Descriptivos (con columna de Promedios incluida dentro)
    if tiene_tipo_cafe and atributos_presentes:
        tabs_definicion.append(("Descriptivos", "render_descriptivos"))

    # 3) Edades y sexo — demografía
        tabs_definicion.append(("Edades y sexo", "render_boxplots_edad"))

    print(f"[DEBUG] tabs_definicion={tabs_definicion}")

    if not tabs_definicion:
        st.info("No hay suficientes columnas para mostrar secciones de exploración.")
        st.stop()

    # -----------------------------------
    # 3) Helpers para cada tab
    # -----------------------------------
    def render_marcas_por_atributo(dataframe: pd.DataFrame, atributos: list[str]) -> None:
        """Boxplots por marca para un atributo elegido."""
        st.subheader("Marcas por atributo")
        atributo_seleccionado = st.selectbox(
            "Selecciona el atributo a visualizar",
            atributos,
            index=0
        )
        marcas_disponibles = sorted(dataframe["tipo_cafe"].dropna().unique().tolist())
        marcas_seleccionadas = st.multiselect(
            "Marcas a comparar",
            marcas_disponibles,
            default=marcas_disponibles
        )
        if not marcas_seleccionadas:
            st.info("Selecciona al menos una marca para mostrar el gráfico.")
            return

        subset = (
            dataframe[dataframe["tipo_cafe"].isin(marcas_seleccionadas)]
            [["tipo_cafe", atributo_seleccionado]].dropna()
        )
        print(f"[DEBUG] atributo_seleccionado={atributo_seleccionado}, n={len(subset)}")

        grafico = (
            alt.Chart(subset)
            .mark_boxplot(size=40)
            .encode(
                x=alt.X("tipo_cafe:N", title="Marca de café"),
                y=alt.Y(f"{atributo_seleccionado}:Q", title=f"Puntuación de {atributo_seleccionado}"),
                color="tipo_cafe:N"
            )
            .properties(width=600, height=400)
        )
        st.altair_chart(grafico, use_container_width=True)

    def render_descriptivos(dataframe: pd.DataFrame, atributos: list[str]) -> None:
        """Descriptivos por marca y atributo, con una columna lateral de promedios."""
        st.subheader("Descriptivos por marca y atributo")
        columna_izquierda, columna_derecha = st.columns([2, 1])

        # --- Tabla de descriptivos (izquierda) ---
        with columna_izquierda:
            piezas: list[pd.DataFrame] = []
            for atributo in atributos:
                tabla = (
                    dataframe.groupby("tipo_cafe")[atributo]
                    .agg(["count", "mean", "std", "median"])
                    .reset_index()
                )
                tabla.insert(0, "atributo", atributo)
                piezas.append(tabla)

            descriptivos = (
                pd.concat(piezas, ignore_index=True)
                .loc[:, ["atributo", "tipo_cafe", "count", "mean", "std", "median"]]
                .sort_values(["atributo", "tipo_cafe"])
            )
            st.dataframe(descriptivos, use_container_width=True)
            print(f"[DEBUG] descriptivos_rows={len(descriptivos)}")

        # --- Promedio por categoría (derecha) ---
        with columna_derecha:
            st.subheader("Promedio por categoría")
            marcas_unicas = sorted(dataframe["tipo_cafe"].dropna().unique().tolist())
            atributo_para_promedio = st.selectbox("Atributo", atributos, index=0, key="atributo_promedio")
            serie_promedio = dataframe.groupby("tipo_cafe")[atributo_para_promedio].mean()
            st.bar_chart(serie_promedio)
            st.caption(f"[DEBUG] Marcas detectadas: {len(marcas_unicas)}")

    def _midpoint(rango: str) -> float | None:
        """Convierte grupo de edad textual a punto medio numérico (p. ej., '18-30' -> 24)."""
        numeros = re.findall(r"\d+", str(rango))
        if len(numeros) >= 2:
            a, b = map(int, numeros[:2])
            return (a + b) / 2
        if len(numeros) == 1:
            return float(numeros[0])
        return None

    def render_boxplots_edad(dataframe: pd.DataFrame) -> None:
        """Boxplots de edad general y por sexo (si existe), en dos columnas."""
        st.subheader("🧑‍🤝‍🧑 Boxplots de edad")

        dataframe_box = dataframe.copy()
        if "edad_num" not in dataframe_box.columns:
            if "grupo_edad" in dataframe_box.columns:
                dataframe_box["edad_num"] = dataframe_box["grupo_edad"].map(_midpoint)
                print("[DEBUG] edad_num construida a partir de grupo_edad")
            else:
                st.info("No hay 'edad_num' ni 'grupo_edad' para construir boxplots de edad.")
                return

        dataframe_box = dataframe_box.dropna(subset=["edad_num"])
        if dataframe_box.empty:
            st.info("No hay datos de edad para graficar.")
            return

        col_izq, col_der = st.columns(2)

        # --- A) General ---
        with col_izq:
            st.markdown("**General**")
            grafico_general = (
                alt.Chart(dataframe_box)
                .mark_boxplot(size=60)
                .encode(y=alt.Y("edad_num:Q", title="Edad"))
                .properties(width=400, height=300)
            )
            st.altair_chart(grafico_general, use_container_width=True)

        # --- B) Por sexo ---
        with col_der:
            if "sexo" in dataframe_box.columns:
                st.markdown("**Por sexo**")
                grafico_sexo = (
                    alt.Chart(dataframe_box)
                    .mark_boxplot(size=40)
                    .encode(
                        x=alt.X("sexo:N", title="Sexo"),
                        y=alt.Y("edad_num:Q", title="Edad"),
                        color="sexo:N"
                    )
                    .properties(width=400, height=300)
                )
                st.altair_chart(grafico_sexo, use_container_width=True)
            else:
                st.info("No se encontró la columna 'sexo' para el boxplot por sexo.")

    # -----------------------------------
    # 4) Render dinámico por tabs válidos
    # -----------------------------------
    etiquetas_tabs = [nombre for nombre, _ in tabs_definicion]
    st.markdown("---")
    tabs = st.tabs(etiquetas_tabs)
    print(f"[DEBUG] etiquetas_tabs={etiquetas_tabs}")

    for indice, (nombre_tab, funcion_id) in enumerate(tabs_definicion):
        with tabs[indice]:
            if funcion_id == "render_marcas_por_atributo":
                render_marcas_por_atributo(df, atributos_presentes)
            elif funcion_id == "render_descriptivos":
                render_descriptivos(df, atributos_presentes)
            elif funcion_id == "render_boxplots_edad":
                render_boxplots_edad(df)

# =============================
#  Pruebas (tabs dinámicos)
# =============================
elif pagina == " Pruebas":
    import altair as alt
    import numpy as np
    import pandas as pd
    import streamlit as st
    from scipy import stats
    from scipy.stats import levene, shapiro

    st.title(" Pruebas de hipótesis")

    # -------------------------
    # 0) Controles de la vista
    # -------------------------
    diseño = st.radio("Selecciona diseño", ["Entre-sujetos (Welch)", "Intra-sujetos (apareado)"], horizontal=True)
    marcas = sorted(df["tipo_cafe"].dropna().unique().tolist()) if "tipo_cafe" in df.columns else []
    atributos_presentes = [a for a in ATRIBUTOS if a in df.columns]

    print(f"[DEBUG] diseño={diseño}")
    print(f"[DEBUG] marcas={marcas}")
    print(f"[DEBUG] atributos_presentes={atributos_presentes}")

    ATR = st.multiselect("Atributos a probar", atributos_presentes, default=atributos_presentes, key="atr_pruebas_tab")
    if not ATR:
        st.info("No hay atributos seleccionados para probar.")
        st.stop()

    # -------------------------
    # 1) Utils estadísticas
    # -------------------------
    def hedges_g_ind(grupo_a: np.ndarray, grupo_b: np.ndarray) -> float:
        """Hedges g para grupos independientes (con corrección J)."""
        na, nb = len(grupo_a), len(grupo_b)
        if na < 2 or nb < 2:
            return np.nan
        sa2, sb2 = np.var(grupo_a, ddof=1), np.var(grupo_b, ddof=1)
        if (na + nb - 2) <= 0:
            return np.nan
        sp = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))
        if sp <= 0:
            return np.nan
        d = (np.mean(grupo_a) - np.mean(grupo_b)) / sp
        J = 1 - 3 / (4 * (na + nb) - 9) if (na + nb) > 2 else 1.0
        return d * J

    def welch_df(sa2: float, na: int, sb2: float, nb: int) -> float:
        num = (sa2/na + sb2/nb)**2
        den = (sa2**2 / (na**2*(na-1))) + (sb2**2 / (nb**2*(nb-1)))
        return num / den if den > 0 else np.nan

    def welch_ci(a: np.ndarray, b: np.ndarray, alpha: float = 0.05):
        na, nb = len(a), len(b)
        ma, mb = np.mean(a), np.mean(b)
        sa2, sb2 = np.var(a, ddof=1), np.var(b, ddof=1)
        se = np.sqrt(sa2/na + sb2/nb)
        dfw = welch_df(sa2, na, sb2, nb)
        tcrit = stats.t.ppf(1 - alpha/2, dfw) if np.isfinite(dfw) else np.nan
        diff = ma - mb
        return diff, (diff - tcrit*se, diff + tcrit*se), dfw, se

    def holm_ajuste(pvals: np.ndarray) -> np.ndarray:
        orden = np.argsort(pvals)
        m = len(pvals)
        ajust = np.empty_like(pvals, dtype=float)
        for rank, idx in enumerate(orden, start=1):
            ajust[idx] = min((m - rank + 1) * pvals[idx], 1.0)
        return ajust

    # -------------------------
    # 2) Flags por sección
    # -------------------------
    tiene_tipo_cafe = "tipo_cafe" in df.columns and len(marcas) >= 2
    tiene_participante = "participante_id" in df.columns
    tiene_sexo = "sexo" in df.columns
    tiene_atributos = len(ATR) > 0

    puede_supuestos = tiene_tipo_cafe and tiene_atributos
    puede_sexo = tiene_sexo and tiene_tipo_cafe and tiene_atributos

    # -------------------------
    # 3) Construir resultados una vez
    # -------------------------
    resultados = []
    if tiene_atributos and (tiene_tipo_cafe or (diseño.startswith("Intra") and tiene_participante)):
        if diseño.startswith("Entre"):
            for atr in ATR:
                for i in range(len(marcas)):
                    for j in range(i+1, len(marcas)):
                        a, b = marcas[i], marcas[j]
                        A = df.loc[df["tipo_cafe"] == a, atr].dropna().values
                        B = df.loc[df["tipo_cafe"] == b, atr].dropna().values
                        if len(A) < 2 or len(B) < 2:
                            continue
                        tval, pval = stats.ttest_ind(A, B, equal_var=False, nan_policy="omit")
                        diff, ci, dfw, se = welch_ci(A, B)
                        g = hedges_g_ind(A, B)
                        resultados.append({
                            "Atributo sensorial": atr,
                            "Café A": a,
                            "Café B": b,
                            "Participantes (A)": int(len(A)),
                            "Participantes (B)": int(len(B)),
                            "Diferencia de medias (A−B)": float(diff),
                            "IC 95 % inferior": float(ci[0]),
                            "IC 95 % superior": float(ci[1]),
                            "Estadístico t": float(tval),
                            "gl": float(dfw),
                            "p-valor": float(pval),
                            "Tamaño del efecto (Hedges g)": float(g),
                        })
        else:
            if tiene_participante and tiene_tipo_cafe:
                for atr in ATR:
                    P = df.pivot_table(index="participante_id", columns="tipo_cafe", values=atr, aggfunc="first")
                    marcas_loc = [m for m in marcas if m in P.columns]
                    for i in range(len(marcas_loc)):
                        for j in range(i+1, len(marcas_loc)):
                            a, b = marcas_loc[i], marcas_loc[j]
                            sub = P[[a, b]].dropna()
                            if len(sub) < 2:
                                continue
                            tval, pval = stats.ttest_rel(sub[a].values, sub[b].values, nan_policy="omit")
                            diff = float(np.nanmean(sub[a].values - sub[b].values))
                            d = sub[a].values - sub[b].values
                            se = stats.sem(d, nan_policy="omit")
                            tcrit = stats.t.ppf(0.975, df=len(d)-1)
                            ci = (diff - tcrit*se, diff + tcrit*se)
                            resultados.append({
                                "Atributo sensorial": atr,
                                "Café A": a,
                                "Café B": b,
                                "Participantes": int(len(sub)),
                                "Diferencia de medias (A−B)": float(diff),
                                "IC 95 % inferior": float(ci[0]),
                                "IC 95 % superior": float(ci[1]),
                                "Estadístico t": float(tval),
                                "gl": int(len(sub)-1),
                                "p-valor": float(pval),
                                "Tamaño del efecto (Hedges g)": np.nan,
                            })

    tabla_res = pd.DataFrame(resultados) if resultados else pd.DataFrame()
    hay_resultados = not tabla_res.empty

    if hay_resultados:
        # Holm por atributo
        tablas_ajustadas = []
        for atr, sub in tabla_res.groupby("Atributo sensorial", as_index=False):
            sub = sub.copy()
            sub["p-valor ajustado (Holm)"] = holm_ajuste(sub["p-valor"].values)
            sub["Significativo (α = 0.05)"] = sub["p-valor ajustado (Holm)"] < 0.05
            tablas_ajustadas.append(sub)
        tabla_res = pd.concat(tablas_ajustadas, ignore_index=True).sort_values(
            ["Atributo sensorial", "p-valor ajustado (Holm)", "p-valor"]
        )

    print(f"[DEBUG] hay_resultados={hay_resultados}, filas_resultados={0 if not hay_resultados else len(tabla_res)}")

    # -------------------------
    # 4) Armar tabs disponibles
    # -------------------------
    tabs_definicion = []

    if puede_supuestos:
        tabs_definicion.append(("Supuestos", "render_supuestos"))

    if hay_resultados:
        tabs_definicion.append(("Resultados", "render_resultados"))
        tabs_definicion.append(("Interpretación", "render_interpretacion"))

    if puede_sexo:
        tabs_definicion.append(("Comparaciones por sexo", "render_sexo"))

    print(f"[DEBUG] tabs_definicion={tabs_definicion}")

    if not tabs_definicion:
        st.info("No hay suficientes datos para mostrar secciones de Pruebas.")
        st.stop()

    # -------------------------
    # 5) Render functions
    # -------------------------
    def render_supuestos() -> None:
        st.markdown("### Pruebas de supuestos")
        # Levene por atributo (entre marcas) y Shapiro global del atributo
        for atr in ATR:
            if atr not in df.columns:
                continue
            grupos = [g[atr].dropna() for _, g in df.groupby("tipo_cafe")] if "tipo_cafe" in df.columns else []
            suficientemente_grande = all(len(g) > 2 for g in grupos) and len(grupos) >= 2
            if suficientemente_grande:
                stat_lev, p_lev = levene(*grupos)
                stat_sh, p_sh = shapiro(df[atr].dropna()) if len(df[atr].dropna()) >= 3 else (np.nan, np.nan)
                st.write(f"**{atr.capitalize()}** — Levene p = {p_lev:.3f}, Shapiro p = {p_sh:.3f}")
            else:
                st.info(f"No hay suficientes datos para supuestos en **{atr}**.")

    def render_resultados() -> None:
        st.subheader("Resultados (resumen)")
        cols_entre = [
            "Atributo sensorial","Café A","Café B",
            "Participantes (A)","Participantes (B)",
            "Diferencia de medias (A−B)","IC 95 % inferior","IC 95 % superior",
            "Estadístico t","gl","p-valor","p-valor ajustado (Holm)","Significativo (α = 0.05)","Tamaño del efecto (Hedges g)"
        ]
        cols_apar = [
            "Atributo sensorial","Café A","Café B",
            "Participantes",
            "Diferencia de medias (A−B)","IC 95 % inferior","IC 95 % superior",
            "Estadístico t","gl","p-valor","p-valor ajustado (Holm)","Significativo (α = 0.05)","Tamaño del efecto (Hedges g)"
        ]
        columnas = cols_entre if diseño.startswith("Entre") else cols_apar
        columnas = [c for c in columnas if c in tabla_res.columns]
        st.dataframe(tabla_res[columnas], use_container_width=True)
        st.caption(f"[DEBUG] columnas_mostradas={columnas}")

    def render_interpretacion() -> None:
        st.subheader("Interpretación")
        atr_sel = st.selectbox("Atributo", ATR, index=0, key="atr_interp_tab")
        sig = (
            tabla_res[(tabla_res["Atributo sensorial"] == atr_sel) & (tabla_res["p-valor ajustado (Holm)"] < 0.05)]
            .copy()
            .sort_values("p-valor ajustado (Holm)")
        )
        st.markdown("**✅ Diferencias significativas (Holm < 0.05)**")
        if sig.empty:
            st.info("No se detectaron diferencias significativas para este atributo.")
            return

        for _, r in sig.iterrows():
            a, b = r["Café A"], r["Café B"]
            diff = r["Diferencia de medias (A−B)"]
            ci_lo, ci_hi = r["IC 95 % inferior"], r["IC 95 % superior"]
            p_adj = r["p-valor ajustado (Holm)"]
            na = str(a).replace("_", " ")
            nb = str(b).replace("_", " ")

            if diff > 0:
                comparacion = f"**{na} obtuvo en promedio {abs(diff):.2f} puntos más que {nb}**."
            elif diff < 0:
                comparacion = f"**{na} obtuvo en promedio {abs(diff):.2f} puntos menos que {nb}**."
            else:
                comparacion = f"**{na} y {nb} obtuvieron promedios iguales**."

            st.markdown(
                f"{comparacion} "
                f"IC95% = [{ci_lo:.2f}, {ci_hi:.2f}] • "
                f"p(Holm) = {p_adj:.4f}."
            )

    def render_sexo() -> None:
        st.subheader("Comparaciones por sexo")
        st.markdown("""
        - Cada fila compara **hombres vs mujeres** para un **atributo** dentro de una **marca**.
        - **t**: magnitud y dirección (signo). **p**: evidencia estadística (p < 0.05 → significativa).
        """)
        filas = []
        for atr in ATR:
            if atr not in df.columns:
                continue
            for cafe in df["tipo_cafe"].dropna().unique():
                sub = df[df["tipo_cafe"] == cafe]
                if "sexo" not in sub.columns:
                    continue
                # armoniza valores
                sexo_series = sub["sexo"].astype(str).str.upper().str.strip()
                gH = sub[sexo_series.eq("M")][atr].dropna().values
                gM = sub[sexo_series.eq("F")][atr].dropna().values
                if len(gH) > 2 and len(gM) > 2:
                    tval, pval = stats.ttest_ind(gH, gM, equal_var=False)
                    filas.append(
                        {"Atributo": atr, "Marca": cafe, "t": float(tval), "p": float(pval),
                         "n_H": int(len(gH)), "n_M": int(len(gM))}
                    )
        if not filas:
            st.info("No hay suficientes datos por sexo para mostrar comparaciones.")
            return
        tabla_sexo = pd.DataFrame(filas).sort_values(["Atributo", "p"])
        st.dataframe(tabla_sexo, use_container_width=True)
        st.caption(f"[DEBUG] comparaciones_sexo={len(tabla_sexo)}")

    # -------------------------
    # 6) Render por tabs válidos
    # -------------------------
    etiquetas_tabs = [nombre for nombre, _ in tabs_definicion]
    tabs = st.tabs(etiquetas_tabs)
    print(f"[DEBUG] etiquetas_tabs={etiquetas_tabs}")

    for i, (nombre_tab, fn_id) in enumerate(tabs_definicion):
        with tabs[i]:
            if fn_id == "render_supuestos":
                render_supuestos()
            elif fn_id == "render_resultados":
                render_resultados()
            elif fn_id == "render_interpretacion":
                render_interpretacion()
            elif fn_id == "render_sexo":
                render_sexo()



# =============================
#  Ayuda
# =============================
else:
    st.title(" Ayuda y notas")
    st.markdown("""
- **Columnas requeridas** (nombres canónicos): `participante_id, grupo_edad, sexo, tipo_cafe, olor, sabor, acidez`.
- Si tus encabezados difieren, ajusta el diccionario **MAPEO_COLUMNAS** en el código (lado izquierdo = nombre real; derecho = canónico).
- **Diseño**: usa "Entre-sujetos (Welch)" cuando cada persona prueba solo una marca; "Intra-sujetos (apareado)" cuando cada persona prueba varias marcas.
- Escalas en texto (malo/bueno/alta...) se convierten automáticamente a **Likert** (1–7).
- Las pruebas múltiples se corrigen con **Holm** por atributo.
""")
