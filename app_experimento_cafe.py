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
    # Entradas del usuario
    # -------------------------
    diseño = st.radio(
        "Selecciona diseño",
        ["Entre-sujetos (Welch)", "Intra-sujetos (apareado)"],
        horizontal=True
    )
    marcas_disponibles = sorted(df["tipo_cafe"].dropna().unique().tolist()) if "tipo_cafe" in df.columns else []
    atributos_presentes = [a for a in ATRIBUTOS if a in df.columns]
    ATR = st.multiselect("Atributos a probar", ATRIBUTOS, default=atributos_presentes or ATRIBUTOS)

    # -------------------------
    # Chequeos de columnas
    # -------------------------
    tiene_tipo_cafe = "tipo_cafe" in df.columns
    tiene_participante = "participante_id" in df.columns
    tiene_sexo = "sexo" in df.columns
    hay_atributos = len(atributos_presentes) > 0

    print(f"[DEBUG] tiene_tipo_cafe={tiene_tipo_cafe}, tiene_participante={tiene_participante}, tiene_sexo={tiene_sexo}")
    print(f"[DEBUG] atributos_presentes={atributos_presentes}, marcas_disponibles={marcas_disponibles}")

    # -------------------------
    # Utilidades estadísticas
    # -------------------------
    def hedges_g_ind(grupo_a: np.ndarray, grupo_b: np.ndarray) -> float:
        """Hedges g para grupos independientes (aprox. válido con varianzas desiguales)."""
        n_a, n_b = len(grupo_a), len(grupo_b)
        var_a, var_b = np.var(grupo_a, ddof=1), np.var(grupo_b, ddof=1)
        if (n_a + n_b - 2) <= 0:
            return np.nan
        desviacion_ponderada = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        if not np.isfinite(desviacion_ponderada) or desviacion_ponderada <= 0:
            return np.nan
        d_cohen = (np.mean(grupo_a) - np.mean(grupo_b)) / desviacion_ponderada
        correccion_hedges = 1 - 3 / (4 * (n_a + n_b) - 9) if (n_a + n_b) > 2 else 1.0
        return d_cohen * correccion_hedges

    def welch_df(var_a, n_a, var_b, n_b) -> float:
        numerador = (var_a / n_a + var_b / n_b) ** 2
        denominador = (var_a ** 2) / (n_a ** 2 * (n_a - 1)) + (var_b ** 2) / (n_b ** 2 * (n_b - 1))
        if denominador == 0:
            return np.nan
        return numerador / denominador

    def welch_ci(grupo_a, grupo_b, alpha=0.05):
        n_a, n_b = len(grupo_a), len(grupo_b)
        media_a, media_b = np.mean(grupo_a), np.mean(grupo_b)
        var_a, var_b = np.var(grupo_a, ddof=1), np.var(grupo_b, ddof=1)
        error_estandar = np.sqrt(var_a / n_a + var_b / n_b)
        gl_welch = welch_df(var_a, n_a, var_b, n_b)
        if not np.isfinite(gl_welch) or error_estandar <= 0:
            return np.nan, (np.nan, np.nan), np.nan, np.nan
        t_critico = stats.t.ppf(1 - alpha / 2, gl_welch)
        diferencia = media_a - media_b
        return diferencia, (diferencia - t_critico * error_estandar, diferencia + t_critico * error_estandar), gl_welch, error_estandar

    def ajuste_holm(p_values: np.ndarray) -> np.ndarray:
        indices_orden = np.argsort(p_values)
        total = len(p_values)
        ajustados = np.empty_like(p_values, dtype=float)
        for rango, idx in enumerate(indices_orden, start=1):
            ajustados[idx] = min((total - rango + 1) * p_values[idx], 1.0)
        return ajustados

    # -------------------------
    # 1) Supuestos (si aplica)
    # -------------------------
    supuestos_posibles = tiene_tipo_cafe and hay_atributos
    resultados_supuestos = []
    if supuestos_posibles:
        st.markdown("###  Pruebas de supuestos (vista previa)")
        for atributo in ATR:
            if atributo not in df.columns:
                continue
            grupos = [subgrupo[atributo].dropna() for _, subgrupo in df.groupby("tipo_cafe")]
            grupos_validos = [g for g in grupos if len(g) > 2]
            if len(grupos_validos) >= 2:
                estadistico_levene, p_levene = levene(*grupos_validos)
                estadistico_shapiro, p_shapiro = shapiro(df[atributo].dropna())
                resultados_supuestos.append((atributo, p_levene, p_shapiro))
                st.write(f"**{atributo.capitalize()}** — Levene p = {p_levene:.3f}, Shapiro p = {p_shapiro:.3f}")
            else:
                st.info(f"No hay suficientes datos para pruebas de supuestos en **{atributo}**.")
    else:
        print("[DEBUG] Supuestos no aplican: faltan columnas o atributos.")

    # -------------------------
    # 2) Resultados (según diseño)
    # -------------------------
    resultados_lista = []

    if diseño.startswith("Entre") and tiene_tipo_cafe and hay_atributos and len(marcas_disponibles) >= 2:
        print("[DEBUG] Computando Welch (entre-sujetos)")
        for atributo in ATR:
            if atributo not in df.columns:
                continue
            for indice_a in range(len(marcas_disponibles)):
                for indice_b in range(indice_a + 1, len(marcas_disponibles)):
                    nombre_a = marcas_disponibles[indice_a]
                    nombre_b = marcas_disponibles[indice_b]
                    grupo_a = df.loc[df["tipo_cafe"] == nombre_a, atributo].dropna().values
                    grupo_b = df.loc[df["tipo_cafe"] == nombre_b, atributo].dropna().values
                    if len(grupo_a) < 2 or len(grupo_b) < 2:
                        continue
                    estadistico_t, p_valor = stats.ttest_ind(grupo_a, grupo_b, equal_var=False, nan_policy="omit")
                    diferencia, intervalo, gl_w, error_estandar = welch_ci(grupo_a, grupo_b)
                    g_hedges = hedges_g_ind(grupo_a, grupo_b)
                    resultados_lista.append({
                        "Atributo sensorial": atributo,
                        "Café A": nombre_a,
                        "Café B": nombre_b,
                        "Participantes (A)": int(len(grupo_a)),
                        "Participantes (B)": int(len(grupo_b)),
                        "Diferencia de medias (A−B)": float(diferencia),
                        "IC 95 % inferior": float(intervalo[0]),
                        "IC 95 % superior": float(intervalo[1]),
                        "Estadístico t": float(estadistico_t),
                        "gl": float(gl_w),
                        "p-valor": float(p_valor),
                        "Tamaño del efecto (Hedges g)": float(g_hedges),
                        "_diseño": "entre"
                    })

    elif (not diseño.startswith("Entre")) and tiene_participante and tiene_tipo_cafe and hay_atributos:
        print("[DEBUG] Computando apareado (intra-sujetos)")
        for atributo in ATR:
            if atributo not in df.columns:
                continue
            tabla_pareada = df.pivot_table(index="participante_id", columns="tipo_cafe", values=atributo, aggfunc="first")
            marcas_en_tabla = [m for m in marcas_disponibles if m in tabla_pareada.columns]
            for indice_a in range(len(marcas_en_tabla)):
                for indice_b in range(indice_a + 1, len(marcas_en_tabla)):
                    nombre_a = marcas_en_tabla[indice_a]
                    nombre_b = marcas_en_tabla[indice_b]
                    sub = tabla_pareada[[nombre_a, nombre_b]].dropna()
                    if len(sub) < 2:
                        continue
                    estadistico_t, p_valor = stats.ttest_rel(sub[nombre_a].values, sub[nombre_b].values, nan_policy="omit")
                    diferencia = float(np.nanmean(sub[nombre_a].values - sub[nombre_b].values))
                    diferencias = sub[nombre_a].values - sub[nombre_b].values
                    error_estandar = stats.sem(diferencias, nan_policy="omit")
                    t_critico = stats.t.ppf(0.975, df=len(diferencias) - 1)
                    intervalo = (diferencia - t_critico * error_estandar, diferencia + t_critico * error_estandar)
                    resultados_lista.append({
                        "Atributo sensorial": atributo,
                        "Café A": nombre_a,
                        "Café B": nombre_b,
                        "Participantes": int(len(sub)),
                        "Diferencia de medias (A−B)": float(diferencia),
                        "IC 95 % inferior": float(intervalo[0]),
                        "IC 95 % superior": float(intervalo[1]),
                        "Estadístico t": float(estadistico_t),
                        "gl": int(len(diferencias) - 1),
                        "p-valor": float(p_valor),
                        "Tamaño del efecto (Hedges g)": np.nan,
                        "_diseño": "apareado"
                    })
    else:
        print("[DEBUG] No hay condiciones para comparar según el diseño elegido.")

    # -------------------------
    # 3) Tabla de resultados + Holm
    # -------------------------
    if resultados_lista:
        tabla_resultados = pd.DataFrame(resultados_lista)

        tablas_con_ajuste = []
        for atributo, subtabla in tabla_resultados.groupby("Atributo sensorial", as_index=False):
            subtabla = subtabla.copy()
            p_values = subtabla["p-valor"].values
            p_ajustados = ajuste_holm(p_values)
            subtabla["p-valor ajustado (Holm)"] = p_ajustados
            subtabla["Significativo (α = 0.05)"] = subtabla["p-valor ajustado (Holm)"] < 0.05
            tablas_con_ajuste.append(subtabla)

        tabla_resultados = pd.concat(tablas_con_ajuste, ignore_index=True).sort_values(
            ["Atributo sensorial", "p-valor ajustado (Holm)", "p-valor"]
        )
        hay_resultados = True
        print(f"[DEBUG] Comparaciones totales={len(tabla_resultados)}")
    else:
        tabla_resultados = pd.DataFrame()
        hay_resultados = False
        print("[DEBUG] Sin resultados de comparación.")

    # -------------------------
    # 4) Construcción dinámica de tabs
    # -------------------------
    tabs_definicion = []

    if supuestos_posibles:
        tabs_definicion.append(("Supuestos", "render_supuestos"))

    if hay_resultados:
        tabs_definicion.append(("Resultados", "render_resultados"))
        tabs_definicion.append(("Interpretación", "render_interpretacion"))

    # Comparaciones por sexo requiere sexo y tipo_cafe y al menos una marca
    comparacion_sexo_posible = tiene_sexo and tiene_tipo_cafe and len(marcas_disponibles) > 0 and hay_atributos
    if comparacion_sexo_posible:
        tabs_definicion.append(("Comparaciones por sexo", "render_sexo"))

    print(f"[DEBUG] tabs_definicion={tabs_definicion}")

    if not tabs_definicion:
        st.info("No hay secciones disponibles en 'Pruebas' con los datos actuales.")
        st.stop()

    # -------------------------
    # 5) Render de cada tab
    # -------------------------
    etiquetas_tabs = [nombre for nombre, _ in tabs_definicion]
    tabs = st.tabs(etiquetas_tabs)
    print(f"[DEBUG] etiquetas_tabs={etiquetas_tabs}")

    def render_supuestos():
        st.subheader("Pruebas de supuestos")
        if not resultados_supuestos:
            st.info("No fue posible calcular supuestos con los datos actuales.")
            return
        for atributo, p_levene, p_shapiro in resultados_supuestos:
            st.write(f"**{atributo.capitalize()}** — Levene p = {p_levene:.3f}, Shapiro p = {p_shapiro:.3f}")

    def render_resultados():
        st.subheader("Resultados (resumen)")
        if tabla_resultados.empty:
            st.info("No hay comparaciones posibles con la configuración actual.")
            return

        columnas_entre = [
            "Atributo sensorial", "Café A", "Café B",
            "Participantes (A)", "Participantes (B)",
            "Diferencia de medias (A−B)", "IC 95 % inferior", "IC 95 % superior",
            "Estadístico t", "gl", "p-valor", "p-valor ajustado (Holm)",
            "Significativo (α = 0.05)", "Tamaño del efecto (Hedges g)"
        ]
        columnas_apareado = [
            "Atributo sensorial", "Café A", "Café B",
            "Participantes",
            "Diferencia de medias (A−B)", "IC 95 % inferior", "IC 95 % superior",
            "Estadístico t", "gl", "p-valor", "p-valor ajustado (Holm)",
            "Significativo (α = 0.05)", "Tamaño del efecto (Hedges g)"
        ]

        if diseño.startswith("Entre"):
            columnas_a_mostrar = [c for c in columnas_entre if c in tabla_resultados.columns]
        else:
            columnas_a_mostrar = [c for c in columnas_apareado if c in tabla_resultados.columns]

        st.dataframe(
            tabla_resultados[columnas_a_mostrar],
            use_container_width=True
        )

    def render_interpretacion():
        st.subheader("Interpretación")
        if tabla_resultados.empty:
            st.info("No hay resultados para interpretar.")
            return

        atributo_seleccionado = st.selectbox(
            "Atributo",
            sorted(tabla_resultados["Atributo sensorial"].unique().tolist()),
            index=0,
            key="atributo_interpretacion"
        )
        significativas = (
            tabla_resultados[
                (tabla_resultados["Atributo sensorial"] == atributo_seleccionado) &
                (tabla_resultados["p-valor ajustado (Holm)"] < 0.05)
            ].copy().sort_values("p-valor ajustado (Holm)")
        )

        st.markdown("**✅ Diferencias significativas (Holm < 0.05)**")
        if significativas.empty:
            st.info("No se detectaron diferencias significativas para este atributo.")
            return

        for _, fila in significativas.iterrows():
            cafe_a = fila["Café A"]
            cafe_b = fila["Café B"]
            diferencia = fila["Diferencia de medias (A−B)"]
            limite_inferior, limite_superior = fila["IC 95 % inferior"], fila["IC 95 % superior"]
            p_ajustado = fila["p-valor ajustado (Holm)"]
            nombre_a = cafe_a.replace("_", " ")
            nombre_b = cafe_b.replace("_", " ")

            if diferencia > 0:
                frase = f"**{nombre_a} obtuvo en promedio {abs(diferencia):.2f} puntos más que {nombre_b}**."
            elif diferencia < 0:
                frase = f"**{nombre_a} obtuvo en promedio {abs(diferencia):.2f} puntos menos que {nombre_b}**."
            else:
                frase = f"**{nombre_a} y {nombre_b} obtuvieron promedios iguales**."

            st.markdown(
                f"{frase} "
                f"La diferencia observada es de **{abs(diferencia):.2f}** puntos. "
                f"El IC95% va de **{limite_inferior:.2f}** a **{limite_superior:.2f}**. "
                f"**p(Holm) = {p_ajustado:.4f}**."
            )

    def render_sexo():
        st.subheader("Comparaciones por sexo")
        if not comparacion_sexo_posible:
            st.info("No hay datos suficientes para comparar por sexo.")
            return

        st.markdown("**Cómo leer:** cada línea compara **hombres vs mujeres** para un **atributo** dentro de una **marca**.")
        for atributo in ATR:
            if atributo not in df.columns:
                continue
            for marca in df["tipo_cafe"].dropna().unique():
                sub = df[df["tipo_cafe"] == marca]
                # Convertir sexo a texto consistente
                serie_sexo = sub["sexo"].astype(str).str.upper() if "sexo" in sub.columns else pd.Series([], dtype=str)
                grupo_h = sub.loc[serie_sexo == "H", atributo].dropna()
                grupo_m = sub.loc[serie_sexo == "M", atributo].dropna()
                if len(grupo_h) > 2 and len(grupo_m) > 2:
                    estadistico_t, p_valor = stats.ttest_ind(grupo_h, grupo_m, equal_var=False)
                    st.write(f"{atributo.capitalize()} ({marca}) → t = {estadistico_t:.2f}, p = {p_valor:.3f}")

    # Pintar tabs efectivos
    for indice, (nombre_tab, identificador) in enumerate(tabs_definicion):
        with tabs[indice]:
            if identificador == "render_supuestos":
                render_supuestos()
            elif identificador == "render_resultados":
                render_resultados()
            elif identificador == "render_interpretacion":
                render_interpretacion()
            elif identificador == "render_sexo":
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
