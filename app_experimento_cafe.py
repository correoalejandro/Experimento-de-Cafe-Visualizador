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
#  Exploración (orden y estructura mejorada)
# =============================
elif pagina == " Exploración":
    import altair as alt
    import numpy as np
    import re
    import pandas as pd
    import streamlit as st

    st.title(" Exploración de datos")

    # --- Chequeos de columnas ---
    atributos_presentes = [a for a in ATRIBUTOS if a in df.columns]
    tiene_tipo_cafe = "tipo_cafe" in df.columns
    tiene_demografia = ("edad_num" in df.columns) or ("grupo_edad" in df.columns)

    # --- Tabs: primero comparación de marcas, luego descriptivos y edad ---
    tabs_disponibles = []
    if tiene_tipo_cafe and atributos_presentes:
        tabs_disponibles.append(("Comparación de marcas", "render_distribuciones"))
        tabs_disponibles.append(("Descriptivos", "render_descriptivos"))
    if tiene_demografia:
        tabs_disponibles.append(("Edad (boxplots)", "render_boxplots_edad"))

    if not tabs_disponibles:
        st.info("No hay suficientes columnas para mostrar secciones de exploración.")
        st.stop()

    # --- Funciones auxiliares ---

    def render_distribuciones(df_local, atributos):
        """Comparación de marcas por atributo."""
        st.subheader("Comparación de marcas por atributo")
        atributo = st.selectbox("Atributo", atributos, index=0)
        marcas = sorted(df_local["tipo_cafe"].dropna().unique().tolist())
        seleccion = st.multiselect("Marcas a comparar", marcas, default=marcas)
        if not seleccion:
            st.info("Selecciona al menos una marca.")
            return
        subset = df_local[df_local["tipo_cafe"].isin(seleccion)][["tipo_cafe", atributo]].dropna()
        chart = (
            alt.Chart(subset)
            .mark_boxplot(size=40)
            .encode(
                x=alt.X("tipo_cafe:N", title="Marca de café"),
                y=alt.Y(f"{atributo}:Q", title=f"Puntuación de {atributo}"),
                color="tipo_cafe:N"
            )
            .properties(width=600, height=400)
        )
        st.altair_chart(chart, use_container_width=True)

    def render_descriptivos(df_local, atributos):
        """Tabla descriptiva + promedios por categoría."""
        st.subheader("Estadísticos descriptivos por marca y atributo")

        col1, col2 = st.columns([2, 1])

        with col1:
            piezas = []
            for atr in atributos:
                g = df_local.groupby("tipo_cafe")[atr].agg(["count", "mean", "std", "median"])
                g["atributo"] = atr
                piezas.append(g.reset_index())
            desc = (
                pd.concat(piezas, ignore_index=True)
                [["atributo", "tipo_cafe", "count", "mean", "std", "median"]]
                .sort_values(["atributo", "tipo_cafe"])
            )
            st.dataframe(desc, use_container_width=True)

        with col2:
            st.markdown("**Promedio por categoría**")
            atributo_sel = st.selectbox("Atributo", atributos, index=0)
            st.bar_chart(df_local.groupby("tipo_cafe")[atributo_sel].mean())

    def render_boxplots_edad(df_local):
        """Boxplots de edad (general y por sexo)."""
        st.subheader("Distribución de edades")

        df_box = df_local.copy()
        if "edad_num" not in df_box.columns:
            if "grupo_edad" in df_box.columns:
                def _midpoint(r):
                    m = re.findall(r"\d+", str(r))
                    if len(m) >= 2:
                        a, b = map(int, m[:2])
                        return (a + b) / 2
                    elif len(m) == 1:
                        return float(m[0])
                    return None
                df_box["edad_num"] = df_box["grupo_edad"].map(_midpoint)
            else:
                st.info("No hay información de edad.")
                return

        df_box = df_box.dropna(subset=["edad_num"])
        if df_box.empty:
            st.info("Sin datos de edad para graficar.")
            return

        st.markdown("**General**")
        chart_general = (
            alt.Chart(df_box)
            .mark_boxplot(size=60)
            .encode(y=alt.Y("edad_num:Q", title="Edad"))
            .properties(width=500, height=250)
        )
        st.altair_chart(chart_general, use_container_width=True)

        if "sexo" in df_box.columns:
            st.markdown("**Por sexo**")
            chart_sexo = (
                alt.Chart(df_box)
                .mark_boxplot(size=40)
                .encode(
                    x=alt.X("sexo:N", title="Sexo"),
                    y=alt.Y("edad_num:Q", title="Edad"),
                    color="sexo:N"
                )
                .properties(width=600, height=350)
            )
            st.altair_chart(chart_sexo, use_container_width=True)
        else:
            st.info("No se encontró la columna 'sexo' para graficar por sexo.")

    # --- Render dinámico ---
    etiquetas_tabs = [n for n, _ in tabs_disponibles]
    tabs = st.tabs(etiquetas_tabs)
    for i, (nombre, fid) in enumerate(tabs_disponibles):
        with tabs[i]:
            if fid == "render_distribuciones":
                render_distribuciones(df, atributos_presentes)
            elif fid == "render_descriptivos":
                render_descriptivos(df, atributos_presentes)
            elif fid == "render_boxplots_edad":
                render_boxplots_edad(df)

# =============================
#  Pruebas
# =============================
elif pagina == " Pruebas":
    import altair as alt
    st.title(" Pruebas de hipótesis")
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
    # --- PRUEBAS DE SUPUESTOS BÁSICOS ---
    from scipy.stats import levene, shapiro

    st.markdown("###  Pruebas de supuestos")

    for atr in ATR:
        grupos = [g[atr].dropna() for _, g in df.groupby("tipo_cafe")]
        if all(len(g) > 2 for g in grupos):
            stat_lev, p_lev = levene(*grupos)
            stat_sh, p_sh = shapiro(df[atr].dropna())
            st.write(f"**{atr.capitalize()}** — Levene p = {p_lev:.3f}, Shapiro p = {p_sh:.3f}")
        else:
            st.info(f"No hay suficientes datos para {atr}.")
        
        
        
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

    if not resultados:
        st.info("No hay comparaciones posibles con la configuración actual.")
        st.stop()

    tabla = pd.DataFrame(resultados)

    # Holm por atributo (usando nombre legible)
    tablas = []
    for atr, sub in tabla.groupby("Atributo sensorial", as_index=False):
        sub = sub.copy()
        pvals = sub["p-valor"].values
        orden = np.argsort(pvals)
        m = len(pvals)
        ajust = np.empty_like(pvals, dtype=float)
        for rank, idx in enumerate(orden, start=1):
            ajust[idx] = min((m - rank + 1) * pvals[idx], 1.0)
        sub["p-valor ajustado (Holm)"] = ajust
        sub["Significativo (α = 0.05)"] = sub["p-valor ajustado (Holm)"] < 0.05
        tablas.append(sub)

    tabla = pd.concat(tablas, ignore_index=True).sort_values(
        ["Atributo sensorial", "p-valor ajustado (Holm)", "p-valor"]
    )

    # ===== Layout compacto y legible =====
    st.subheader("Resultados (resumen)")
    cols_entre = ["Atributo sensorial","Café A","Café B",
                "Participantes (A)","Participantes (B)",
                "Diferencia de medias (A−B)","IC 95 % inferior","IC 95 % superior",
                "Estadístico t","gl","p-valor","p-valor ajustado (Holm)","Significativo (α = 0.05)","Tamaño del efecto (Hedges g)"]
    cols_apar = ["Atributo sensorial","Café A","Café B",
                "Participantes",
                "Diferencia de medias (A−B)","IC 95 % inferior","IC 95 % superior",
                "Estadístico t","gl","p-valor","p-valor ajustado (Holm)","Significativo (α = 0.05)","Tamaño del efecto (Hedges g)"]

    st.dataframe(
        tabla[cols_entre] if diseño.startswith("Entre") else tabla[cols_apar],
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Interpretación")

    atr_sel = st.selectbox("Atributo", ATR, index=0, key="atr_interp")

    

    # 2) Solo comparaciones significativas (Holm < 0.05)
    sig = (
        tabla[(tabla["Atributo sensorial"] == atr_sel) & (tabla["p-valor ajustado (Holm)"] < 0.05)]
        .copy()
        .sort_values("p-valor ajustado (Holm)")
    )

    st.markdown("**✅ Diferencias significativas (Holm < 0.05)**")
    if sig.empty:
        st.info("No se detectaron diferencias significativas para este atributo.")
    else:
        for _, r in sig.iterrows():
            a, b = r["Café A"], r["Café B"]
            diff  = r["Diferencia de medias (A−B)"]
            ci_lo, ci_hi = r["IC 95 % inferior"], r["IC 95 % superior"]
            p_adj = r["p-valor ajustado (Holm)"]

            # nombres legibles
            na = a.replace("_", " ")
            nb = b.replace("_", " ")

            # definir sentido
            if diff > 0:
                comparacion = f"**{na} obtuvo en promedio {abs(diff):.2f} puntos más que {nb}**."
            elif diff < 0:
                comparacion = f"**{na} obtuvo en promedio {abs(diff):.2f} puntos menos que {nb}**."
            else:
                comparacion = f"**{na} y {nb} obtuvieron promedios iguales**."

            # mensaje explicativo
            st.markdown(
                f"{comparacion} "
                f"Esto significa que la diferencia observada entre ambos cafés es de **{abs(diff):.2f} puntos**.\n\n"
                f"El intervalo de confianza al 95 % va de **{ci_lo:.2f}** a **{ci_hi:.2f}**, "
                f"lo que indica que, si repitiésemos el experimento muchas veces, "
                f"la verdadera diferencia estaría probablemente dentro de ese rango.\n\n"
                f"El valor de **p(Holm) = {p_adj:.4f}**, "
                f"que corrige por las múltiples comparaciones, sugiere que esta diferencia "
                f"{'es **estadísticamente significativa** (p < 0.05)' if p_adj < 0.05 else 'no es significativa (p ≥ 0.05)'}."
            )
    
    st.markdown("---")
   
    # --- Comparaciones por sexo ---
        # --- Explicación de la sección Comparaciones por sexo ---
    st.markdown("###  Comparaciones por sexo")
    
    st.markdown("####  Cómo leer resultados a continuación:")
    st.markdown("""
    - Cada línea compara **hombres vs mujeres** para un **atributo** dentro de una **marca**.
    - **t** indica magnitud y dirección (signo): negativo → promedio H < M; positivo → H > M (según orden interno).
    - **p** es la evidencia estadística: si **p < 0.05**, la diferencia se considera **significativa**.
    - Si **p ≥ 0.05**, no hay evidencia suficiente de diferencia entre sexos para esa marca/atributo.
    - Recordar: escalas Likert son ordinales; tratarlas como intervalares es una aproximación habitual.
    """)
    st.markdown("#### Resultados:   ")
    
    for atr in ATR:
        for cafe in df["tipo_cafe"].dropna().unique():
            sub = df[df["tipo_cafe"] == cafe]
            gM = sub[sub["sexo"].str.upper() == "M"][atr].dropna()
            gF = sub[sub["sexo"].str.upper() == "F"][atr].dropna()
            if len(gM) > 2 and len(gF) > 2:
                tval, pval = stats.ttest_ind(gM, gF, equal_var=False)
                st.write(f"{atr.capitalize()} ({cafe}) → t = {tval:.2f}, p = {pval:.3f}")

    st.markdown("---")
   



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
