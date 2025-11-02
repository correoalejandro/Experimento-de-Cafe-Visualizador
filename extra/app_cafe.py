import streamlit as st
import pandas as pd
import numpy as np
import itertools
from scipy import stats
import altair as alt

st.set_page_config(page_title="Análisis sensorial de café", layout="wide")

# === Funciones base ===
def convertir_texto_a_likert(df):
    mapas = {
        "Olor":   {"malo": 1, "regular": 3, "bueno": 5, "excelente": 7},
        "Sabor":  {"malo": 1, "regular": 3, "bueno": 5, "excelente": 7},
        "Acidez": {"baja": 2, "media": 4, "alta": 6},
    }
    for col, mapa in mapas.items():
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().map(mapa)
    return df

def descriptivos_por_marca(df, atributos):
    partes = []
    for atr in atributos:
        g = df.groupby("Marca")[atr].agg(["count", "mean", "std", "median"])
        g["Atributo"] = atr
        partes.append(g.reset_index())
    out = pd.concat(partes, ignore_index=True)
    cols = ["Atributo", "Marca", "count", "mean", "std", "median"]
    return out[cols].sort_values(["Atributo", "Marca"])

def t_apareadas_todas_parejas(df, atributos):
    marcas = sorted(df["Marca"].unique())
    pares = list(itertools.combinations(marcas, 2))
    resultados = []

    for atr in atributos:
        # 🔹 Usa el nombre exacto de la columna en tu CSV
        pivot = df.pivot_table(index="ID_Participante", columns="Marca", values=atr)
        for a, b in pares:
            sub = pivot[[a, b]].dropna()
            if len(sub) < 2:
                continue
            t_stat, p_val = stats.ttest_rel(sub[a], sub[b], nan_policy="omit")
            mean_diff = np.nanmean(sub[a] - sub[b])
            resultados.append({
                "Atributo": atr,
                "Marca_A": a, "Marca_B": b,
                "N_pares": len(sub),
                "Diferencia_media": mean_diff,
                "t": t_stat, "p": p_val,
            })

    res = pd.DataFrame(resultados)
    if res.empty:
        return res

    # Corrección Holm
    frames = []
    for atr, sub in res.groupby("Atributo"):
        p = sub["p"].values
        orden = np.argsort(p)
        m = len(p)
        ajust = np.empty_like(p, dtype=float)
        for r, idx in enumerate(orden, start=1):
            ajust[idx] = min((m - r + 1) * p[idx], 1.0)
        sub["p_holm"] = ajust
        sub["signif_0_05"] = sub["p_holm"] < 0.05
        frames.append(sub)
    return pd.concat(frames).sort_values(["Atributo", "p_holm", "p"])

# === Interfaz Streamlit ===
st.title("☕ Análisis sensorial de cafés — Escala Likert")

st.sidebar.header("📂 Datos")
archivo = st.sidebar.file_uploader("Sube tu archivo CSV o XLSX", type=["csv", "xlsx"])

if archivo:
    if archivo.name.endswith(".xlsx"):
        df = pd.read_excel(archivo)
    else:
        df = pd.read_csv(archivo)

    df = convertir_texto_a_likert(df)
    atributos = ["Olor", "Sabor", "Acidez"]

    st.sidebar.success("Archivo cargado correctamente ✅")

    with st.expander("👁️ Vista previa de los datos", expanded=False):
        st.dataframe(df.head())

    # Descriptivos
    desc = descriptivos_por_marca(df, atributos)
    st.header("📊 Estadísticas descriptivas")
    atr_sel = st.multiselect("Selecciona atributos a visualizar", atributos, default=atributos)
    desc_filtrado = desc[desc["Atributo"].isin(atr_sel)]
    st.dataframe(desc_filtrado, use_container_width=True)

    # Gráfico comparativo
    st.subheader("📈 Comparación de medias por atributo")
    chart = (
        alt.Chart(desc_filtrado)
        .mark_bar()
        .encode(
            x=alt.X("Marca:N", title="Marca de café"),
            y=alt.Y("mean:Q", title="Media (escala Likert)"),
            color="Marca:N",
            column="Atributo:N"
        )
        .properties(height=250)
    )
    st.altair_chart(chart, use_container_width=True)

    # t-tests
    st.header("🧪 Pruebas t apareadas (Holm)")
    tt = t_apareadas_todas_parejas(df, atributos)
    atr_sel2 = st.selectbox("Filtrar por atributo", ["Todos"] + atributos)
    if atr_sel2 != "Todos":
        tt = tt[tt["Atributo"] == atr_sel2]

    st.dataframe(tt, use_container_width=True)

    # Mostrar solo significativos
    st.subheader("⭐ Comparaciones significativas (p<0.05 Holm)")
    signif = tt[tt["signif_0_05"]]
    st.dataframe(signif, use_container_width=True)

else:
    st.info("Sube un archivo para comenzar el análisis.")
