#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analisis_cafe_es.py — Análisis sensorial de café (CSV en español)
- Entrada: CSV (por defecto: MuestreoCafe.csv) con encabezados en español.
- Columnas esperadas (en formato largo): 
  participante_id, grupo_edad, sexo, tipo_cafe, (opcional) orden_presentacion, olor, sabor, acidez
- Funciones: descriptivos por café/atributo y pruebas t apareadas entre cafés (odor/sabor/acidez).
- Conversión automática de texto ordinal → escala Likert (1–7).
- Uso: python analisis_cafe_es.py [ruta_csv]
"""

import sys
import re
import itertools
import pandas as pd
import numpy as np
from scipy import stats

# ===== 1) Configuración mínima =====
ARCHIVO_POR_DEFECTO = "MuestreoCafe.csv"

# Si tus encabezados difieren, ajusta aquí (clave = nombre en tu CSV, valor = canónico)
MAPEO_COLUMNAS = {
    "ID_Participante": "participante_id",
    "Edad": "grupo_edad",
    "Sexo": "sexo",
    "Marca": "tipo_cafe",
    # "Orden": "orden_presentacion",  # si más adelante tienes una columna de orden
    "Olor": "olor",
    "Sabor": "sabor",
    "Acidez": "acidez",
}

# Patrones comunes (se usan solo si faltan columnas tras MAPEO_COLUMNAS)
ALIAS_PATTERNS = {
    "participante_id": [r"(?i)^id$", r"(?i)participante", r"(?i)id_?participante"],
    "grupo_edad":      [r"(?i)edad", r"(?i)rango_?edad"],
    "sexo":            [r"(?i)^sexo$", r"(?i)g[eé]nero"],
    "tipo_cafe":       [r"(?i)tipo.*caf[eé]", r"(?i)^caf[eé]$", r"(?i)caf[eé]_?[abcd]?$"],
    "orden_presentacion": [r"(?i)orden", r"(?i)orden_?presentaci[oó]n", r"(?i)posici[oó]n"],
    "olor":            [r"(?i)^olor$"],
    "sabor":           [r"(?i)^sabor$"],
    "acidez":          [r"(?i)^acidez$"],
}

ATRIBUTOS = ["olor", "sabor", "acidez"]


# ===== 2) Utilidades =====
def _leer_csv_flexible(ruta: str) -> pd.DataFrame:
    # Detecta separador automáticamente y maneja utf-8/utf-8-sig
    try:
        return pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8")


def _aplicar_mapeo(df: pd.DataFrame) -> pd.DataFrame:
    if MAPEO_COLUMNAS:
        ren = {k: v for k, v in MAPEO_COLUMNAS.items() if k in df.columns}
        df = df.rename(columns=ren)
    return df


def _aplicar_alias(df: pd.DataFrame) -> pd.DataFrame:
    faltan = {"participante_id", "grupo_edad", "sexo", "tipo_cafe", "olor", "sabor", "acidez"} - set(df.columns)
    if not faltan:
        return df
    ren = {}
    for canonico, patrones in ALIAS_PATTERNS.items():
        if canonico in df.columns:
            continue
        for col in df.columns:
            for pat in patrones:
                if re.search(pat, str(col)):
                    ren[col] = canonico
                    break
    if ren:
        df = df.rename(columns=ren)
    return df


def convertir_texto_a_likert(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte texto ordinal a escala 1–7. Si ya son números, no los cambia."""
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
        raise ValueError(f"Faltan columnas requeridas: {faltan}\nEncabezados disponibles: {list(df.columns)}")


# ===== 3) EDA y pruebas =====
def descriptivos_por_cafe(df: pd.DataFrame) -> pd.DataFrame:
    piezas = []
    for atr in ATRIBUTOS:
        g = df.groupby("tipo_cafe")[atr].agg(["count", "mean", "std", "median"])
        g["atributo"] = atr
        piezas.append(g.reset_index())
    out = pd.concat(piezas, ignore_index=True)
    out = out[["atributo", "tipo_cafe", "count", "mean", "std", "median"]]
    return out.sort_values(["atributo", "tipo_cafe"]).reset_index(drop=True)


def ttests_independientes(df: pd.DataFrame) -> pd.DataFrame:
    cafes = sorted(df["tipo_cafe"].dropna().unique().tolist())
    pares = list(itertools.combinations(cafes, 2))
    resultados = []

    for atr in ATRIBUTOS:
        for a, b in pares:
            datos_a = df.loc[df["tipo_cafe"] == a, atr].dropna()
            datos_b = df.loc[df["tipo_cafe"] == b, atr].dropna()
            if len(datos_a) < 2 or len(datos_b) < 2:
                continue
            # t de Welch (no asume varianzas iguales)
            tval, pval = stats.ttest_ind(datos_a, datos_b, equal_var=False, nan_policy="omit")
            diff = float(np.nanmean(datos_a) - np.nanmean(datos_b))
            resultados.append({
                "atributo": atr,
                "cafe_a": a,
                "cafe_b": b,
                "n_a": int(len(datos_a)),
                "n_b": int(len(datos_b)),
                "diferencia_media_a_menos_b": diff,
                "t": float(tval),
                "p": float(pval),
            })

    res = pd.DataFrame(resultados)
    if res.empty:
        return res

    # Corrección Holm
    corregidos = []
    for atr, subdf in res.groupby("atributo", as_index=False):
        p = subdf["p"].values
        orden = np.argsort(p)
        m = len(p)
        ajust = np.empty_like(p, dtype=float)
        for rank, idx in enumerate(orden, start=1):
            ajust[idx] = min((m - rank + 1) * p[idx], 1.0)
        subdf = subdf.copy()
        subdf["p_holm"] = ajust
        subdf["significativo_0_05_holm"] = subdf["p_holm"] < 0.05
        corregidos.append(subdf)
    return pd.concat(corregidos, ignore_index=True).sort_values(["atributo", "p_holm", "p"]).reset_index(drop=True)


# ===== 4) Main =====
def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else ARCHIVO_POR_DEFECTO
    print(f"[DEBUG] Archivo: {ruta}")
    try:
        df = _leer_csv_flexible(ruta)
        df = _aplicar_mapeo(df)
        df = _aplicar_alias(df)
        # conversión ordinal → Likert
        df = convertir_texto_a_likert(df)
        validar_columnas(df)
    except Exception as e:
        print(f"[DEBUG] Error al cargar/preparar datos: {e}")
        return

    print("[DEBUG] Calculando descriptivos...")
    desc = descriptivos_por_cafe(df)
    print(desc.to_string(index=False))

    print("\n[DEBUG] Ejecutando t-tests independientes...")
    ttests = ttests_independientes(df)
    if ttests.empty:
        print("(No hay suficientes pares completos para alguna comparación)")
    else:
        print(ttests.to_string(index=False))


if __name__ == "__main__":
    main()
