#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analisis_cafe.py — EDA y pruebas t apareadas para 4 cafés en un test sensorial.
- Entrada: CSV en formato largo con columnas:
  participant_id, age_group, sex, coffee_type, presentation_order,
  odor_rating, flavor_rating, acidity_rating
- Salidas: tablas en consola con descriptivos y t-tests pareados por atributo.
- Uso (cero typing): `python analisis_cafe.py` (descubre el CSV por defecto).
- Reglas de estilo: sin abreviaciones, funciones pequeñas, logs con [DEBUG].

Notas:
- Las escalas son ordinales (1-7). Aquí se tratan como intervalares para t-tests.
  Alternativas no paramétricas: Friedman (global), Wilcoxon pareado (por pares).
"""

import os
import sys
import itertools
import textwrap
from typing import List, Tuple

import pandas as pd
import numpy as np
from scipy import stats


DEFAULT_CSV_CANDIDATES = [
    "cafe_sensory_sample.csv",
    "./cafe_sensory_sample.csv",
]




def find_default_csv_file() -> str:
    print("[DEBUG] Buscando archivo CSV por defecto...")
    for candidate in DEFAULT_CSV_CANDIDATES:
        if os.path.exists(candidate):
            print(f"[DEBUG] Encontrado: {candidate}")
            return candidate
    # Buscar en carpeta actual por archivos que contengan 'cafe' y 'sensory'
    for name in os.listdir("."):
        if name.lower().endswith(".csv") and "cafe" in name.lower():
            print(f"[DEBUG] Encontrado por heurística: {name}")
            return name
    raise FileNotFoundError("No se encontró un CSV por defecto. Pase la ruta como argumento opcional.")


def load_dataset(csv_path: str) -> pd.DataFrame:
    print(f"[DEBUG] Cargando datos desde: {csv_path}")
    df = pd.read_csv(csv_path)
    expected_cols = {
        "participant_id", "age_group", "sex", "coffee_type", "presentation_order",
        "odor_rating", "flavor_rating", "acidity_rating"
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    return df



def convertir_texto_a_likert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte valores cualitativos (malo, regular, bueno, excelente, baja, media, alta)
    a una escala numérica tipo Likert (1–7). Si ya son numéricos, no los modifica.
    """
    print("[DEBUG] Convirtiendo variables ordinales a escala Likert...")

    mapas = {
        "odor_rating": {"malo": 1, "regular": 3, "bueno": 5, "excelente": 7},
        "flavor_rating": {"malo": 1, "regular": 3, "bueno": 5, "excelente": 7},
        "acidity_rating": {"baja": 2, "media": 4, "alta": 6},
    }

    for col, mapa in mapas.items():
        if df[col].dtype == object:
            df[col] = df[col].str.lower().map(mapa)
    return df


def show_main_menu_and_get_choice() -> int:
    print("\n=== Menú de análisis (elige una opción; Enter ejecuta 'todo') ===")
    print("  1) Estadística descriptiva (por café y atributo)")
    print("  2) Pruebas t apareadas para todas las parejas (por atributo)")
    print("  3) Descriptivos + t-tests (completo) [por defecto]")
    print("  4) Salir")
    choice = input("[DEBUG] Opción (Enter = 3): ").strip()
    if choice == "":
        return 3
    if choice.isdigit():
        return int(choice)
    return 3


def compute_descriptive_statistics(df: pd.DataFrame, attributes: List[str]) -> pd.DataFrame:
    print("[DEBUG] Calculando descriptivos por café y atributo...")
    pieces = []
    for attribute in attributes:
        g = df.groupby("coffee_type")[attribute].agg(["count", "mean", "std", "median"])
        g["attribute"] = attribute
        pieces.append(g.reset_index())
    out = pd.concat(pieces, ignore_index=True)
    cols = ["attribute", "coffee_type", "count", "mean", "std", "median"]
    out = out[cols]
    return out.sort_values(["attribute", "coffee_type"]).reset_index(drop=True)


def run_paired_t_tests(df: pd.DataFrame, attributes: List[str]) -> pd.DataFrame:
    print("[DEBUG] Ejecutando pruebas t apareadas por atributo y pareja de cafés...")
    coffee_levels = sorted(df["coffee_type"].unique())
    pairs = list(itertools.combinations(coffee_levels, 2))

    results = []
    for attribute in attributes:
        print(f"[DEBUG] Atributo: {attribute}")
        for cafe_a, cafe_b in pairs:
            pivot = df.pivot_table(index="participant_id",
                                   columns="coffee_type",
                                   values=attribute)
            # Alinear participantes con ambos cafés evaluados
            sub = pivot[[cafe_a, cafe_b]].dropna()
            values_a = sub[cafe_a].values
            values_b = sub[cafe_b].values
            if len(values_a) < 2:
                print(f"[DEBUG] Pareja {cafe_a} vs {cafe_b}: datos insuficientes.")
                continue
            t_statistic, p_value = stats.ttest_rel(values_a, values_b, nan_policy="omit")

            mean_diff = float(np.nanmean(values_a - values_b))
            results.append({
                "attribute": attribute,
                "coffee_a": cafe_a,
                "coffee_b": cafe_b,
                "number_of_pairs": int(len(values_a)),
                "mean_difference_a_minus_b": mean_diff,
                "t_statistic": float(t_statistic),
                "p_value": float(p_value),
            })

    res = pd.DataFrame(results)
    if res.empty:
        return res

    # Corrección por comparaciones múltiples (Holm)
    print("[DEBUG] Aplicando corrección de Holm por atributo...")
    corrected_frames = []
    for attribute, subdf in res.groupby("attribute", as_index=False):
        pvals = subdf["p_value"].values
        order = np.argsort(pvals)
        m = len(pvals)
        adjusted = np.empty_like(pvals, dtype=float)
        for rank, idx in enumerate(order, start=1):
            adjusted[idx] = min((m - rank + 1) * pvals[idx], 1.0)
        subdf = subdf.copy()
        subdf["p_value_holm"] = adjusted
        corrected_frames.append(subdf)
    res = pd.concat(corrected_frames, ignore_index=True)

    # Señal rápida de significancia
    res["significant_at_0_05_holm"] = res["p_value_holm"] < 0.05
    return res.sort_values(["attribute", "p_value_holm", "p_value"]).reset_index(drop=True)


def print_table(df: pd.DataFrame, title: str, max_rows: int = 200):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if df.empty:
        print("(tabla vacía)")
    else:
        with pd.option_context("display.max_rows", max_rows, "display.max_columns", None):
            print(df.to_string(index=False))


def main():
    try:
        csv_path = sys.argv[1] if len(sys.argv) > 1 else find_default_csv_file()
    except Exception as exc:
        print(f"[DEBUG] No se pudo inferir archivo CSV: {exc}")
        print("[DEBUG] Uso: python analisis_cafe.py [ruta_csv]")
        return

    try:
        data = load_dataset(csv_path)

        # Convertir textos a escala Likert si es necesario
        # data = convertir_texto_a_likert(data)
    except Exception as exc:
        print(f"[DEBUG] Error al cargar datos: {exc}")
        return

    attributes = ["odor_rating", "flavor_rating", "acidity_rating"]

    choice = show_main_menu_and_get_choice()

    if choice == 1:
        descriptives = compute_descriptive_statistics(data, attributes)
        print_table(descriptives, "Descriptivos por café y atributo")
    elif choice == 2:
        tests = run_paired_t_tests(data, attributes)
        print_table(tests, "t-tests apareados por atributo y pareja de cafés")
    elif choice == 3:
        descriptives = compute_descriptive_statistics(data, attributes)
        print_table(descriptives, "Descriptivos por café y atributo")
        tests = run_paired_t_tests(data, attributes)
        print_table(tests, "t-tests apareados por atributo y pareja de cafés (Holm)")
    else:
        print("[DEBUG] Saliendo...")


if __name__ == "__main__":
    main()
