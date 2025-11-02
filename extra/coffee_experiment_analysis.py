
import os
import sys
import itertools
import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def print_debug(message: str) -> None:
    print(f"[DEBUG] {message}")

def read_dataset_or_generate_example(default_csv_path: str) -> pd.DataFrame:
    if os.path.exists(default_csv_path):
        print_debug(f"Loading dataset from {default_csv_path}")
        df = pd.read_csv(default_csv_path)
    else:
        print_debug("Default dataset not found. Generating a small demo dataset with 4 groups.")
        rng = np.random.default_rng(42)
        group_names = ["Treatment_A", "Treatment_B", "Treatment_C", "Treatment_D"]
        means = [20.0, 22.0, 23.0, 21.0]
        stds = [2.0, 2.2, 1.8, 2.5]
        sizes = [25, 25, 25, 25]
        rows = []
        for g, mu, sd, n in zip(group_names, means, stds, sizes):
            values = rng.normal(mu, sd, n)
            for v in values:
                rows.append({"group": g, "response": float(v)})
        df = pd.DataFrame(rows)
    # Basic validation
    required_cols = {"group", "response"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required_cols}. Found: {list(df.columns)}")
    # Ensure proper types
    df["group"] = df["group"].astype(str)
    df["response"] = pd.to_numeric(df["response"], errors="coerce")
    df = df.dropna(subset=["response", "group"])
    return df

def summarize_groups(data: pd.DataFrame) -> pd.DataFrame:
    summaries = (
        data.groupby("group")["response"]
        .agg(n="count", mean="mean", std="std", median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .reset_index()
    )
    return summaries

def test_normality_by_group(data: pd.DataFrame) -> pd.DataFrame:
    # Shapiro-Wilk for small to medium samples; will cap at 5000
    rows = []
    for g, sub in data.groupby("group"):
        x = sub["response"].values
        x = x[~np.isnan(x)]
        x = x[:5000]
        if len(x) < 3:
            W, p = np.nan, np.nan
        else:
            W, p = stats.shapiro(x)
        rows.append({"group": g, "shapiro_W": W, "shapiro_pvalue": p, "n": len(x)})
    return pd.DataFrame(rows)

def levene_test_equal_variances(data: pd.DataFrame) -> dict:
    arrays = [sub["response"].values for _, sub in data.groupby("group")]
    stat, p = stats.levene(*arrays, center="median")
    return {"levene_statistic": float(stat), "levene_pvalue": float(p)}

def compute_hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    n1 = len(group1)
    n2 = len(group2)
    s1 = np.var(group1, ddof=1)
    s2 = np.var(group2, ddof=1)
    # Pooled SD for effect size only (not for test statistic because we use Welch for test)
    sp = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else np.nan
    d = (np.mean(group1) - np.mean(group2)) / sp if sp and sp > 0 else np.nan
    # Small sample correction (Hedges' g)
    J = 1 - (3 / (4*(n1 + n2) - 9)) if (n1 + n2) > 2 else 1.0
    return d * J if d is not None else np.nan

def welch_t_ci(group1: np.ndarray, group2: np.ndarray, alpha: float = 0.05):
    # Welch t confidence interval for difference in means (group1 - group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    se = math.sqrt(s1/n1 + s2/n2)
    # Welch-Satterthwaite degrees of freedom
    df_num = (s1/n1 + s2/n2)**2
    df_den = (s1**2)/(n1**2*(n1-1)) + (s2**2)/(n2**2*(n2-1))
    df = df_num / df_den if df_den > 0 else np.nan
    tcrit = stats.t.ppf(1 - alpha/2, df) if df and df > 0 else np.nan
    diff = m1 - m2
    ci_low = diff - tcrit * se if tcrit and se else np.nan
    ci_high = diff + tcrit * se if tcrit and se else np.nan
    return diff, se, df, ci_low, ci_high

def holm_adjustment(pvalues: list) -> list:
    # Holm-Bonferroni step-down adjustment
    m = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [None]*m
    prev = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = (m - rank + 1) * p
        adj = max(adj, prev)  # ensure monotonicity
        adjusted[idx] = min(adj, 1.0)
        prev = adjusted[idx]
    return adjusted

def run_exploratory_analysis(data: pd.DataFrame, output_dir: str) -> None:
    print_debug("Running exploratory analysis...")
    summaries = summarize_groups(data)
    print_debug(f"Group summaries:\n{summaries.to_string(index=False)}")

    normality = test_normality_by_group(data)
    print_debug(f"Normality (Shapiro-Wilk):\n{normality.to_string(index=False)}")

    levene = levene_test_equal_variances(data)
    print_debug(f"Levene's test (center=median): {levene}")

    # Save tables
    summaries.to_csv(os.path.join(output_dir, "summaries.csv"), index=False)
    normality.to_csv(os.path.join(output_dir, "normality_shapiro.csv"), index=False)
    pd.DataFrame([levene]).to_csv(os.path.join(output_dir, "levene_test.csv"), index=False)

    # Boxplot
    plt.figure()
    data.boxplot(column="response", by="group")
    plt.suptitle("")
    plt.title("Response by group (boxplot)")
    plt.xlabel("Group")
    plt.ylabel("Response")
    plot_path = os.path.join(output_dir, "boxplot_by_group.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print_debug(f"Saved plot: {plot_path}")

def run_pairwise_welch_tests(data: pd.DataFrame, output_dir: str, alpha: float = 0.05) -> pd.DataFrame:
    print_debug("Running pairwise Welch t-tests (all pairs) with Holm correction...")
    groups = sorted(data["group"].unique())
    pairs = list(itertools.combinations(groups, 2))
    results = []
    raw_pvalues = []

    for g1, g2 in pairs:
        arr1 = data.loc[data["group"] == g1, "response"].values
        arr2 = data.loc[data["group"] == g2, "response"].values
        t_stat, p_val = stats.ttest_ind(arr1, arr2, equal_var=False)  # Welch
        diff, se, df, ci_low, ci_high = welch_t_ci(arr1, arr2, alpha=alpha)
        hedges_g = compute_hedges_g(arr1, arr2)
        results.append({
            "group_1": g1,
            "group_2": g2,
            "mean_1": float(np.mean(arr1)),
            "mean_2": float(np.mean(arr2)),
            "difference_mean_1_minus_2": float(diff),
            "standard_error": float(se),
            "welch_df": float(df),
            "t_statistic": float(t_stat),
            "pvalue_raw": float(p_val),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "hedges_g": float(hedges_g),
        })
        raw_pvalues.append(p_val)

    adjusted = holm_adjustment(raw_pvalues)
    for row, p_adj in zip(results, adjusted):
        row["pvalue_holm"] = float(p_adj)
        row["reject_at_alpha"] = bool(p_adj < alpha)

    df_res = pd.DataFrame(results).sort_values("pvalue_holm").reset_index(drop=True)
    out_csv = os.path.join(output_dir, "pairwise_welch_tests.csv")
    df_res.to_csv(out_csv, index=False)
    print_debug(f"Saved pairwise test results: {out_csv}")
    return df_res

def simple_menu() -> int:
    print("\n=== Coffee Experiment Analysis ===")
    print("1) Run full analysis now (default)")
    print("2) Only exploratory analysis")
    print("3) Only pairwise Welch t-tests")
    print("4) Exit")
    choice = input("Choose an option [1]: ").strip()
    if choice == "":
        return 1
    try:
        return int(choice)
    except ValueError:
        return 1

def main():
    default_csv_path = os.environ.get("COFFEE_CSV_PATH", "coffee_experiment.csv")
    output_dir = os.environ.get("COFFEE_OUTPUT_DIR", "coffee_outputs")
    os.makedirs(output_dir, exist_ok=True)

    print_debug(f"Using CSV path: {default_csv_path}")
    print_debug(f"Using output dir: {output_dir}")

    data = read_dataset_or_generate_example(default_csv_path)
    print_debug(f"Data shape: {data.shape}, groups: {sorted(data['group'].unique())}")

    choice = simple_menu()

    if choice == 4:
        print_debug("Exiting without analysis.")
        sys.exit(0)

    if choice in (1, 2):
        run_exploratory_analysis(data, output_dir)

    if choice in (1, 3):
        results = run_pairwise_welch_tests(data, output_dir, alpha=0.05)
        print_debug("Pairwise Welch t-tests (Holm-adjusted) summary:")
        print(results.to_string(index=False))

    print_debug("Done.")

if __name__ == "__main__":
    main()
