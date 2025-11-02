# coffee_app_streamlit.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from extra.coffee_experiment_analysis import (
    read_dataset_or_generate_example,
    summarize_groups,
    test_normality_by_group,
    levene_test_equal_variances,
    run_pairwise_welch_tests
)

st.set_page_config(page_title="Coffee Experiment Analysis", layout="wide")

st.title("☕ Coffee Experiment Analysis Dashboard")

# --- Load data
st.sidebar.header("Data options")
uploaded = st.sidebar.file_uploader("Upload your CSV (columns: group, response)", type=["csv"])
if uploaded:
    data = pd.read_csv(uploaded)
else:
    data = read_dataset_or_generate_example("coffee_experiment.csv")

st.write(f"Dataset loaded: {data.shape[0]} observations across {data['group'].nunique()} groups")

# --- Exploratory analysis
st.header("1️⃣ Exploratory Analysis")

summaries = summarize_groups(data)
st.subheader("Group summaries")
st.dataframe(summaries)

# --- Boxplot
st.subheader("Response by group")
fig, ax = plt.subplots()
sns.boxplot(data=data, x="group", y="response", ax=ax)
ax.set_title("Response distribution by coffee treatment")
st.pyplot(fig)

# --- Normality + Levene test
st.subheader("Normality (Shapiro-Wilk) by group")
normality = test_normality_by_group(data)
st.dataframe(normality)

st.subheader("Levene’s Test for Equal Variances")
levene = levene_test_equal_variances(data)
st.write(levene)

# --- Pairwise t-tests
st.header("2️⃣ Pairwise Welch t-tests (all group comparisons)")
if st.button("Run pairwise tests"):
    results = run_pairwise_welch_tests(data, output_dir="coffee_outputs", alpha=0.05)
    st.success("Pairwise Welch t-tests computed ✅")

    st.subheader("Results table (Holm-corrected p-values)")
    st.dataframe(results)

    st.subheader("Significant comparisons (p < 0.05)")
    sig = results[results["reject_at_alpha"]]
    if not sig.empty:
        st.dataframe(sig)
    else:
        st.info("No significant differences found at α = 0.05")

    # --- Plot mean differences
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.barplot(
        data=results,
        x="group_1",
        y="difference_mean_1_minus_2",
        hue="group_2",
        ax=ax2
    )
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_title("Mean differences (group1 - group2)")
    st.pyplot(fig2)
else:
    st.info("Press the button above to run t-tests.")

st.markdown("---")
st.caption("Developed by Rafael – Coffee Experiment Analysis 🧠☕")
