"""
eda.py
Generates exploratory data analysis plots and saves them to reports/eda_plots/.
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "eda_plots")


def _ensure_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_class_balance(df: pd.DataFrame):
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["diagnosis"].map({0: "Malignant", 1: "Benign"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                palette=["#e11d48", "#22c55e"], ax=ax, legend=False)
    ax.set_title("Class Distribution: Benign vs Malignant")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "class_balance.png"), dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame):
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(16, 14))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True,
                cbar_kws={"shrink": 0.7})
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame, top_n: int = 6):
    """Plots distributions for the top N 'mean' features, split by class."""
    _ensure_dir()
    mean_features = [c for c in df.columns if c.startswith("mean")][:top_n]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, feat in enumerate(mean_features):
        sns.kdeplot(data=df, x=feat, hue=df["diagnosis"].map({0: "Malignant", 1: "Benign"}),
                    fill=True, ax=axes[i], palette=["#e11d48", "#22c55e"], alpha=0.4)
        axes[i].set_title(feat)
    fig.suptitle("Feature Distributions by Diagnosis (Mean Features)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "feature_distributions.png"), dpi=150)
    plt.close(fig)


def plot_boxplots(df: pd.DataFrame, top_n: int = 6):
    """Boxplots to visualize outliers for top mean features."""
    _ensure_dir()
    mean_features = [c for c in df.columns if c.startswith("mean")][:top_n]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, feat in enumerate(mean_features):
        sns.boxplot(data=df, x="diagnosis", y=feat, hue="diagnosis", ax=axes[i],
                    palette=["#e11d48", "#22c55e"], legend=False)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["Malignant", "Benign"])
        axes[i].set_title(feat)
    fig.suptitle("Outlier Detection (Boxplots, Mean Features)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "boxplots.png"), dpi=150)
    plt.close(fig)


def plot_pairwise_top_features(df: pd.DataFrame):
    """Pairplot of the 4 most diagnostically relevant features."""
    _ensure_dir()
    key_features = ["mean radius", "mean concave points", "mean texture", "mean smoothness"]
    plot_df = df[key_features + ["diagnosis"]].copy()
    plot_df["diagnosis"] = plot_df["diagnosis"].map({0: "Malignant", 1: "Benign"})

    g = sns.pairplot(plot_df, hue="diagnosis", palette=["#e11d48", "#22c55e"], corner=True)
    g.fig.suptitle("Pairwise Relationships: Key Features", y=1.02)
    g.savefig(os.path.join(PLOTS_DIR, "pairplot.png"), dpi=150)
    plt.close(g.fig)


def run_full_eda(df: pd.DataFrame):
    """Runs all EDA plot functions and reports summary stats."""
    _ensure_dir()
    print("Generating EDA plots...")
    plot_class_balance(df)
    print("  - class_balance.png")
    plot_correlation_heatmap(df)
    print("  - correlation_heatmap.png")
    plot_feature_distributions(df)
    print("  - feature_distributions.png")
    plot_boxplots(df)
    print("  - boxplots.png")
    plot_pairwise_top_features(df)
    print("  - pairplot.png")
    print(f"\nAll plots saved to {os.path.abspath(PLOTS_DIR)}")

    summary = {
        "shape": df.shape,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_balance": df["diagnosis"].value_counts().to_dict(),
    }
    return summary


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import clean_data

    df = load_data()
    df = clean_data(df)
    summary = run_full_eda(df)
    print(f"\nEDA Summary: {summary}")