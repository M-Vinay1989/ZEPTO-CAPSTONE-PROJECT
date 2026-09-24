"""
Exploratory Data Analysis (EDA) & Visualization Module for Module 2: Customer Analytics.
Generates distribution plots, correlation heatmaps, feature relationship scatter plots,
and exports figure PNGs to analytics/output/figures/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"

def set_plot_style():
    """Sets consistent seaborn aesthetic style for charts."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

def plot_target_distribution(df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> str:
    """Plots histogram and KDE for annual_spending target variable."""
    set_plot_style()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["annual_spending"], kde=True, color="teal", ax=ax)
    ax.set_title("Distribution of Customer Annual Spending ($)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Annual Spending ($)")
    ax.set_ylabel("Customer Count")

    fig_path = output_dir / "target_distribution.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[EDA] Saved target distribution plot to '{fig_path}'.")
    return str(fig_path)

def plot_numerical_distributions(df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> str:
    """Plots histograms for numerical features."""
    set_plot_style()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    num_cols = ["age", "income", "membership_years", "number_of_purchases", "average_order_value", "discount_usage"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, color="steelblue", ax=axes[idx])
        axes[idx].set_title(f"Distribution of {col.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel(col.replace('_', ' ').title())

    fig_path = output_dir / "numerical_distributions.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[EDA] Saved numerical distributions plot to '{fig_path}'.")
    return str(fig_path)

def plot_categorical_distribution(df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> str:
    """Plots bar chart of preferred customer categories."""
    set_plot_style()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df,
        x="preferred_category",
        order=df["preferred_category"].value_counts().index,
        hue="preferred_category",
        legend=False,
        palette="viridis",
        ax=ax
    )
    ax.set_title("Customer Count by Preferred Product Category", fontsize=13, fontweight="bold")
    ax.set_xlabel("Preferred Category")
    ax.set_ylabel("Count")

    fig_path = output_dir / "categorical_distribution.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[EDA] Saved categorical distribution plot to '{fig_path}'.")
    return str(fig_path)

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> str:
    """Plots correlation heatmap for numerical variables."""
    set_plot_style()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    num_df = df.select_dtypes(include=["float64", "int64", "float32", "int32"])
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = num_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")

    fig_path = output_dir / "correlation_heatmap.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[EDA] Saved correlation heatmap plot to '{fig_path}'.")
    return str(fig_path)

def plot_spending_relationships(df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> str:
    """Plots scatter plots of top numerical features vs annual_spending."""
    set_plot_style()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    key_features = ["number_of_purchases", "average_order_value", "income", "membership_years"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, col in enumerate(key_features):
        sns.scatterplot(data=df, x=col, y="annual_spending", hue="preferred_category", alpha=0.7, ax=axes[idx])
        sns.regplot(data=df, x=col, y="annual_spending", scatter=False, color="darkred", ax=axes[idx])
        axes[idx].set_title(f"{col.replace('_', ' ').title()} vs Annual Spending", fontsize=11, fontweight="bold")

    fig_path = output_dir / "spending_relationships.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[EDA] Saved spending relationships scatter plots to '{fig_path}'.")
    return str(fig_path)

def run_full_eda(df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> dict:
    """Executes full EDA suite and returns generated plot filepaths."""
    paths = {
        "target_dist": plot_target_distribution(df, output_dir),
        "num_dist": plot_numerical_distributions(df, output_dir),
        "cat_dist": plot_categorical_distribution(df, output_dir),
        "heatmap": plot_correlation_heatmap(df, output_dir),
        "relationships": plot_spending_relationships(df, output_dir)
    }
    return paths

if __name__ == "__main__":
    from data_loader import load_dataset
    from preprocessing import clean_dataset
    df = clean_dataset(load_dataset())
    run_full_eda(df)
