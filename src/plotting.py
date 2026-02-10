"""
Visualisation functions for the LMD-2022 exploratory analysis.

Every function takes its data as arguments so charts can be reproduced
from saved CSVs without rerunning the full pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot_style():
    """Set publication-quality matplotlib/seaborn defaults."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10


# ============================================================================
# EARLY EDA PLOTS (Cells 6 & 8)
# ============================================================================

def plot_class_distribution(lmd2022_df: pd.DataFrame, save_path: str | None = None):
    """
    Pie chart of malicious vs normal samples.

    Parameters
    ----------
    lmd2022_df : pd.DataFrame
        Must contain an 'isMalicious' column.
    save_path : str or None
        If provided, save the figure to this path.
    """
    counts = lmd2022_df["isMalicious"].value_counts().sort_index()
    fig, ax = plt.subplots()
    counts.plot(
        kind="pie",
        colors=["skyblue", "salmon"],
        autopct="%1.0f%%",
        startangle=90,
        ax=ax,
        labels=["", ""],
    )
    ax.set_title("Distribution of Malicious vs Normal Samples in LMD-2022 Dataset")
    ax.legend(["Normal", "Malicious"], loc="lower left", bbox_to_anchor=(1, 0.5))
    ax.set_ylabel("")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")
    plt.show()


def plot_event_counts(lmd2022_df: pd.DataFrame, event_mapping: dict,
                      save_path: str | None = None):
    """
    Horizontal bar chart of EventID counts split by Normal vs Malicious.

    Parameters
    ----------
    lmd2022_df : pd.DataFrame
        Must contain 'EventID' and 'isMalicious' columns.
    event_mapping : dict
        Mapping of EventID (int) → human-readable name.
    save_path : str or None
        If provided, save the figure to this path.
    """
    events_info = lmd2022_df[["EventID", "isMalicious"]].copy()
    events_info["EventName"] = events_info["EventID"].map(event_mapping)

    counts = events_info.groupby(["EventID", "isMalicious"]).size().unstack(fill_value=0)
    counts.columns = ["Normal", "Malicious"]
    counts["Total"] = counts.sum(axis=1)
    counts = counts.sort_values("Total").drop(columns="Total")

    new_index = []
    for eid in counts.index:
        event_name = event_mapping.get(eid, "")
        new_index.append(f"{eid} - {event_name}")
    counts.index = new_index

    counts.plot.barh(figsize=(12, 8), color=["skyblue", "salmon"])
    plt.xscale("log")
    plt.xlabel("Count (log scale)")
    plt.ylabel("EventID - Name")
    plt.title("EventID counts split by Normal vs Malicious")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")
    plt.show()


# ============================================================================
# FEATURE SELECTION PLOTS (Section 10 / Cell 19 + MI plot from Cell 14)
# ============================================================================

def plot_mi_encoded(mi_df: pd.DataFrame, top_n: int = 30,
                    title_suffix: str = "(Train-only, Group Split)",
                    save_path: str | None = None):
    """
    Horizontal bar chart of top encoded features by Mutual Information.

    Parameters
    ----------
    mi_df : pd.DataFrame
        Must contain 'Feature' and 'MI_Score' columns, sorted descending.
    top_n : int
        Number of top features to show.
    title_suffix : str
        Extra text appended to the plot title.
    save_path : str or None
        If provided, save the figure to this path.
    """
    top_n = min(top_n, len(mi_df))
    plt.figure(figsize=(12, 7))
    plt.barh(range(top_n), mi_df.head(top_n)["MI_Score"].values[::-1])
    plt.yticks(range(top_n), mi_df.head(top_n)["Feature"].values[::-1], fontsize=7)
    plt.xlabel("Mutual Information Score")
    plt.title(f"Top {top_n} Encoded Features by Mutual Information\n{title_suffix}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")
    plt.show()


def plot_original_feature_importance(summary: pd.DataFrame, top_n: int = 25,
                                     save_path: str | None = None):
    """
    Side-by-side RF importance and MI scores aggregated to original features.

    Parameters
    ----------
    summary : pd.DataFrame
        Must contain 'Original_Feature', 'RF_Importance_Sum', 'MI_Score_Sum'.
    top_n : int
        Number of top features to show per subplot.
    save_path : str or None
        If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    top_rf = summary.head(top_n)
    axes[0].barh(range(len(top_rf)), top_rf["RF_Importance_Sum"].values[::-1], color="#2196F3")
    axes[0].set_yticks(range(len(top_rf)))
    axes[0].set_yticklabels(top_rf["Original_Feature"].values[::-1], fontsize=9)
    axes[0].set_xlabel("Aggregated RF Importance (Sum)")
    axes[0].set_title("RF Feature Importance\n(Aggregated to Original Features)")
    axes[0].grid(True, alpha=0.3, axis="x")

    top_mi = summary.sort_values("MI_Score_Sum", ascending=False).head(top_n)
    axes[1].barh(range(len(top_mi)), top_mi["MI_Score_Sum"].values[::-1], color="#FF9800")
    axes[1].set_yticks(range(len(top_mi)))
    axes[1].set_yticklabels(top_mi["Original_Feature"].values[::-1], fontsize=9)
    axes[1].set_xlabel("Aggregated MI Score (Sum)")
    axes[1].set_title("Mutual Information\n(Aggregated to Original Features)")
    axes[1].grid(True, alpha=0.3, axis="x")

    fig.suptitle("Feature Importance by Original (Pre-Encoding) Feature",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")
    plt.show()


def plot_performance_vs_features(sweep_df: pd.DataFrame, optimal_n: int,
                                 pareto_threshold: float,
                                 save_path: str | None = None):
    """
    Line plot of Train/Val F1 vs number of original features.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Must contain 'Original_Features', 'Train_F1', 'Val_F1'.
    optimal_n : int
        Chosen number of features (vertical line).
    pareto_threshold : float
        F1 threshold (horizontal line).
    save_path : str or None
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sweep_df["Original_Features"], sweep_df["Train_F1"],
            marker="o", label="Train F1 (macro)")
    ax.plot(sweep_df["Original_Features"], sweep_df["Val_F1"],
            marker="s", label="Val F1 (unseen groups)")
    ax.axvline(optimal_n, linestyle="--", color="green", alpha=0.7,
               label=f"Chosen: {optimal_n} features")
    ax.axhline(pareto_threshold, linestyle=":", color="grey", alpha=0.5,
               label=f"99% of best ({pareto_threshold:.4f})")
    ax.set_ylim(max(0.0, sweep_df["Val_F1"].min() - 0.05), 1.02)
    ax.set_xlabel("Number of Original Features")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Detection Performance vs Feature Count\n(Group Split — Unseen Computers in Validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")
    plt.show()


def plot_overfit_gap(sweep_df: pd.DataFrame, save_path: str | None = None):
    """
    Bar chart of the generalisation gap (Train F1 − Val F1) per feature count.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Must contain 'Original_Features' and 'Overfit_Gap'.
    save_path : str or None
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["green" if g <= 0 else "salmon" for g in sweep_df["Overfit_Gap"]]
    ax.bar(sweep_df["Original_Features"].astype(str), sweep_df["Overfit_Gap"], color=colours)
    ax.set_xlabel("Number of Original Features")
    ax.set_ylabel("Train F1 − Val F1 (Overfit Gap)")
    ax.set_title("Generalisation Gap by Feature Count\n(Negative = val outperforms train)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")
    plt.show()


def plot_per_class_f1(sweep_df: pd.DataFrame, optimal_n: int,
                      save_path: str | None = None):
    """
    Line plot of per-class F1 (Normal, Attack, Macro) vs feature count.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Must contain 'Original_Features', 'Val_F1_Normal', 'Val_F1_Attack', 'Val_F1'.
    optimal_n : int
        Chosen number of features (vertical line).
    save_path : str or None
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sweep_df["Original_Features"], sweep_df["Val_F1_Normal"],
            marker="^", label="Normal (class 0)")
    ax.plot(sweep_df["Original_Features"], sweep_df["Val_F1_Attack"],
            marker="v", label="Attack (class 1)")
    ax.plot(sweep_df["Original_Features"], sweep_df["Val_F1"],
            marker="s", linestyle="--", label="Macro Average", color="grey")
    ax.axvline(optimal_n, linestyle=":", color="green", alpha=0.7,
               label=f"Chosen: {optimal_n}")
    ax.set_xlabel("Number of Original Features")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Scores vs Feature Count (Validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")
    plt.show()


# ============================================================================
# CONVENIENCE: GENERATE ALL SECTION-10 CHARTS AT ONCE
# ============================================================================

def plot_all_selection_charts(summary, sweep_df, mi_df, optimal_n,
                              pareto_threshold, output_dir: str = "."):
    """
    Generate all feature-selection visualisations (Section 10).

    Parameters
    ----------
    summary : pd.DataFrame
        Original feature importance summary.
    sweep_df : pd.DataFrame
        Feature-count sweep results.
    mi_df : pd.DataFrame
        Encoded-level MI scores.
    optimal_n : int
        Chosen number of original features.
    pareto_threshold : float
        F1 threshold for Pareto selection.
    output_dir : str
        Directory for saved PNGs.
    """
    plot_original_feature_importance(
        summary, save_path=f"{output_dir}/06_original_feature_importance.png"
    )
    plot_performance_vs_features(
        sweep_df, optimal_n, pareto_threshold,
        save_path=f"{output_dir}/07_performance_vs_features.png"
    )
    plot_overfit_gap(
        sweep_df, save_path=f"{output_dir}/08_overfit_gap.png"
    )
    plot_per_class_f1(
        sweep_df, optimal_n, save_path=f"{output_dir}/09_per_class_f1.png"
    )
    plot_mi_encoded(
        mi_df, save_path=f"{output_dir}/10_mi_encoded_features.png"
    )
