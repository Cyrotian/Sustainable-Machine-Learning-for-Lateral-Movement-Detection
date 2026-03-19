"""
plotting.py — Generate all dissertation charts.

PURPOSE: Produce publication-quality figures. Each function takes data
         as input so charts can be reproduced without rerunning the pipeline.

CHARTS:
  1. Class distribution pie chart
  2. EventID bar chart (Normal vs Malicious)
  3. MI bar chart (top encoded features)
  4. RF vs MI vs CatBoost importance (three-panel, original features)
  4b. CatBoost importance bar chart (top encoded features)
  5. F1 vs feature count (key sustainability chart)
  6. Overfit gap
  7. Per-class F1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Sysmon EventID lookup — only needed for the EventID bar chart
EVENT_MAPPING = {
    1:   "Process Creation",
    2:   "File Creation Time Changed",
    3:   "Network Connection",
    5:   "Process Terminated",
    7:   "Image Loaded",
    8:   "CreateRemoteThread",
    10:  "ProcessAccess",
    11:  "FileCreate",
    12:  "RegistryEvent (Create/Delete)",
    13:  "RegistryEvent (Value Set)",
    15:  "FileCreateStreamHash",
    17:  "PipeEvent (Created)",
    18:  "PipeEvent (Connected)",
    22:  "DNSEvent",
    23:  "FileDelete",
    255: "Error",
}


def setup_plot_style():
    """Set consistent defaults for all charts."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10


# ============================================================================
# CHART 1: Class Distribution
# ============================================================================

def plot_class_distribution(df, save_path=None):
    """
    Pie chart: % malicious vs normal.
    VIVA: "The dataset is imbalanced — this is why I used class_weight='balanced'."
    """
    counts = df["isMalicious"].value_counts().sort_index()

    fig, ax = plt.subplots()
    counts.plot(
        kind="pie",
        colors=["skyblue", "salmon"],
        autopct="%1.0f%%",
        startangle=90,
        ax=ax,
        labels=["", ""],
    )
    ax.set_title("Distribution of Malicious vs Normal Samples (LMD-2022)")
    ax.legend(["Normal", "Malicious"], loc="lower left", bbox_to_anchor=(1, 0.5))
    ax.set_ylabel("")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)


# ============================================================================
# CHART 2: EventID Counts by Class
# ============================================================================

def plot_event_counts(df, save_path=None):
    """
    Horizontal bar chart: Sysmon EventID frequency, Normal vs Malicious.
    """
    events = df[["EventID", "isMalicious"]].copy()

    counts = events.groupby(["EventID", "isMalicious"]).size().unstack(fill_value=0)
    counts.columns = ["Normal", "Malicious"]

    # Sort by total count
    counts["Total"] = counts["Normal"] + counts["Malicious"]
    counts = counts.sort_values("Total")
    counts = counts.drop(columns="Total")

    # Add human-readable names
    new_labels = []
    for event_id in counts.index:
        name = EVENT_MAPPING.get(event_id, "")
        new_labels.append(str(event_id) + " - " + name)
    counts.index = new_labels

    counts.plot.barh(figsize=(12, 8), color=["skyblue", "salmon"])
    plt.xscale("log")
    plt.xlabel("Count (log scale)")
    plt.title("EventID Counts: Normal vs Malicious")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)


# ============================================================================
# CHART 3: Mutual Information (Encoded Features)
# ============================================================================

def plot_mi_encoded(mi_df, top_n=30, save_path=None):
    """Bar chart of top encoded features by MI score."""
    if top_n > len(mi_df):
        top_n = len(mi_df)

    top_features = mi_df.head(top_n)
    names = top_features["Feature"].values
    scores = top_features["MI_Score"].values

    plt.figure(figsize=(12, 7))
    positions = range(top_n)
    plt.barh(positions, scores[::-1])
    plt.yticks(positions, names[::-1], fontsize=7)
    plt.xlabel("Mutual Information Score")
    plt.title("Top " + str(top_n) + " Encoded Features by Mutual Information")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)


# ============================================================================
# CHART 4: RF vs MI vs CatBoost Importance (Original Features)
# ============================================================================

def plot_original_feature_importance(summary, top_n=25, save_path=None):
    """
    Side-by-side panels: RF (left), MI (centre), CatBoost (right, if available).
    VIVA: "All three methods agree on the top features — the ranking is robust."
    """
    has_cb = "CB_Importance_Sum" in summary.columns

    n_panels = 3 if has_cb else 2
    fig_width = 24 if has_cb else 16
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 8))

    # Left: RF
    top_rf = summary.head(top_n)
    rf_names = top_rf["Original_Feature"].values
    rf_scores = top_rf["RF_Importance_Sum"].values

    axes[0].barh(range(len(top_rf)), rf_scores[::-1], color="#2196F3")
    axes[0].set_yticks(range(len(top_rf)))
    axes[0].set_yticklabels(rf_names[::-1], fontsize=9)
    axes[0].set_xlabel("RF Importance")
    axes[0].set_title("Random Forest Importance")

    # Centre: MI
    top_mi = summary.sort_values("MI_Score_Sum", ascending=False).head(top_n)
    mi_names = top_mi["Original_Feature"].values
    mi_scores = top_mi["MI_Score_Sum"].values

    axes[1].barh(range(len(top_mi)), mi_scores[::-1], color="#FF9800")
    axes[1].set_yticks(range(len(top_mi)))
    axes[1].set_yticklabels(mi_names[::-1], fontsize=9)
    axes[1].set_xlabel("MI Score")
    axes[1].set_title("Mutual Information")

    # Right: CatBoost (optional)
    if has_cb:
        top_cb = summary.sort_values("CB_Importance_Sum", ascending=False).head(top_n)
        cb_names = top_cb["Original_Feature"].values
        cb_scores = top_cb["CB_Importance_Sum"].values

        axes[2].barh(range(len(top_cb)), cb_scores[::-1], color="#4CAF50")
        axes[2].set_yticks(range(len(top_cb)))
        axes[2].set_yticklabels(cb_names[::-1], fontsize=9)
        axes[2].set_xlabel("CatBoost Importance")
        axes[2].set_title("CatBoost Importance")

    fig.suptitle("Feature Importance: Method Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)


# ============================================================================
# CHART 4b: CatBoost Importance (Encoded Features)
# ============================================================================

def plot_cb_encoded(cb_df, top_n=30, save_path=None):
    """Bar chart of top encoded features by CatBoost importance score."""
    if top_n > len(cb_df):
        top_n = len(cb_df)

    top_features = cb_df.head(top_n)
    names = top_features["Feature"].values
    scores = top_features["CB_Importance"].values

    plt.figure(figsize=(12, 7))
    positions = range(top_n)
    plt.barh(positions, scores[::-1], color="#4CAF50")
    plt.yticks(positions, names[::-1], fontsize=7)
    plt.xlabel("CatBoost Importance Score")
    plt.title("Top " + str(top_n) + " Encoded Features by CatBoost Importance")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)


# ============================================================================
# CHART 5: F1 vs Feature Count (KEY SUSTAINABILITY CHART)
# ============================================================================

def plot_performance_vs_features(sweep_df, optimal_n, pareto_threshold, save_path=None):
    """
    THE MOST IMPORTANT CHART.
    VIVA: "Performance plateaus at 5 features — adding more doesn't improve
           F1 but increases computation. This justifies the 88% reduction."
    """
    feature_counts = sweep_df["Original_Features"]
    train_f1 = sweep_df["Train_F1"]
    val_f1 = sweep_df["Val_F1"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(feature_counts, train_f1, marker="o", label="Train F1")
    ax.plot(feature_counts, val_f1, marker="s", label="Val F1 (unseen computers)")
    ax.axvline(optimal_n, linestyle="--", color="green", alpha=0.7,
               label="Chosen: " + str(optimal_n) + " features")
    ax.axhline(pareto_threshold, linestyle=":", color="grey", alpha=0.5,
               label="Threshold: " + str(round(pareto_threshold, 4)))
    ax.set_xlabel("Number of Original Features")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Detection Performance vs Feature Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)


# ============================================================================
# CHART 6: Overfit Gap
# ============================================================================

def plot_overfit_gap(sweep_df, save_path=None):
    """
    VIVA: "The gap is near zero — no overfitting, even on unseen computers."
    """
    feature_counts = sweep_df["Original_Features"].astype(str)
    gaps = sweep_df["Overfit_Gap"]

    colours = []
    for gap in gaps:
        if gap <= 0:
            colours.append("green")
        else:
            colours.append("salmon")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(feature_counts, gaps, color=colours)
    ax.set_xlabel("Number of Original Features")
    ax.set_ylabel("Train F1 - Val F1")
    ax.set_title("Generalisation Gap by Feature Count")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)


# ============================================================================
# CHART 7: Per-Class F1
# ============================================================================

def plot_per_class_f1(sweep_df, optimal_n, save_path=None):
    """
    VIVA: "Strong F1 for BOTH classes — not biased toward the majority."
    """
    feature_counts = sweep_df["Original_Features"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(feature_counts, sweep_df["Val_F1_Normal"], marker="^", label="Normal")
    ax.plot(feature_counts, sweep_df["Val_F1_Attack"], marker="v", label="Attack")
    ax.plot(feature_counts, sweep_df["Val_F1"], marker="s",
            linestyle="--", label="Macro Average", color="grey")
    ax.axvline(optimal_n, linestyle=":", color="green", alpha=0.7,
               label="Chosen: " + str(optimal_n))
    ax.set_xlabel("Number of Original Features")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 vs Feature Count (Validation)")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)


# ============================================================================
# CONVENIENCE: Generate all charts at once
# ============================================================================

def plot_all(raw_df, results, output_dir="."):
    """
    Generate all dissertation charts. Call from main.py like:
        plot_all(df, results, output_dir="plots")

    Parameters:
      raw_df  = the raw LMD-2022 dataframe (with isMalicious and EventID)
      results = dict returned by run_feature_selection_pipeline()
      output_dir = folder to save PNG files into
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    setup_plot_style()

    # EDA charts (need raw data)
    plot_class_distribution(raw_df, save_path=output_dir + "/01_class_distribution.png")
    plot_event_counts(raw_df, save_path=output_dir + "/02_event_counts.png")

    # Feature selection charts (need pipeline results)
    plot_mi_encoded(
        results["mi_df"],
        save_path=output_dir + "/03_mi_encoded_features.png",
    )
    plot_original_feature_importance(
        results["summary"],
        save_path=output_dir + "/04_original_feature_importance.png",
    )

    # CatBoost encoded chart (only if cb_df present in results)
    cb_df = results.get("cb_df")
    n_charts = 7
    if cb_df is not None:
        plot_cb_encoded(
            cb_df,
            save_path=output_dir + "/04b_catboost_encoded_features.png",
        )
        n_charts = 8

    plot_performance_vs_features(
        results["sweep_df"],
        results["OPTIMAL_N"],
        results["pareto_threshold"],
        save_path=output_dir + "/05_performance_vs_features.png",
    )
    plot_overfit_gap(
        results["sweep_df"],
        save_path=output_dir + "/06_overfit_gap.png",
    )
    plot_per_class_f1(
        results["sweep_df"],
        results["OPTIMAL_N"],
        save_path=output_dir + "/07_per_class_f1.png",
    )

    print("\nAll", n_charts, "charts saved to:", output_dir)

