# Generate  charts.

import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================================
# CONFIGURATION
# ============================================================================

EVENT_MAPPING = {
    1: "Process Creation",
    2: "File Creation Time Changed",
    3: "Network Connection",
    5: "Process Terminated",
    7: "Image Loaded",
    8: "CreateRemoteThread",
    10: "ProcessAccess",
    11: "FileCreate",
    12: "RegistryEvent (Create/Delete)",
    13: "RegistryEvent (Value Set)",
    15: "FileCreateStreamHash",
    17: "PipeEvent (Created)",
    18: "PipeEvent (Connected)",
    22: "DNSEvent",
    23: "FileDelete",
    255: "Error",
}


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def setup_plot_style():
    # Apply a consistent visual style to all charts.
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10


def save_and_close(figure, save_path):
    # Save a figure if a path is provided, then close it.
    if save_path is not None:
        figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print("Saved:", save_path)


def get_top_rows(dataframe, top_n):
    # Return the top N rows, capped at the dataframe length.
    adjusted_top_n = min(top_n, len(dataframe))
    top_rows = dataframe.head(adjusted_top_n)
    return top_rows, adjusted_top_n


def reverse_for_barh(labels, values):
    # Reverse labels and values so the largest item appears at the top.
    return labels[::-1], values[::-1]


# CLASS DISTRIBUTION


def get_class_distribution_counts(dataframe):
    # Count normal and malicious samples.
    return dataframe["isMalicious"].value_counts().sort_index()


def plot_class_distribution(dataframe, save_path=None):
    # Plot the malicious vs normal class distribution.
    class_counts = get_class_distribution_counts(dataframe)

    figure, axis = plt.subplots()

    class_counts.plot(
        kind="pie",
        colors=["skyblue", "salmon"],
        autopct="%1.0f%%",
        startangle=90,
        ax=axis,
        labels=["", ""],
    )

    axis.set_title("Distribution of Malicious vs Normal Samples (LMD-2022)")
    axis.legend(["Normal", "Malicious"], loc="lower left", bbox_to_anchor=(1, 0.5))
    axis.set_ylabel("")

    plt.tight_layout()
    save_and_close(figure, save_path)



# EVENTID COUNTS BY CLASS


def build_event_count_table(dataframe):
    # Build a Normal vs Malicious count table for each EventID.
    event_dataframe = dataframe[["EventID", "isMalicious"]].copy()

    event_counts = (
        event_dataframe.groupby(["EventID", "isMalicious"])
        .size()
        .unstack(fill_value=0)
    )

    event_counts.columns = ["Normal", "Malicious"]
    event_counts["Total"] = event_counts["Normal"] + event_counts["Malicious"]
    event_counts = event_counts.sort_values("Total")
    event_counts = event_counts.drop(columns="Total")

    return event_counts


def add_event_labels(event_count_table):
    # Replace EventID index values with "EventID - Name" labels.
    labelled_table = event_count_table.copy()
    new_index_labels = []

    for event_id in labelled_table.index:
        event_name = EVENT_MAPPING.get(event_id, "")
        new_index_labels.append(str(event_id) + " - " + event_name)

    labelled_table.index = new_index_labels
    return labelled_table


def plot_event_counts(dataframe, save_path=None):
    # Plot EventID frequencies for normal and malicious samples.
    event_count_table = build_event_count_table(dataframe)
    event_count_table = add_event_labels(event_count_table)

    figure, axis = plt.subplots(figsize=(12, 8))

    event_count_table.plot.barh(
        ax=axis,
        color=["skyblue", "salmon"],
    )

    axis.set_xscale("log")
    axis.set_xlabel("Count (log scale)")
    axis.set_title("EventID Counts: Normal vs Malicious")

    plt.tight_layout()
    save_and_close(figure, save_path)



# MUTUAL INFORMATION (ENCODED FEATURES)


def plot_mi_encoded(mutual_information_dataframe, top_n=30, save_path=None):
    # Plot the top encoded features ranked by Mutual Information.
    top_features, adjusted_top_n = get_top_rows(mutual_information_dataframe, top_n)

    feature_names = top_features["Feature"].values
    feature_scores = top_features["MI_Score"].values

    feature_names, feature_scores = reverse_for_barh(
        feature_names,
        feature_scores,
    )

    figure, axis = plt.subplots(figsize=(12, 7))

    axis.barh(range(adjusted_top_n), feature_scores)
    axis.set_yticks(range(adjusted_top_n))
    axis.set_yticklabels(feature_names, fontsize=7)
    axis.set_xlabel("Mutual Information Score")
    axis.set_title(f"Top {adjusted_top_n} Encoded Features by Mutual Information")

    plt.tight_layout()
    save_and_close(figure, save_path)



 # ORIGINAL FEATURE IMPORTANCE COMPARISON


def plot_random_forest_importance_panel(axis, summary_dataframe, top_n):
    # Plot the Random Forest importance panel.
    top_random_forest = summary_dataframe.head(top_n)

    feature_names = top_random_forest["Original_Feature"].values
    feature_scores = top_random_forest["RF_Importance_Sum"].values

    feature_names, feature_scores = reverse_for_barh(
        feature_names,
        feature_scores,
    )

    axis.barh(range(len(top_random_forest)), feature_scores, color="#2196F3")
    axis.set_yticks(range(len(top_random_forest)))
    axis.set_yticklabels(feature_names, fontsize=9)
    axis.set_xlabel("RF Importance")
    axis.set_title("Random Forest Importance")


def plot_mutual_information_panel(axis, summary_dataframe, top_n):
    # Plot the Mutual Information panel.
    top_mutual_information = (
        summary_dataframe.sort_values("MI_Score_Sum", ascending=False)
        .head(top_n)
    )

    feature_names = top_mutual_information["Original_Feature"].values
    feature_scores = top_mutual_information["MI_Score_Sum"].values

    feature_names, feature_scores = reverse_for_barh(
        feature_names,
        feature_scores,
    )

    axis.barh(range(len(top_mutual_information)), feature_scores, color="#FF9800")
    axis.set_yticks(range(len(top_mutual_information)))
    axis.set_yticklabels(feature_names, fontsize=9)
    axis.set_xlabel("MI Score")
    axis.set_title("Mutual Information")


def plot_catboost_importance_panel(axis, summary_dataframe, top_n):
    # Plot the CatBoost importance panel.
    top_catboost = (
        summary_dataframe.sort_values("CB_Importance_Sum", ascending=False)
        .head(top_n)
    )

    feature_names = top_catboost["Original_Feature"].values
    feature_scores = top_catboost["CB_Importance_Sum"].values

    feature_names, feature_scores = reverse_for_barh(
        feature_names,
        feature_scores,
    )

    axis.barh(range(len(top_catboost)), feature_scores, color="#4CAF50")
    axis.set_yticks(range(len(top_catboost)))
    axis.set_yticklabels(feature_names, fontsize=9)
    axis.set_xlabel("CatBoost Importance")
    axis.set_title("CatBoost Importance")


def plot_original_feature_importance(summary_dataframe, top_n=25, save_path=None):
    # Plot side-by-side importance charts for original features.
    has_catboost = "CB_Importance_Sum" in summary_dataframe.columns

    number_of_panels = 3 if has_catboost else 2
    figure_width = 24 if has_catboost else 16

    figure, axes = plt.subplots(1, number_of_panels, figsize=(figure_width, 8))

    plot_random_forest_importance_panel(axes[0], summary_dataframe, top_n)
    plot_mutual_information_panel(axes[1], summary_dataframe, top_n)

    if has_catboost:
        plot_catboost_importance_panel(axes[2], summary_dataframe, top_n)

    figure.suptitle(
        "Feature Importance: Method Comparison",
        fontsize=14,
        fontweight="bold",
    )

    figure.tight_layout()
    save_and_close(figure, save_path)



# CATBOOST IMPORTANCE


def plot_cb_encoded(catboost_dataframe, top_n=30, save_path=None):
    # Plot the top encoded features ranked by CatBoost importance.
    top_features, adjusted_top_n = get_top_rows(catboost_dataframe, top_n)

    feature_names = top_features["Feature"].values
    feature_scores = top_features["CB_Importance"].values

    feature_names, feature_scores = reverse_for_barh(
        feature_names,
        feature_scores,
    )

    figure, axis = plt.subplots(figsize=(12, 7))

    axis.barh(range(adjusted_top_n), feature_scores, color="#4CAF50")
    axis.set_yticks(range(adjusted_top_n))
    axis.set_yticklabels(feature_names, fontsize=7)
    axis.set_xlabel("CatBoost Importance Score")
    axis.set_title(f"Top {adjusted_top_n} Encoded Features by CatBoost Importance")

    plt.tight_layout()
    save_and_close(figure, save_path)



# PERFORMANCE VS FEATURE COUNT

def plot_performance_vs_features(
    sweep_dataframe,
    optimal_feature_count,
    pareto_threshold,
    save_path=None,
):
    # Plot training and validation F1 against the number of original features.
    feature_counts = sweep_dataframe["Original_Features"]
    training_f1 = sweep_dataframe["Train_F1"]
    validation_f1 = sweep_dataframe["Val_F1"]

    figure, axis = plt.subplots(figsize=(10, 5))

    axis.plot(feature_counts, training_f1, marker="o", label="Train F1")
    axis.plot(
        feature_counts,
        validation_f1,
        marker="s",
        label="Val F1 (unseen computers)",
    )

    axis.axvline(
        optimal_feature_count,
        linestyle="--",
        color="green",
        alpha=0.7,
        label=f"Chosen: {optimal_feature_count} features",
    )

    axis.axhline(
        pareto_threshold,
        linestyle=":",
        color="grey",
        alpha=0.5,
        label=f"Threshold: {round(pareto_threshold, 4)}",
    )

    axis.set_xlabel("Number of Original Features")
    axis.set_ylabel("Macro F1 Score")
    axis.set_title("Detection Performance vs Feature Count")
    axis.legend()
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    save_and_close(figure, save_path)


# GENERATE ALL CHARTS


def plot_all(raw_dataframe, results, output_dir="."):
    # Generate all chart
    os.makedirs(output_dir, exist_ok=True)

    setup_plot_style()

    plot_class_distribution(
        raw_dataframe,
        save_path=output_dir + "/01_class_distribution.png",
    )

    plot_event_counts(
        raw_dataframe,
        save_path=output_dir + "/02_event_counts.png",
    )

    plot_mi_encoded(
        results["mi_df"],
        save_path=output_dir + "/03_mi_encoded_features.png",
    )

    plot_original_feature_importance(
        results["summary"],
        save_path=output_dir + "/04_original_feature_importance.png",
    )

    catboost_dataframe = results.get("cb_df")
    number_of_charts = 7

    if catboost_dataframe is not None:
        plot_cb_encoded(
            catboost_dataframe,
            save_path=output_dir + "/04b_catboost_encoded_features.png",
        )
        number_of_charts = 8

    plot_performance_vs_features(
        results["sweep_df"],
        results["OPTIMAL_N"],
        results["pareto_threshold"],
        save_path=output_dir + "/05_performance_vs_features.png",
    )

    print("\nAll", number_of_charts, "charts saved to:", output_dir)