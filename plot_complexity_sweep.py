# plot_complexity_sweep.py — Plot complexity sweep results and regression charts.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_CSV = "results/complexity_sweep.csv"
OUTPUT_DIR = "figures"

MODEL_COLORS = {
    "Random Forest": "#3B82F6",
    "LightGBM": "#FF9500",
    "LSTM": "#22C55E",
}

PARAMETER_LABELS = {
    "n_estimators": "Number of Estimators",
    "hidden_units": "LSTM Hidden Units",
    "epochs": "Training Epochs",
}

MODEL_MARKERS = {
    "Random Forest": "o",
    "LightGBM": "s",
    "LSTM": "^",
}


def style_axis(axis):
    # Apply consistent axis styling.
    axis.spines["top"].set_visible(False)
    axis.tick_params(labelsize=10)


def load_sweep_results(csv_path):
    # Load the complexity sweep CSV.
    return pd.read_csv(csv_path)


def save_figure(output_path):
    # Save and close the current matplotlib figure.
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def get_parameter_label(parameter_name):
    # Return a user-friendly parameter label.
    return PARAMETER_LABELS.get(parameter_name, parameter_name)


# ============================================================================
# INDIVIDUAL MODEL CHARTS
# ============================================================================

def get_model_plot_data(model_dataframe):
    # Extract the main arrays used for one model chart.
    parameter_name = model_dataframe["parameter_name"].iloc[0]
    x_values = model_dataframe["parameter_value"].values
    f1_values = model_dataframe["f1"].values
    training_emissions_micrograms = model_dataframe["train_emissions"].values * 1e6
    inference_emissions_micrograms = model_dataframe["inference_emissions"].values * 1e6

    return {
        "parameter_name": parameter_name,
        "x_values": x_values,
        "f1_values": f1_values,
        "training_emissions_micrograms": training_emissions_micrograms,
        "inference_emissions_micrograms": inference_emissions_micrograms,
    }


def plot_f1_axis(axis, x_values, f1_values, model_name):
    # Plot F1 values on the left axis.
    model_color = MODEL_COLORS.get(model_name, "#888888")

    axis.plot(x_values, f1_values, "o-", color=model_color, linewidth=2.5, markersize=8, label="F1-Score", zorder=5)

    axis.set_ylabel("F1-Score", fontsize=12, color=model_color)
    axis.tick_params(axis="y", labelcolor=model_color)

    y_min = min(f1_values) * 0.995
    y_max = min(max(f1_values) * 1.002, 1.0)
    axis.set_ylim(y_min, y_max)

    style_axis(axis)


def plot_emissions_axis(axis, x_values, training_emissions, inference_emissions):
    # Plot training and inference emissions on the right axis.
    axis.plot(x_values, training_emissions, "s--", color="#DC2626", linewidth=2, markersize=7, label="Train CO₂", alpha=0.8)
    axis.plot(x_values, inference_emissions, "^--", color="#9333EA", linewidth=2, markersize=7, label="Inference CO₂", alpha=0.8)

    axis.set_ylabel("CO₂ Emissions (µg)", fontsize=12, color="#DC2626")
    axis.tick_params(axis="y", labelcolor="#DC2626")

    style_axis(axis)


def find_99_percent_peak_f1_index(f1_values):
    # Find the first configuration that achieves at least 99% of peak F1.
    peak_f1 = max(f1_values)
    threshold_f1 = peak_f1 * 0.99

    for index, f1_value in enumerate(f1_values):
        if f1_value >= threshold_f1:
            return index

    return None


def annotate_sustainability_optimal_point(axis, x_values, f1_values, parameter_name, optimal_index):
    # Annotate the first point that reaches 99% of peak F1.
    if optimal_index is None:
        return

    x_value = x_values[optimal_index]
    y_value = f1_values[optimal_index]

    x_offset = (max(x_values) - min(x_values)) * 0.1
    y_offset = (max(f1_values) - min(f1_values)) * 0.3

    axis.axvline(x=x_value, color="gray", linestyle=":", alpha=0.6, linewidth=1.5)
    axis.annotate(
        f"99% peak F1\n{parameter_name}={x_value}",
        xy=(x_value, y_value),
        xytext=(x_value + x_offset, y_value - y_offset),
        fontsize=10,
        fontweight="bold",
        color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )


def add_combined_legend(left_axis, right_axis):
    # Combine legends from both y-axes.
    left_lines, left_labels = left_axis.get_legend_handles_labels()
    right_lines, right_labels = right_axis.get_legend_handles_labels()
    left_axis.legend(left_lines + right_lines, left_labels + right_labels, loc="center right", fontsize=10)


def plot_model_dual_axis(model_dataframe, model_name, output_dir):
    # Plot one model's F1 and emissions against its complexity parameter.
    plot_data = get_model_plot_data(model_dataframe)

    parameter_name = plot_data["parameter_name"]
    x_values = plot_data["x_values"]
    f1_values = plot_data["f1_values"]
    training_emissions = plot_data["training_emissions_micrograms"]
    inference_emissions = plot_data["inference_emissions_micrograms"]

    figure, left_axis = plt.subplots(figsize=(10, 6))

    plot_f1_axis(left_axis, x_values, f1_values, model_name)

    right_axis = left_axis.twinx()
    plot_emissions_axis(right_axis, x_values, training_emissions, inference_emissions)

    left_axis.set_xlabel(get_parameter_label(parameter_name), fontsize=12)

    optimal_index = find_99_percent_peak_f1_index(f1_values)
    annotate_sustainability_optimal_point(left_axis, x_values, f1_values, parameter_name, optimal_index)

    left_axis.set_title(
        f"{model_name}: Detection Performance vs Environmental Cost\n"
        f"Varying {get_parameter_label(parameter_name)}",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    left_axis.grid(axis="y", alpha=0.2)
    add_combined_legend(left_axis, right_axis)

    figure.tight_layout()

    file_name = model_name.lower().replace(" ", "_")
    output_path = os.path.join(output_dir, f"complexity_{file_name}.png")
    save_figure(output_path)

    return optimal_index


# ============================================================================
# SUMMARY CHART
# ============================================================================

def annotate_summary_endpoints(axis, total_emissions, f1_values, parameter_values, color):
    # Label the lowest and highest complexity points.
    axis.annotate(f"{parameter_values[0]}", (total_emissions.iloc[0], f1_values.iloc[0]), fontsize=8, color=color, fontweight="bold", textcoords="offset points", xytext=(-15, -12))
    axis.annotate(f"{parameter_values[-1]}", (total_emissions.iloc[-1], f1_values.iloc[-1]), fontsize=8, color=color, fontweight="bold", textcoords="offset points", xytext=(5, 5))


def plot_summary(dataframe, output_dir):
    # Plot all model configurations on one performance-vs-emissions chart.
    figure, axis = plt.subplots(figsize=(10, 7))

    for model_name, model_group in dataframe.groupby("model"):
        model_color = MODEL_COLORS.get(model_name, "#888888")
        total_emissions_micrograms = (model_group["train_emissions"] + model_group["inference_emissions"]) * 1e6

        axis.scatter(
            total_emissions_micrograms,
            model_group["f1"],
            color=model_color,
            s=80,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.8,
            label=model_name,
            zorder=5,
        )

        axis.plot(total_emissions_micrograms, model_group["f1"], "-", color=model_color, alpha=0.3, linewidth=1.5)

        parameter_values = model_group["parameter_value"].values
        annotate_summary_endpoints(axis, total_emissions_micrograms, model_group["f1"], parameter_values, model_color)

    axis.set_xlabel("Total CO₂ Emissions (µg) - Training + Inference", fontsize=12)
    axis.set_ylabel("F1-Score", fontsize=12)
    axis.set_title(
        "Detection Performance vs Environmental Cost\n"
        "Each point = one complexity configuration",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    axis.legend(fontsize=11, loc="lower right")
    axis.grid(alpha=0.2)
    style_axis(axis)

    figure.tight_layout()

    output_path = os.path.join(output_dir, "complexity_summary.png")
    save_figure(output_path)


# ============================================================================
# HEURISTICS
# ============================================================================

def calculate_heuristic_row(model_name, model_group):
    # Calculate the peak and sustainability-optimal configuration for one model.
    sorted_group = model_group.sort_values("parameter_value").reset_index(drop=True)

    peak_index = sorted_group["f1"].idxmax()
    peak_row = sorted_group.loc[peak_index]

    peak_f1 = peak_row["f1"]
    peak_parameter_value = peak_row["parameter_value"]
    peak_total_emissions = peak_row["train_emissions"] + peak_row["inference_emissions"]

    threshold_f1 = peak_f1 * 0.99
    optimal_row = sorted_group[sorted_group["f1"] >= threshold_f1].iloc[0]

    optimal_parameter_value = optimal_row["parameter_value"]
    optimal_f1 = optimal_row["f1"]
    optimal_total_emissions = optimal_row["train_emissions"] + optimal_row["inference_emissions"]

    f1_loss_percent = (peak_f1 - optimal_f1) / peak_f1 * 100
    emissions_saved_percent = (peak_total_emissions - optimal_total_emissions) / peak_total_emissions * 100

    parameter_name = sorted_group["parameter_name"].iloc[0]

    return {
        "model": model_name,
        "parameter": parameter_name,
        "peak_value": int(peak_parameter_value),
        "peak_f1": peak_f1,
        "peak_co2_ug": peak_total_emissions * 1e6,
        "optimal_value": int(optimal_parameter_value),
        "optimal_f1": optimal_f1,
        "optimal_co2_ug": optimal_total_emissions * 1e6,
        "f1_loss_pct": f1_loss_percent,
        "co2_saved_pct": emissions_saved_percent,
    }


def print_heuristic_row(row):
    # Print one model's heuristic summary.
    print(f"\n  {row['model']}:")
    print(f"    Peak:    {row['parameter']}={row['peak_value']:>4} → F1={row['peak_f1']:.4f}, CO₂={row['peak_co2_ug']:.1f}µg")
    print(f"    Optimal: {row['parameter']}={row['optimal_value']:>4} → F1={row['optimal_f1']:.4f}, CO₂={row['optimal_co2_ug']:.1f}µg")
    print(f"    Saving:  {row['co2_saved_pct']:.1f}% less CO₂ for {row['f1_loss_pct']:.2f}% F1 loss")


def compute_heuristics(dataframe):
    # For each model, find the lowest complexity that reaches 99% of peak F1.
    print("\n" + "=" * 80)
    print("SUSTAINABILITY HEURISTICS")
    print("=" * 80)

    heuristic_rows = []

    for model_name, model_group in dataframe.groupby("model"):
        row = calculate_heuristic_row(model_name, model_group)
        print_heuristic_row(row)
        heuristic_rows.append(row)

    print("=" * 80)

    heuristics_dataframe = pd.DataFrame(heuristic_rows)
    output_path = "results/complexity_heuristics.csv"
    heuristics_dataframe.to_csv(output_path, index=False)

    print(f"\nHeuristics saved to {output_path}")
    return heuristics_dataframe


# ============================================================================
# REGRESSION HELPERS
# ============================================================================

def fit_regression(x_values, y_values):
    # Fit a simple linear regression and return the coefficients.
    x_array = np.array(x_values, dtype=float)
    y_array = np.array(y_values, dtype=float)

    slope, intercept, r_value, p_value, standard_error = stats.linregress(x_array, y_array)

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": standard_error,
    }


def plot_regression_line(axis, x_values, regression_result, color):
    # Plot a dashed regression line.
    x_line = np.linspace(min(x_values), max(x_values), 100)
    y_line = regression_result["slope"] * x_line + regression_result["intercept"]
    axis.plot(x_line, y_line, color=color, linestyle="--", linewidth=1.5)


def plot_model_regression_points(axis, x_values, y_values, regression_result, model_name):
    # Plot scatter points and a regression line for one model.
    model_color = MODEL_COLORS.get(model_name, "#888888")
    marker = MODEL_MARKERS.get(model_name, "o")

    axis.scatter(
        x_values,
        y_values,
        color=model_color,
        marker=marker,
        s=60,
        zorder=5,
        label=f'{model_name} (R$^2$={regression_result["r_squared"]:.4f})',
    )

    plot_regression_line(axis, x_values, regression_result, model_color)


# ============================================================================
# REGRESSION CSV
# ============================================================================

def get_regression_metric_mapping():
    # Return the mapping between output metric names and dataframe columns.
    return {
        "train_co2": "train_emissions",
        "infer_co2": "inference_emissions",
        "train_time": "train_time",
        "infer_time": "inference_time",
    }


def build_regression_row(model_name, parameter_name, metric_key, regression_result):
    # Build one regression-results row.
    phase_name = "training" if "train" in metric_key else "inference"
    measure_name = "co2_kg" if "co2" in metric_key else "time_s"

    return {
        "model": model_name,
        "complexity_parameter": parameter_name,
        "metric": f"{phase_name}_{measure_name}",
        "slope": regression_result["slope"],
        "intercept": regression_result["intercept"],
        "r_squared": regression_result["r_squared"],
        "p_value": regression_result["p_value"],
        "std_err": regression_result["std_err"],
    }


def print_regression_result(model_name, phase_name, measure_name, parameter_name, regression_result):
    # Print one regression equation.
    print(f"  {model_name} {phase_name} {measure_name}:")
    print(f"    y = {regression_result['slope']:.6e} * {parameter_name} + {regression_result['intercept']:.6e}")
    print(f"    R^2 = {regression_result['r_squared']:.6f},  p = {regression_result['p_value']:.2e}")


def compute_regressions(dataframe):
    # Fit linear regressions for every model and every tracked metric.
    regression_rows = []
    metric_mapping = get_regression_metric_mapping()

    print("\n" + "=" * 80)
    print("LINEAR REGRESSION: All Scaling Laws")
    print("=" * 80)

    for model_name, model_group in dataframe.groupby("model"):
        sorted_group = model_group.sort_values("parameter_value")
        x_values = sorted_group["parameter_value"].values
        parameter_name = sorted_group["parameter_name"].iloc[0]

        for metric_key, column_name in metric_mapping.items():
            y_values = sorted_group[column_name].values
            regression_result = fit_regression(x_values, y_values)

            phase_name = "training" if "train" in metric_key else "inference"
            measure_name = "co2_kg" if "co2" in metric_key else "time_s"

            print_regression_result(model_name, phase_name, measure_name, parameter_name, regression_result)

            regression_row = build_regression_row(model_name, parameter_name, metric_key, regression_result)
            regression_rows.append(regression_row)

    print("=" * 80)

    regression_dataframe = pd.DataFrame(regression_rows)
    output_path = "results/complexity_regression.csv"
    regression_dataframe.to_csv(output_path, index=False)

    print(f"\nRegression constants saved to {output_path}")
    return regression_dataframe


# ============================================================================
# REGRESSION CHARTS
# ============================================================================

def configure_regression_axis(axis, x_label, y_label, title):
    # Set labels, title, grid, and styling for a regression axis.
    axis.set_xlabel(x_label, fontsize=10)
    axis.set_ylabel(y_label, fontsize=10)
    axis.set_title(title, fontsize=11)
    axis.legend(fontsize=9)
    axis.grid(True, alpha=0.2)
    style_axis(axis)


def plot_tree_regression_panel(axis, dataframe, emission_column, y_label, title):
    # Plot one regression panel for the tree models.
    tree_model_names = ["Random Forest", "LightGBM"]
    tree_dataframe = dataframe[dataframe["model"].isin(tree_model_names)]

    for model_name in tree_model_names:
        model_group = tree_dataframe[tree_dataframe["model"] == model_name]
        sorted_group = model_group.sort_values("parameter_value")

        x_values = sorted_group["parameter_value"].values
        y_values = sorted_group[emission_column].values
        regression_result = fit_regression(x_values, y_values)

        plot_model_regression_points(axis, x_values, y_values, regression_result, model_name)

    configure_regression_axis(axis, get_parameter_label("n_estimators"), y_label, title)


def plot_regression_tree(dataframe, output_dir):
    # Plot training and inference CO2 scaling for Random Forest and LightGBM.
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    figure.suptitle("CO$_2$ Emissions Scale for Tree Models", fontsize=13, fontweight="bold")

    plot_tree_regression_panel(axes[0], dataframe, "train_emissions", "Training CO$_2$ (kg)", "Training CO$_2$ vs Complexity")
    plot_tree_regression_panel(axes[1], dataframe, "inference_emissions", "Inference CO$_2$ (kg)", "Inference CO$_2$ vs Complexity")

    plt.tight_layout()

    output_path = os.path.join(output_dir, "complexity_co2_regression_tree.png")
    save_figure(output_path)


def plot_lstm_regression_panel(axis, lstm_dataframe, emission_column, y_label, title):
    # Plot one regression panel for the LSTM model.
    if lstm_dataframe.empty:
        return

    x_values = lstm_dataframe["parameter_value"].values
    parameter_name = lstm_dataframe["parameter_name"].iloc[0]
    y_values = lstm_dataframe[emission_column].values
    regression_result = fit_regression(x_values, y_values)

    plot_model_regression_points(axis, x_values, y_values, regression_result, "LSTM")
    configure_regression_axis(axis, get_parameter_label(parameter_name), y_label, title)


def plot_regression_lstm(dataframe, output_dir):
    # Plot training and inference CO2 scaling for the LSTM model.
    lstm_dataframe = dataframe[dataframe["model"] == "LSTM"].sort_values("parameter_value")

    if lstm_dataframe.empty:
        print("  No LSTM data found, skipping.")
        return

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    figure.suptitle("CO$_2$ Emissions Scale for LSTM", fontsize=13, fontweight="bold")

    plot_lstm_regression_panel(axes[0], lstm_dataframe, "train_emissions", "Training CO$_2$ (kg)", "Training CO$_2$ vs Complexity")
    plot_lstm_regression_panel(axes[1], lstm_dataframe, "inference_emissions", "Inference CO$_2$ (kg)", "Inference CO$_2$ vs Complexity")

    plt.tight_layout()

    output_path = os.path.join(output_dir, "complexity_co2_regression_lstm.png")
    save_figure(output_path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading sweep results...")
    sweep_dataframe = load_sweep_results(INPUT_CSV)
    print(f"  {len(sweep_dataframe)} configurations loaded")

    print("\nGenerating individual model charts...")
    for model_name in sweep_dataframe["model"].unique():
        model_dataframe = sweep_dataframe[sweep_dataframe["model"] == model_name].sort_values("parameter_value")
        plot_model_dual_axis(model_dataframe, model_name, OUTPUT_DIR)

    print("\nGenerating summary chart...")
    plot_summary(sweep_dataframe, OUTPUT_DIR)

    compute_heuristics(sweep_dataframe)
    compute_regressions(sweep_dataframe)

    print("\nGenerating regression charts...")
    plot_regression_tree(sweep_dataframe, OUTPUT_DIR)
    plot_regression_lstm(sweep_dataframe, OUTPUT_DIR)


if __name__ == "__main__":
    main()