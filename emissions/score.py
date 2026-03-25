"""
sustainability_score.py — Calculate Sustainability Scores under multiple scenarios.

This script:
1. Loads model-level inference metrics
2. Normalises the metrics to a common scale
3. Applies weighted Sustainability Score scenarios
4. Prints a summary table
5. Saves the results and a comparison chart

Sustainability Score formula:
SS = α·F1 + β·Speed + γ·Low_CO2 + δ·Low_FPR

Where:
- F1 is used directly after min-max normalisation
- Speed is inverted from inference time so lower time becomes better
- Low_CO2 is inverted from emissions so lower emissions become better
- Low_FPR is inverted from false positive rate so lower FPR becomes better
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# ============================================================================
# CONFIGURATION
# ============================================================================

SCENARIOS = {
    "Equal": {
        "alpha": 0.25,
        "beta": 0.25,
        "gamma": 0.25,
        "delta": 0.25,
    },
    "Detection_First": {
        "alpha": 0.50,
        "beta": 0.10,
        "gamma": 0.20,
        "delta": 0.20,
    },
    "Speed_First": {
        "alpha": 0.20,
        "beta": 0.50,
        "gamma": 0.15,
        "delta": 0.15,
    },
    "Green_First": {
        "alpha": 0.15,
        "beta": 0.15,
        "gamma": 0.55,
        "delta": 0.15,
    },
    "SOC_Operational": {
        "alpha": 0.20,
        "beta": 0.15,
        "gamma": 0.15,
        "delta": 0.50,
    },
}

MODEL_COLORS = {
    "LightGBM": "#FF9500",
    "Random Forest": "#3B82F6",
    "LSTM": "#22C55E",
}

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# NORMALISATION HELPERS
# ============================================================================

def min_max_normalise(series):
    """Min-max normalise a pandas Series to the range [0, 1]."""
    scaler = MinMaxScaler()

    normalised_values = scaler.fit_transform(series.to_frame()).flatten()

    normalised_series = pd.Series(
        normalised_values,
        index=series.index,
    )

    return normalised_series


def normalise_f1(series):
    """Normalise F1 so higher remains better."""
    return min_max_normalise(series)


def normalise_time_as_speed(series):
    """Normalise inference time and invert it so lower time becomes better."""
    return 1 - min_max_normalise(series)


def normalise_emissions_as_low_co2(series):
    """Normalise emissions and invert them so lower CO2 becomes better."""
    return 1 - min_max_normalise(series)


def normalise_fpr_as_low_fpr(series):
    """Normalise false positive rate and invert it so lower FPR becomes better."""
    return 1 - min_max_normalise(series)


def add_normalised_metric_columns(dataframe):
    """
    Add normalised metric columns used by the Sustainability Score.
    """
    normalised_dataframe = dataframe.copy()

    normalised_dataframe["f1_norm"] = normalise_f1(normalised_dataframe["f1_mean"])
    normalised_dataframe["time_norm"] = normalise_time_as_speed(
        normalised_dataframe["inference_time_mean"]
    )
    normalised_dataframe["emissions_norm"] = normalise_emissions_as_low_co2(
        normalised_dataframe["emissions_mean"]
    )
    normalised_dataframe["fpr_norm"] = normalise_fpr_as_low_fpr(
        normalised_dataframe["fpr_mean"]
    )

    return normalised_dataframe


# ============================================================================
# SUSTAINABILITY SCORE CALCULATION
# ============================================================================

def calculate_sustainability_score_for_row(row, weights):
    """
    Calculate one Sustainability Score using one scenario's weights.
    """
    score = (
        weights["alpha"] * row["f1_norm"]
        + weights["beta"] * row["time_norm"]
        + weights["gamma"] * row["emissions_norm"]
        + weights["delta"] * row["fpr_norm"]
    )

    return score


def add_sustainability_score_columns(dataframe, scenarios):
    """
    Add one Sustainability Score column for each scenario.
    """
    scored_dataframe = dataframe.copy()

    for scenario_name, scenario_weights in scenarios.items():
        scenario_scores = []

        for _, row in scored_dataframe.iterrows():
            score = calculate_sustainability_score_for_row(row, scenario_weights)
            scenario_scores.append(score)

        scored_dataframe[scenario_name] = scenario_scores

    return scored_dataframe


def compute_sustainability_scores(dataframe, scenarios=SCENARIOS):
    """
    Normalise the raw metrics and compute Sustainability Scores
    for every scenario.
    """
    scored_dataframe = add_normalised_metric_columns(dataframe)
    scored_dataframe = add_sustainability_score_columns(scored_dataframe, scenarios)

    return scored_dataframe


# ============================================================================
# VISUALISATION
# ============================================================================

def get_scenario_display_names(scenarios):
    """Return scenario names with underscores replaced by spaces."""
    display_names = []

    for scenario_name in scenarios.keys():
        display_names.append(scenario_name.replace("_", " "))

    return display_names


def get_model_scores_for_scenario_list(dataframe, model_name, scenario_names):
    """Return one model's Sustainability Scores in scenario order."""
    model_row = dataframe[dataframe["model"] == model_name].iloc[0]

    scores = []
    for scenario_name in scenario_names:
        scores.append(model_row[scenario_name])

    return scores


def plot_sustainability_scores(dataframe, scenarios=SCENARIOS, output_dir="figures"):
    """Create a grouped bar chart comparing Sustainability Scores."""
    os.makedirs(output_dir, exist_ok=True)

    scenario_names = list(scenarios.keys())
    scenario_display_names = get_scenario_display_names(scenarios)
    model_names = dataframe["model"].tolist()

    x_positions = np.arange(len(scenario_names))
    bar_width = 0.25
    bar_offsets = [-bar_width, 0, bar_width]

    figure, axis = plt.subplots(figsize=(12, 6))

    for model_index, model_name in enumerate(model_names):
        model_scores = get_model_scores_for_scenario_list(
            dataframe,
            model_name,
            scenario_names,
        )

        model_color = MODEL_COLORS.get(model_name, "#888888")

        bars = axis.bar(
            x_positions + bar_offsets[model_index],
            model_scores,
            bar_width,
            label=model_name,
            color=model_color,
            edgecolor="white",
            linewidth=0.8,
        )

        for bar, score in zip(bars, model_scores):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    axis.set_title(
        "Sustainability Score by Scenario\n"
        "SS = αF1 + β(1−Inference) + γ(1−CO₂) + δ(1−FPR)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    axis.set_ylabel("Sustainability Score", fontsize=11)
    axis.set_xlabel("Scenario", fontsize=11)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(scenario_display_names, fontsize=10)
    axis.set_ylim(0, 1.05)
    axis.legend(fontsize=10, loc="upper right")
    axis.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = f"{output_dir}/sustainability_score_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Chart saved to {output_path}")


# ============================================================================
# SUMMARY PRINTING
# ============================================================================

def print_raw_metrics_table(dataframe):
    """Print the raw metric values."""
    print("\nRaw Metrics:")
    print(f"  {'Model':<15} {'F1':>8} {'Time(s)':>10} {'CO₂(µg)':>10} {'FPR':>12}")

    for _, row in dataframe.iterrows():
        print(
            f"  {row['model']:<15} "
            f"{row['f1_mean']:>8.4f} "
            f"{row['inference_time_mean']:>10.3f} "
            f"{row['emissions_mean'] * 1e6:>10.2f} "
            f"{row['fpr_mean']:>12.7f}"
        )


def print_normalised_metrics_table(dataframe):
    """Print the normalised metric values."""
    print("\nNormalised Metrics (higher = better):")
    print(f"  {'Model':<15} {'F1':>8} {'Speed':>8} {'Low CO₂':>8} {'Low FPR':>8}")

    for _, row in dataframe.iterrows():
        print(
            f"  {row['model']:<15} "
            f"{row['f1_norm']:>8.4f} "
            f"{row['time_norm']:>8.4f} "
            f"{row['emissions_norm']:>8.4f} "
            f"{row['fpr_norm']:>8.4f}"
        )


def print_sustainability_score_table(dataframe, scenarios):
    """Print the Sustainability Score values for each scenario."""
    scenario_names = list(scenarios.keys())

    print("\nSustainability Scores:")
    header = f"  {'Model':<15}"

    for scenario_name in scenario_names:
        header += f" {scenario_name:>16}"

    print(header)

    for _, row in dataframe.iterrows():
        line = f"  {row['model']:<15}"

        for scenario_name in scenario_names:
            line += f" {row[scenario_name]:>16.4f}"

        print(line)


def print_scenario_winners(dataframe, scenarios):
    """Print the winning model for each scenario."""
    scenario_names = list(scenarios.keys())

    print("\nWinner per Scenario:")

    for scenario_name in scenario_names:
        winner_index = dataframe[scenario_name].idxmax()
        winner_name = dataframe.loc[winner_index, "model"]
        winner_score = dataframe.loc[winner_index, scenario_name]

        print(f"  {scenario_name:<20} → {winner_name} ({winner_score:.4f})")


def print_summary(dataframe, scenarios=SCENARIOS):
    """Print a full summary of the Sustainability Score results."""
    print("\n" + "=" * 80)
    print("SUSTAINABILITY SCORE RESULTS")
    print("=" * 80)

    print_raw_metrics_table(dataframe)
    print_normalised_metrics_table(dataframe)
    print_sustainability_score_table(dataframe, scenarios)
    print_scenario_winners(dataframe, scenarios)

    print("=" * 80)


# ============================================================================
# SAVING
# ============================================================================

def save_results_csv(dataframe, output_path="results/sustainability_scores.csv"):
    """Save the full Sustainability Score table to CSV."""
    os.makedirs("results", exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(f"\n  Full results saved to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def run_sustainability_analysis(
    csv_path=os.path.join(SCRIPT_DIRECTORY, "model_metrics.csv"),
    output_dir="figures",
):
    """Run the full Sustainability Score analysis."""
    print("Loading inference results from:", csv_path)

    results_dataframe = pd.read_csv(csv_path)

    scored_dataframe = compute_sustainability_scores(results_dataframe)

    print_summary(scored_dataframe)
    plot_sustainability_scores(scored_dataframe, output_dir=output_dir)
    save_results_csv(scored_dataframe)

    return scored_dataframe


if __name__ == "__main__":
    run_sustainability_analysis()