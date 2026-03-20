"""
sustainability_score.py — Calculate the Sustainability Score with sensitivity analysis.

PURPOSE: Apply the weighted Sustainability Score formula mapped to
         Usman et al.'s Systems Theory Model (T, E, S):
         
         Technical (T):     F1-Score (α) + Inference Time (β)
         Environmental (E): CO₂ Emissions (γ)
         Social (S):        False Positive Rate (δ)

FORMULA: SS = α·F̃1 + β·(1 - Ĩ) + γ·(1 - Ẽ) + δ·(1 - FPR̃)

         All metrics min-max normalised to [0, 1].
         Cost metrics (time, emissions, FPR) inverted so higher = better.

INPUT:   CSV with columns: model, f1_mean, inference_time_mean, emissions_mean, fpr_mean
OUTPUT:  CSV with normalised values and SS for each scenario + bar chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# 5 sensitivity scenarios: weights must sum to 1.0
SCENARIOS = {
    "Equal":           {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25},
    "Detection_First": {"alpha": 0.50, "beta": 0.10, "gamma": 0.20, "delta": 0.20},
    "Speed_First":     {"alpha": 0.20, "beta": 0.50, "gamma": 0.15, "delta": 0.15},
    "Green_First":     {"alpha": 0.15, "beta": 0.15, "gamma": 0.55, "delta": 0.15},
    "SOC_Operational": {"alpha": 0.20, "beta": 0.15, "gamma": 0.15, "delta": 0.50},
}

# Model display colours
MODEL_COLORS = {
    "LightGBM": "#FF9500",
    "Random Forest": "#3B82F6",
    "LSTM": "#22C55E",
}


# ============================================================================
# NORMALISATION
# ============================================================================

def min_max_normalise(series):
    """Min-max normalise a series to [0, 1]."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def normalise_metrics(df):
    """
    Normalise all metrics to [0, 1] and invert cost metrics.

    After normalisation:
      - F1: higher = better (no inversion)
      - Time: inverted (1 - norm) so lower time = higher score
      - Emissions: inverted (1 - norm) so lower CO₂ = higher score
      - FPR: inverted (1 - norm) so lower FPR = higher score
    """
    df = df.copy()

    # Normalise
    df["f1_norm"] = min_max_normalise(df["f1_mean"])
    df["time_norm"] = 1 - min_max_normalise(df["inference_time_mean"])
    df["emissions_norm"] = 1 - min_max_normalise(df["emissions_mean"])
    df["fpr_norm"] = 1 - min_max_normalise(df["fpr_mean"])

    return df


# ============================================================================
# SUSTAINABILITY SCORE CALCULATION
# ============================================================================

def compute_sustainability_scores(df, scenarios=SCENARIOS):
    """
    Compute SS for each model under each scenario.

    SS = α·F̃1 + β·(1 - Ĩ) + γ·(1 - Ẽ) + δ·(1 - FPR̃)
    """
    df = normalise_metrics(df)

    for name, weights in scenarios.items():
        df[name] = (
            weights["alpha"] * df["f1_norm"] +
            weights["beta"]  * df["time_norm"] +
            weights["gamma"] * df["emissions_norm"] +
            weights["delta"] * df["fpr_norm"]
        )

    return df



# ============================================================================
# VISUALISATION
# ============================================================================

def plot_sustainability_scores(df, scenarios=SCENARIOS, output_dir="figures"):
    """Generate grouped bar chart of SS across all scenarios."""
    os.makedirs(output_dir, exist_ok=True)

    scenario_names = list(scenarios.keys())
    display_names = [s.replace("_", " ") for s in scenario_names]
    models = df["model"].tolist()

    x = np.arange(len(scenario_names))
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        scores = [df[df["model"] == model][s].values[0] for s in scenario_names]
        color = MODEL_COLORS.get(model, "#888888")
        bars = ax.bar(x + offsets[i], scores, width, label=model,
                      color=color, edgecolor="white", linewidth=0.8)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{score:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")

    ax.set_title("Sustainability Score by Scenario\n"
                 "SS = αF1 + β(1−Time) + γ(1−CO₂) + δ(1−FPR)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Sustainability Score", fontsize=11)
    ax.set_xlabel("Scenario", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sustainability_score_comparison.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved to {output_dir}/sustainability_score_comparison.png")


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def print_summary(df, scenarios=SCENARIOS):
    """Print a clean summary table."""
    print("\n" + "=" * 80)
    print("SUSTAINABILITY SCORE RESULTS")
    print("=" * 80)

    # Raw metrics
    print("\nRaw Metrics:")
    print(f"  {'Model':<15} {'F1':>8} {'Time(s)':>10} {'CO₂(µg)':>10} {'FPR':>12}")
    for _, row in df.iterrows():
        print(f"  {row['model']:<15} {row['f1_mean']:>8.4f} "
              f"{row['inference_time_mean']:>10.3f} "
              f"{row['emissions_mean']*1e6:>10.2f} "
              f"{row['fpr_mean']:>12.7f}")

    # Normalised metrics
    print("\nNormalised Metrics (higher = better):")
    print(f"  {'Model':<15} {'F1':>8} {'Speed':>8} {'Low CO₂':>8} {'Low FPR':>8}")
    for _, row in df.iterrows():
        print(f"  {row['model']:<15} {row['f1_norm']:>8.4f} "
              f"{row['time_norm']:>8.4f} "
              f"{row['emissions_norm']:>8.4f} "
              f"{row['fpr_norm']:>8.4f}")

    # SS scores
    scenario_names = list(scenarios.keys())
    print("\nSustainability Scores:")
    header = f"  {'Model':<15}"
    for s in scenario_names:
        header += f" {s:>16}"
    print(header)

    for _, row in df.iterrows():
        line = f"  {row['model']:<15}"
        for s in scenario_names:
            line += f" {row[s]:>16.4f}"
        print(line)

    # Winner per scenario
    print("\nWinner per Scenario:")
    for s in scenario_names:
        winner_idx = df[s].idxmax()
        winner = df.loc[winner_idx, "model"]
        score = df.loc[winner_idx, s]
        print(f"  {s:<20} → {winner} ({score:.4f})")

    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_sustainability_analysis(csv_path=os.path.join(_SCRIPT_DIR, "model_metrics.csv"),
                                 output_dir="figures"):
    """Run the full Sustainability Score analysis."""
    print("Loading inference results from:", csv_path)
    df = pd.read_csv(csv_path)

    # Compute scores
    df = compute_sustainability_scores(df)

    # Print summary
    print_summary(df)

    # Generate charts
    plot_sustainability_scores(df, output_dir=output_dir)

    # Save full results
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/sustainability_scores.csv", index=False)
    print("\n  Full results saved to results/sustainability_scores.csv")

    return df


if __name__ == "__main__":
    run_sustainability_analysis()