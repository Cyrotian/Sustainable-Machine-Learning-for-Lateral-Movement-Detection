"""
plot_complexity_sweep.py — Generate dual-axis charts from complexity sweep results.

PURPOSE: Visualise how F1 and CO₂ emissions change as model complexity increases.
         One chart per model with F1 on left axis, emissions on right axis.
         Plus a summary chart showing all models together.

INPUT:   results/complexity_sweep.csv (from complexity_sweep.py)
OUTPUT:  figures/complexity_rf.png
         figures/complexity_lgbm.png
         figures/complexity_lstm.png
         figures/complexity_summary.png

USAGE:   python plot_complexity_sweep.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

PARAM_LABELS = {
    "n_estimators": "Number of Estimators",
    "hidden_units": "LSTM Hidden Units",
    "epochs": "Training Epochs",
}


def style_ax(ax):
    """Apply consistent styling."""
    ax.spines["top"].set_visible(False)
    ax.tick_params(labelsize=10)


# ============================================================================
# INDIVIDUAL MODEL CHARTS (DUAL AXIS)
# ============================================================================

def plot_model_dual_axis(df_model, model_name, output_dir):
    """
    Plot F1 and emissions vs complexity parameter on dual y-axes.

    Left axis:  F1-Score (detection performance)
    Right axis: CO₂ emissions (environmental cost)

    The key insight is where F1 plateaus while emissions keep rising.
    """
    param_name = df_model["parameter_name"].iloc[0]
    x = df_model["parameter_value"].values
    f1 = df_model["f1"].values
    # Use training emissions for the complexity curve (shows cost of complexity)
    train_co2 = df_model["train_emissions"].values * 1e6  # convert to µg
    infer_co2 = df_model["inference_emissions"].values * 1e6

    color_main = MODEL_COLORS.get(model_name, "#888888")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # ── Left axis: F1-Score ──────────────────────────────────────────
    ax1.plot(x, f1, "o-", color=color_main, linewidth=2.5, markersize=8,
             label="F1-Score", zorder=5)
    ax1.set_xlabel(PARAM_LABELS.get(param_name, param_name), fontsize=12)
    ax1.set_ylabel("F1-Score", fontsize=12, color=color_main)
    ax1.tick_params(axis="y", labelcolor=color_main)
    ax1.set_ylim(min(f1) * 0.995, min(max(f1) * 1.002, 1.0))
    style_ax(ax1)

    # ── Right axis: Emissions ────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.plot(x, train_co2, "s--", color="#DC2626", linewidth=2, markersize=7,
             label="Train CO₂", alpha=0.8)
    ax2.plot(x, infer_co2, "^--", color="#9333EA", linewidth=2, markersize=7,
             label="Inference CO₂", alpha=0.8)
    ax2.set_ylabel("CO₂ Emissions (µg)", fontsize=12, color="#DC2626")
    ax2.tick_params(axis="y", labelcolor="#DC2626")
    style_ax(ax2)

    # ── Find sustainability-optimal point ────────────────────────────
    # Point where F1 reaches 99% of its peak
    peak_f1 = max(f1)
    threshold_f1 = peak_f1 * 0.99
    optimal_idx = None
    for i, val in enumerate(f1):
        if val >= threshold_f1:
            optimal_idx = i
            break

    if optimal_idx is not None:
        ax1.axvline(x=x[optimal_idx], color="gray", linestyle=":", alpha=0.6, linewidth=1.5)
        ax1.annotate(
            f"99% peak F1\n{param_name}={x[optimal_idx]}",
            xy=(x[optimal_idx], f1[optimal_idx]),
            xytext=(x[optimal_idx] + (max(x) - min(x)) * 0.1, f1[optimal_idx] - (max(f1) - min(f1)) * 0.3),
            fontsize=10, fontweight="bold", color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
        )

    # ── Title and legend ─────────────────────────────────────────────
    ax1.set_title(f"{model_name}: Detection Performance vs Environmental Cost\n"
                  f"Varying {PARAM_LABELS.get(param_name, param_name)}",
                  fontsize=13, fontweight="bold", pad=15)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    ax1.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    filename = model_name.lower().replace(" ", "_")
    filepath = os.path.join(output_dir, f"complexity_{filename}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")

    return optimal_idx


# ============================================================================
# SUMMARY CHART (ALL MODELS)
# ============================================================================

def plot_summary(df, output_dir):
    """
    Plot normalised F1 vs normalised total emissions for all models.
    Each point is one complexity configuration.
    Shows the Pareto-like frontier across all configurations.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for model_name, group in df.groupby("model"):
        color = MODEL_COLORS.get(model_name, "#888888")
        total_co2 = (group["train_emissions"] + group["inference_emissions"]) * 1e6

        ax.scatter(total_co2, group["f1"], color=color, s=80, alpha=0.7,
                   edgecolors="white", linewidth=0.8, label=model_name, zorder=5)

        # Connect points in order
        ax.plot(total_co2, group["f1"], "-", color=color, alpha=0.3, linewidth=1.5)

        # Label first and last points
        param = group["parameter_value"].values
        ax.annotate(f"{param[0]}", (total_co2.iloc[0], group["f1"].iloc[0]),
                    fontsize=8, color=color, fontweight="bold",
                    textcoords="offset points", xytext=(-15, -12))
        ax.annotate(f"{param[-1]}", (total_co2.iloc[-1], group["f1"].iloc[-1]),
                    fontsize=8, color=color, fontweight="bold",
                    textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel("Total CO₂ Emissions (µg) — Training + Inference", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title("Detection Performance vs Environmental Cost\n"
                 "Each point = one complexity configuration",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.2)
    style_ax(ax)

    fig.tight_layout()
    filepath = os.path.join(output_dir, "complexity_summary.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


# ============================================================================
# HEURISTICS TABLE
# ============================================================================

def compute_heuristics(df):
    """
    For each model, find the sustainability-optimal configuration:
    the lowest complexity that achieves >= 99% of peak F1.

    Prints a summary table with:
    - Peak configuration (max F1)
    - Optimal configuration (99% F1 at minimum cost)
    - Emissions saved (%)
    - F1 lost (%)
    """
    print("\n" + "=" * 80)
    print("SUSTAINABILITY HEURISTICS")
    print("=" * 80)

    rows = []

    for model_name, group in df.groupby("model"):
        group = group.sort_values("parameter_value").reset_index(drop=True)

        peak_idx = group["f1"].idxmax()
        peak_f1 = group.loc[peak_idx, "f1"]
        peak_param = group.loc[peak_idx, "parameter_value"]
        peak_train_co2 = group.loc[peak_idx, "train_emissions"]
        peak_infer_co2 = group.loc[peak_idx, "inference_emissions"]
        peak_total_co2 = peak_train_co2 + peak_infer_co2

        # Find first config that reaches 99% of peak F1
        threshold = peak_f1 * 0.99
        optimal = group[group["f1"] >= threshold].iloc[0]
        opt_param = optimal["parameter_value"]
        opt_f1 = optimal["f1"]
        opt_total_co2 = optimal["train_emissions"] + optimal["inference_emissions"]

        f1_loss = (peak_f1 - opt_f1) / peak_f1 * 100
        co2_saved = (peak_total_co2 - opt_total_co2) / peak_total_co2 * 100

        param_name = group["parameter_name"].iloc[0]

        print(f"\n  {model_name}:")
        print(f"    Peak:    {param_name}={int(peak_param):>4} → F1={peak_f1:.4f}, "
              f"CO₂={peak_total_co2*1e6:.1f}µg")
        print(f"    Optimal: {param_name}={int(opt_param):>4} → F1={opt_f1:.4f}, "
              f"CO₂={opt_total_co2*1e6:.1f}µg")
        print(f"    Saving:  {co2_saved:.1f}% less CO₂ for {f1_loss:.2f}% F1 loss")

        rows.append({
            "model": model_name,
            "parameter": param_name,
            "peak_value": int(peak_param),
            "peak_f1": peak_f1,
            "peak_co2_ug": peak_total_co2 * 1e6,
            "optimal_value": int(opt_param),
            "optimal_f1": opt_f1,
            "optimal_co2_ug": opt_total_co2 * 1e6,
            "f1_loss_pct": f1_loss,
            "co2_saved_pct": co2_saved,
        })

    print("=" * 80)

    # Save heuristics table
    heuristics_df = pd.DataFrame(rows)
    heuristics_path = "results/complexity_heuristics.csv"
    heuristics_df.to_csv(heuristics_path, index=False)
    print(f"\nHeuristics saved to {heuristics_path}")

    return heuristics_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading sweep results...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df)} configurations loaded")

    # Individual model charts
    print("\nGenerating individual model charts...")
    for model_name in df["model"].unique():
        df_model = df[df["model"] == model_name].sort_values("parameter_value")
        plot_model_dual_axis(df_model, model_name, OUTPUT_DIR)

    # Summary chart
    print("\nGenerating summary chart...")
    plot_summary(df, OUTPUT_DIR)

    # Heuristics
    compute_heuristics(df)


if __name__ == "__main__":
    main()