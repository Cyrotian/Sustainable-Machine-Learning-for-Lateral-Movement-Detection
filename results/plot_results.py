# Generates ALL dissertation charts from two data sources:
#   1. emissions/emissions.csv      (from CodeCarbon — energy, time, CO2)
#   2. emissions/model_metrics.csv  (from save_model_metrics — F1, precision, recall)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# CONFIG

MODEL_ORDER = ["Random Forest", "LightGBM", "LSTM"]
MODEL_COLORS = {
    "Random Forest": "#2196F3",
    "LightGBM": "#FF9800",
    "LSTM": "#4CAF50",
}

os.makedirs("figures", exist_ok=True)


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)


def add_bar_labels(ax, bars, values, fmt, offset_frac=0.02):
    ymax = max(v for v in values if v == v)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * offset_frac,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


# LOAD EMISSIONS DATA

def classify(project_name):
    p = str(project_name).lower()
    if "lstm" in p:
        model = "LSTM"
    elif "lgbm" in p:
        model = "LightGBM"
    elif "rf" in p:
        model = "Random Forest"
    else:
        model = "Unknown"

    phase = "Training" if "training" in p else "Inference"
    return model, phase


print("Loading emissions/emissions.csv...")
if not os.path.exists("emissions/emissions.csv"):
    print("ERROR: emissions/emissions.csv not found. Run all models first.")
    sys.exit(1)

df = pd.read_csv("emissions/emissions.csv")
df[["model", "phase"]] = df["project_name"].apply(lambda x: pd.Series(classify(x)))
df = df[df["model"] != "Unknown"]

inference = df[df["phase"] == "Inference"].copy()
training = df[df["phase"] == "Training"].copy()

inf_summary = inference.groupby("model").agg(
    emissions_mean=("emissions", "mean"),
    emissions_std=("emissions", "std"),
    duration_mean=("duration", "mean"),
    duration_std=("duration", "std"),
    energy_mean=("energy_consumed", "mean"),
    energy_std=("energy_consumed", "std"),
).reset_index()

inf_summary["model"] = pd.Categorical(inf_summary["model"], categories=MODEL_ORDER, ordered=True)
inf_summary = inf_summary.sort_values("model").reset_index(drop=True)
colors = [MODEL_COLORS[m] for m in inf_summary["model"]]


# LOAD PERFORMANCE METRICS

print("Loading emissions/model_metrics.csv...")
has_perf = os.path.exists("emissions/model_metrics.csv")

if has_perf:
    perf = pd.read_csv("emissions/model_metrics.csv")
    perf["model"] = pd.Categorical(perf["model"], categories=MODEL_ORDER, ordered=True)
    perf = perf.sort_values("model").reset_index(drop=True)

    perf_dict = {}
    for _, row in perf.iterrows():
        perf_dict[row["model"]] = {
            "f1": row["f1_mean"],
            "f1_std": row.get("f1_std", 0),
            "precision": row["precision_mean"],
            "precision_std": row.get("precision_std", 0),
            "recall": row["recall_mean"],
            "recall_std": row.get("recall_std", 0),
            "accuracy": row.get("accuracy_mean", 0),
            "fpr": row.get("fpr_mean", 0),
            "fpr_std": row.get("fpr_std", 0),
        }
else:
    print("  WARNING: model_metrics.csv not found — skipping performance charts (7-11).")
    print("  Add save_model_metrics() calls to main.py and run all models to enable them.")


# ENERGY / COST CHARTS (from emissions.csv)

# CHART 1: Inference CO2 Emissions
fig, ax = plt.subplots(figsize=(8, 5))
vals = inf_summary["emissions_mean"] * 1e6
errs = inf_summary["emissions_std"] * 1e6
bars = ax.bar(
    inf_summary["model"],
    vals,
    yerr=errs,
    color=colors,
    width=0.5,
    edgecolor="white",
    linewidth=1.2,
    capsize=6,
    error_kw={"linewidth": 2},
)
add_bar_labels(ax, bars, vals, "{:.2f}")
ax.set_title("Inference CO\u2082 Emissions by Model\n(Mean \u00b1 Std, 10 runs)", fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("CO\u2082 Emissions (\u00b5g CO\u2082eq per inference)", fontsize=11)
ax.set_xlabel("Model", fontsize=11)
ax.set_ylim(0, vals.max() * 1.25)
style_ax(ax)
plt.tight_layout()
plt.savefig("figures/chart1_inference_emissions.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Chart 1: Inference Emissions")

# CHART 2: Inference Time
fig, ax = plt.subplots(figsize=(8, 5))
vals = inf_summary["duration_mean"]
errs = inf_summary["duration_std"]
bars = ax.bar(
    inf_summary["model"],
    vals,
    yerr=errs,
    color=colors,
    width=0.5,
    edgecolor="white",
    linewidth=1.2,
    capsize=6,
    error_kw={"linewidth": 2},
)
add_bar_labels(ax, bars, vals, "{:.2f}s")
ax.set_title("Inference Time by Model\n(Mean \u00b1 Std, 10 runs)", fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("Inference Time (seconds)", fontsize=11)
ax.set_xlabel("Model", fontsize=11)
ax.set_ylim(0, vals.max() * 1.25)
style_ax(ax)
plt.tight_layout()
plt.savefig("figures/chart2_inference_time.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Chart 2: Inference Time")

# CHART 3: Energy Consumed
fig, ax = plt.subplots(figsize=(8, 5))
vals = inf_summary["energy_mean"] * 1000
errs = inf_summary["energy_std"] * 1000
bars = ax.bar(
    inf_summary["model"],
    vals,
    yerr=errs,
    color=colors,
    width=0.5,
    edgecolor="white",
    linewidth=1.2,
    capsize=6,
    error_kw={"linewidth": 2},
)
add_bar_labels(ax, bars, vals, "{:.4f}")
ax.set_title("Energy Consumed During Inference by Model\n(Mean \u00b1 Std, 10 runs)", fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("Energy Consumed (Wh)", fontsize=11)
ax.set_xlabel("Model", fontsize=11)
ax.set_ylim(0, vals.max() * 1.25)
style_ax(ax)
plt.tight_layout()
plt.savefig("figures/chart3_inference_energy.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Chart 3: Inference Energy")

# CHART 4: Scatter all 10 runs
fig, ax = plt.subplots(figsize=(9, 5))
np.random.seed(42)

for i, model in enumerate(MODEL_ORDER):
    runs = inference[inference["model"] == model]["emissions"].values * 1e6
    if len(runs) == 0:
        continue

    x_positions = np.full(len(runs), i) + np.random.uniform(-0.1, 0.1, len(runs))
    ax.scatter(x_positions, runs, color=MODEL_COLORS[model], alpha=0.7, s=60, zorder=3)
    ax.hlines(runs.mean(), i - 0.25, i + 0.25, color=MODEL_COLORS[model], linewidth=2.5, zorder=4)

ax.set_xticks(range(len(MODEL_ORDER)))
ax.set_xticklabels(MODEL_ORDER, fontsize=11)
ax.set_title("Inference CO\u2082 Emissions: All 10 Runs per Model", fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("CO\u2082 Emissions (\u00b5g CO\u2082eq)", fontsize=11)
ax.set_xlabel("Model", fontsize=11)
style_ax(ax)
plt.tight_layout()
plt.savefig("figures/chart4_emissions_scatter.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Chart 4: Emissions Scatter")

# Pre-compute training/inference values for charts 5 & 6
train_vals = [training[training["model"] == m]["emissions"].values[0] * 1e6 for m in MODEL_ORDER]
inf_vals = [inf_summary[inf_summary["model"] == m]["emissions_mean"].values[0] * 1e6 for m in MODEL_ORDER]
ymax = max(max(train_vals), max(inf_vals))

# CHART 5: Training vs Inference
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(MODEL_ORDER))
width = 0.35

bars1 = ax.bar(
    x - width / 2,
    train_vals,
    width,
    label="Training (one-off)",
    color=[MODEL_COLORS[m] for m in MODEL_ORDER],
    edgecolor="white",
    linewidth=1.2,
)
bars2 = ax.bar(
    x + width / 2,
    inf_vals,
    width,
    label="Inference (mean per run)",
    color=[MODEL_COLORS[m] for m in MODEL_ORDER],
    edgecolor="white",
    linewidth=1.2,
    alpha=0.5,
)

for bar, val in zip(bars1, train_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.01, f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
for bar, val in zip(bars2, inf_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.01, f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(MODEL_ORDER, fontsize=11)
ax.set_title("Training vs Inference CO\u2082 Emissions by Model", fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("CO\u2082 Emissions (\u00b5g CO\u2082eq)", fontsize=11)
ax.set_xlabel("Model", fontsize=11)
ax.legend(fontsize=10)
style_ax(ax)
plt.tight_layout()
plt.savefig("figures/chart5_training_vs_inference.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Chart 5: Training vs Inference")

# CHART 6: Training vs Single Inference (hatched side-by-side)
fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(MODEL_ORDER))
width = 0.3

bars1 = ax.bar(
    x - width / 2,
    train_vals,
    width,
    label="Training (one-off)",
    color=[MODEL_COLORS[m] for m in MODEL_ORDER],
    edgecolor="white",
    linewidth=1.2,
)
bars2 = ax.bar(
    x + width / 2,
    inf_vals,
    width,
    label="Inference (single run)",
    color=[MODEL_COLORS[m] for m in MODEL_ORDER],
    edgecolor="white",
    linewidth=1.2,
    alpha=0.55,
    hatch="//",
)

for bar, val in zip(bars1, train_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.01, f"{val:.1f}\u00b5g", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, val in zip(bars2, inf_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.01, f"{val:.1f}\u00b5g", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(MODEL_ORDER, fontsize=11)
ax.set_ylabel("CO\u2082 Emissions (\u00b5g CO\u2082eq)", fontsize=11)
ax.set_xlabel("Model", fontsize=11)
ax.set_title("Training vs Single Inference CO\u2082 Emissions by Model", fontsize=13, fontweight="bold", pad=15)
ax.set_ylim(0, ymax * 1.2)
ax.legend(fontsize=10)
style_ax(ax)
plt.tight_layout()
plt.savefig("figures/chart6_training_vs_inference_sidebyside.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Chart 6: Training vs Single Inference (side-by-side)")


# PERFORMANCE CHARTS (from model_metrics.csv)

if has_perf:
    # CHART 7: F1 / Precision / Recall Grouped Bar
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODEL_ORDER))
    width = 0.22

    metric_list = [
        ("f1", "f1_std", "F1-Score", "#1565C0"),
        ("precision", "precision_std", "Precision", "#2E7D32"),
        ("recall", "recall_std", "Recall", "#E65100"),
    ]

    for i, (key, std_key, label, color) in enumerate(metric_list):
        vals = [perf_dict[m][key] for m in MODEL_ORDER]
        errs = [perf_dict[m][std_key] for m in MODEL_ORDER]
        offset = (i - 1) * width

        bars = ax.bar(
            x + offset,
            vals,
            width,
            yerr=errs,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            capsize=4,
            error_kw={"linewidth": 1.5},
        )

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Detection Performance by Model on LMD-2023", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10, loc="lower right")
    style_ax(ax)
    plt.tight_layout()
    plt.savefig("figures/chart7_performance_grouped.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Chart 7: Performance Grouped Bar")

    # CHART 8: F1 Standalone
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [perf_dict[m]["f1"] for m in MODEL_ORDER]
    errs = [perf_dict[m]["f1_std"] for m in MODEL_ORDER]
    bars = ax.bar(
        MODEL_ORDER,
        vals,
        yerr=errs,
        color=colors,
        width=0.5,
        edgecolor="white",
        linewidth=1.2,
        capsize=6,
        error_kw={"linewidth": 2},
    )

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("F1-Score by Model on LMD-2023\n(Mean \u00b1 Std, 10 inference runs)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.4, label="0.95 threshold")
    ax.legend(fontsize=10)
    style_ax(ax)
    plt.tight_layout()
    plt.savefig("figures/chart8_f1_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Chart 8: F1 Comparison")

    # CHART 9: Precision vs Recall Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for model in MODEL_ORDER:
        m = perf_dict[model]
        ax.scatter(m["recall"], m["precision"], color=MODEL_COLORS[model], s=250, zorder=5, edgecolors="black", linewidth=1)
        ax.errorbar(m["recall"], m["precision"], xerr=m["recall_std"], yerr=m["precision_std"], color=MODEL_COLORS[model], fmt="none", capsize=4, linewidth=1.5)
        ax.annotate(model, (m["recall"], m["precision"]), textcoords="offset points", xytext=(12, 8), fontsize=11, fontweight="bold", color=MODEL_COLORS[model])

    ax.set_xlabel("Recall (sensitivity to attacks)", fontsize=11)
    ax.set_ylabel("Precision (alert accuracy)", fontsize=11)
    ax.set_title("Precision vs Recall Tradeoff by Model", fontsize=13, fontweight="bold", pad=15)
    style_ax(ax)
    plt.tight_layout()
    plt.savefig("figures/chart9_precision_vs_recall.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Chart 9: Precision vs Recall")

    # CHART 10: Recall Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [perf_dict[m]["recall"] for m in MODEL_ORDER]
    errs = [perf_dict[m]["recall_std"] for m in MODEL_ORDER]
    bars = ax.bar(
        MODEL_ORDER,
        vals,
        yerr=errs,
        color=colors,
        width=0.5,
        edgecolor="white",
        linewidth=1.2,
        capsize=6,
        error_kw={"linewidth": 2},
    )

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Recall by Model on LMD-2023\n(Proportion of attacks correctly detected)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    min_val = min(vals) - 0.05
    max_val = max(vals) + 0.05
    ax.set_ylim(max(0, min_val), min(1.0, max_val))
    ax.axhline(y=0.75, color="gray", linestyle=":", alpha=0.5, label="0.75 baseline")
    ax.legend(fontsize=10)
    style_ax(ax)
    plt.tight_layout()
    plt.savefig("figures/chart10_recall_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Chart 10: Recall Comparison")

    # CHART 11: Combined Overview (Performance + Cost)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # Panel 1: F1
    ax = axes[0]
    vals = [perf_dict[m]["f1"] for m in MODEL_ORDER]
    bars = ax.bar(MODEL_ORDER, vals, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_title("F1-Score", fontsize=11, fontweight="bold")
    ax.set_ylim(min(vals) - 0.02, max(vals) + 0.02)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    style_ax(ax)

    # Panel 2: Precision
    ax = axes[1]
    vals = [perf_dict[m]["precision"] for m in MODEL_ORDER]
    bars = ax.bar(MODEL_ORDER, vals, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002, f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_title("Precision", fontsize=11, fontweight="bold")
    ax.set_ylim(0.990, 1.003)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    style_ax(ax)

    # Panel 3: Inference Time
    ax = axes[2]
    vals = [inf_summary[inf_summary["model"] == m]["duration_mean"].values[0] for m in MODEL_ORDER]
    bars = ax.bar(MODEL_ORDER, vals, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02, f"{val:.2f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_title("Inference Time (s)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    style_ax(ax)

    # Panel 4: Emissions
    ax = axes[3]
    vals = [inf_summary[inf_summary["model"] == m]["emissions_mean"].values[0] * 1e6 for m in MODEL_ORDER]
    bars = ax.bar(MODEL_ORDER, vals, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02, f"{val:.1f}\u00b5g", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_title("CO\u2082 Emissions (\u00b5g)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    style_ax(ax)

    fig.suptitle("Model Comparison: Performance vs Environmental Cost", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/chart11_combined_overview.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Chart 11: Combined Overview")

    # CHART 12: False Positive Rate (FPR)
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [perf_dict[m]["fpr"] for m in MODEL_ORDER]
    errs = [perf_dict[m]["fpr_std"] for m in MODEL_ORDER]
    bars = ax.bar(
        MODEL_ORDER,
        vals,
        yerr=errs,
        color=colors,
        width=0.5,
        edgecolor="white",
        linewidth=1.2,
        capsize=6,
        error_kw={"linewidth": 2},
    )

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.03, f"{val:.6f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("False Positive Rate by Model on LMD-2023\n(Proportion of benign traffic misclassified as attacks)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("False Positive Rate", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.35 if max(vals) > 0 else 0.01)
    style_ax(ax)
    plt.tight_layout()
    plt.savefig("figures/chart12_fpr_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Chart 12: FPR Comparison")


# SUMMARY TABLES

print("\n" + "=" * 90)
print("INFERENCE SUMMARY (mean \u00b1 std across 10 runs)")
print("=" * 90)
print(f"{'Model':<20} {'Emissions (\u00b5g CO2)':<28} {'Time (s)':<22} {'Energy (Wh)'}")
print("-" * 90)

for _, row in inf_summary.iterrows():
    print(
        f"{row['model']:<20} "
        f"{row['emissions_mean']*1e6:.3f} \u00b1 {row['emissions_std']*1e6:.3f}        "
        f"{row['duration_mean']:.3f} \u00b1 {row['duration_std']:.3f}     "
        f"{row['energy_mean']*1000:.5f} \u00b1 {row['energy_std']*1000:.5f}"
    )

print("\n" + "=" * 90)
print("TRAINING SUMMARY (single run)")
print("=" * 90)
print(f"{'Model':<20} {'Emissions (\u00b5g CO2)':<22} {'Time (s)':<15} {'Energy (Wh)'}")
print("-" * 90)

for _, row in training.iterrows():
    if row["model"] in MODEL_ORDER:
        print(f"{row['model']:<20} {row['emissions']*1e6:<22.3f} {row['duration']:<15.3f} {row['energy_consumed']*1000:.5f}")

if has_perf:
    print("\n" + "=" * 90)
    print("PERFORMANCE SUMMARY (mean \u00b1 std across 10 inference runs)")
    print("=" * 90)
    print(f"{'Model':<20} {'F1':<22} {'Precision':<22} {'Recall':<22} {'FPR'}")
    print("-" * 110)

    for model in MODEL_ORDER:
        m = perf_dict[model]
        print(
            f"{model:<20} "
            f"{m['f1']:.4f} \u00b1 {m['f1_std']:.4f}     "
            f"{m['precision']:.4f} \u00b1 {m['precision_std']:.4f}     "
            f"{m['recall']:.4f} \u00b1 {m['recall_std']:.4f}     "
            f"{m['fpr']:.4f} \u00b1 {m['fpr_std']:.4f}"
        )

chart_count = 12 if has_perf else 6
print(f"\nAll {chart_count} charts saved to figures/")