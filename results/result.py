import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv('emissions/emissions.csv')

def classify(project_name):
    p = str(project_name).lower()
    if   'lstm'  in p: model = 'LSTM'
    elif 'rf'    in p: model = 'Random Forest'
    elif 'if'    in p: model = 'Isolation Forest'
    else:              model = 'Unknown'
    phase = 'Training' if 'training' in p else 'Inference'
    return model, phase

df[['model', 'phase']] = df['project_name'].apply(lambda x: pd.Series(classify(x)))
inference = df[df['phase'] == 'Inference'].copy()
training  = df[df['phase'] == 'Training'].copy()

inf_summary = inference.groupby('model').agg(
    emissions_mean   = ('emissions',       'mean'),
    emissions_std    = ('emissions',       'std'),
    duration_mean    = ('duration',        'mean'),
    duration_std     = ('duration',        'std'),
    energy_mean      = ('energy_consumed', 'mean'),
    energy_std       = ('energy_consumed', 'std'),
).reset_index()

MODEL_ORDER  = ['Random Forest', 'Isolation Forest', 'LSTM']
MODEL_COLORS = {'Random Forest': '#2196F3', 'Isolation Forest': '#FF9800', 'LSTM': '#4CAF50'}

inf_summary['model'] = pd.Categorical(inf_summary['model'], categories=MODEL_ORDER, ordered=True)
inf_summary = inf_summary.sort_values('model').reset_index(drop=True)
colors = [MODEL_COLORS[m] for m in inf_summary['model']]

os.makedirs('figures', exist_ok=True)

def style_ax(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=10)

def add_bar_labels(ax, bars, values, fmt, offset_frac=0.02):
    ymax = max(v for v in values if v == v)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * offset_frac,
                fmt.format(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

# CHART 1 – Inference CO2 Emissions
fig, ax = plt.subplots(figsize=(8, 5))
vals = inf_summary['emissions_mean'] * 1e6
errs = inf_summary['emissions_std']  * 1e6
bars = ax.bar(inf_summary['model'], vals, yerr=errs, color=colors, width=0.5,
              edgecolor='white', linewidth=1.2, capsize=6, error_kw={'linewidth': 2})
add_bar_labels(ax, bars, vals, '{:.2f}')
ax.set_title('Inference CO\u2082 Emissions by Model\n(Mean \u00b1 Std, 10 runs)', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('CO\u2082 Emissions (\u00b5g CO\u2082eq per inference)', fontsize=11)
ax.set_xlabel('Model', fontsize=11)
ax.set_ylim(0, vals.max() * 1.2)
style_ax(ax); plt.tight_layout()
plt.savefig('figures/chart1_inference_emissions.png', dpi=300, bbox_inches='tight'); plt.close()
print("✓ Chart 1: Inference Emissions")

# CHART 2 – Inference Time
fig, ax = plt.subplots(figsize=(8, 5))
vals = inf_summary['duration_mean']
errs = inf_summary['duration_std']
bars = ax.bar(inf_summary['model'], vals, yerr=errs, color=colors, width=0.5,
              edgecolor='white', linewidth=1.2, capsize=6, error_kw={'linewidth': 2})
add_bar_labels(ax, bars, vals, '{:.2f}s')
ax.set_title('Inference Time by Model\n(Mean \u00b1 Std, 10 runs)', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Inference Time (seconds)', fontsize=11)
ax.set_xlabel('Model', fontsize=11)
ax.set_ylim(0, vals.max() * 1.2)
style_ax(ax); plt.tight_layout()
plt.savefig('figures/chart2_inference_time.png', dpi=300, bbox_inches='tight'); plt.close()
print("✓ Chart 2: Inference Time")

# CHART 3 – Energy Consumed
fig, ax = plt.subplots(figsize=(8, 5))
vals = inf_summary['energy_mean'] * 1000
errs = inf_summary['energy_std']  * 1000
bars = ax.bar(inf_summary['model'], vals, yerr=errs, color=colors, width=0.5,
              edgecolor='white', linewidth=1.2, capsize=6, error_kw={'linewidth': 2})
add_bar_labels(ax, bars, vals, '{:.4f}')
ax.set_title('Energy Consumed During Inference by Model\n(Mean \u00b1 Std, 10 runs)', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Energy Consumed (Wh)', fontsize=11)
ax.set_xlabel('Model', fontsize=11)
ax.set_ylim(0, vals.max() * 1.2)
style_ax(ax); plt.tight_layout()
plt.savefig('figures/chart3_inference_energy.png', dpi=300, bbox_inches='tight'); plt.close()
print("✓ Chart 3: Inference Energy")

# CHART 4 – Scatter all 10 runs
fig, ax = plt.subplots(figsize=(9, 5))
np.random.seed(42)
for i, model in enumerate(MODEL_ORDER):
    runs = inference[inference['model'] == model]['emissions'].values * 1e6
    x_positions = np.full(len(runs), i) + np.random.uniform(-0.1, 0.1, len(runs))
    ax.scatter(x_positions, runs, color=MODEL_COLORS[model], alpha=0.7, s=60, zorder=3)
    ax.hlines(runs.mean(), i - 0.25, i + 0.25, color=MODEL_COLORS[model], linewidth=2.5, zorder=4)
ax.set_xticks(range(len(MODEL_ORDER)))
ax.set_xticklabels(MODEL_ORDER, fontsize=11)
ax.set_title('Inference CO\u2082 Emissions: All 10 Runs per Model', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('CO\u2082 Emissions (\u00b5g CO\u2082eq)', fontsize=11)
ax.set_xlabel('Model', fontsize=11)
style_ax(ax); plt.tight_layout()
plt.savefig('figures/chart4_emissions_scatter.png', dpi=300, bbox_inches='tight'); plt.close()
print("✓ Chart 4: Emissions Scatter")

# CHART 5 – Training vs Inference all models
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(MODEL_ORDER))
width = 0.35
train_vals = [training[training['model'] == m]['emissions'].values[0] * 1e6 for m in MODEL_ORDER]
inf_vals   = [inf_summary[inf_summary['model'] == m]['emissions_mean'].values[0] * 1e6 for m in MODEL_ORDER]
bars1 = ax.bar(x - width/2, train_vals, width, label='Training (one-off)',
               color=[MODEL_COLORS[m] for m in MODEL_ORDER], edgecolor='white', linewidth=1.2)
bars2 = ax.bar(x + width/2, inf_vals, width, label='Inference (mean per run)',
               color=[MODEL_COLORS[m] for m in MODEL_ORDER], edgecolor='white', linewidth=1.2, alpha=0.5)
for bar, val in zip(bars1, train_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(train_vals)*0.01,
            f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, val in zip(bars2, inf_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(train_vals)*0.01,
            f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER, fontsize=11)
ax.set_title('Training vs Inference CO\u2082 Emissions by Model', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('CO\u2082 Emissions (\u00b5g CO\u2082eq)', fontsize=11)
ax.set_xlabel('Model', fontsize=11)
ax.legend(fontsize=10)
style_ax(ax); plt.tight_layout()
plt.savefig('figures/chart5_training_vs_inference.png', dpi=300, bbox_inches='tight'); plt.close()
print("✓ Chart 5: Training vs Inference")

print("\n" + "="*72)
print("INFERENCE SUMMARY (mean ± std across 10 runs)")
print("="*72)
print(f"{'Model':<20} {'Emissions (µg CO2)':<28} {'Time (s)':<22} {'Energy (Wh)'}")
print("-"*72)
for _, row in inf_summary.iterrows():
    print(f"{row['model']:<20} "
          f"{row['emissions_mean']*1e6:.3f} ± {row['emissions_std']*1e6:.3f}        "
          f"{row['duration_mean']:.3f} ± {row['duration_std']:.3f}     "
          f"{row['energy_mean']*1000:.5f} ± {row['energy_std']*1000:.5f}")

print("\n" + "="*72)
print("TRAINING SUMMARY (single run)")
print("="*72)
print(f"{'Model':<20} {'Emissions (µg CO2)':<22} {'Time (s)':<15} {'Energy (Wh)'}")
print("-"*72)
for _, row in training.iterrows():
    print(f"{row['model']:<20} {row['emissions']*1e6:<22.3f} "
          f"{row['duration']:<15.3f} {row['energy_consumed']*1000:.5f}")

print("\n✓ All charts saved to figures/")