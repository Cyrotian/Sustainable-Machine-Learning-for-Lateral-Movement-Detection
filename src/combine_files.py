import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from config import Config

# Paths
print(f"Loading EoRS data from: {Config.LMD_2023_EORS_PATH}")
print(f"Loading Normal data from: {Config.LMD_2023_NORMAL_PATH}")

# Load CSVs
df_eors = pd.read_csv(Config.LMD_2023_EORS_PATH)
df_other = pd.read_csv(Config.LMD_2023_NORMAL_PATH)

# Target size = EoRS rows
target_rows = len(df_eors)

# Shuffle unique Computer IDs
rng = np.random.default_rng(Config.RANDOM_STATE)
computers = df_other["Computer"].unique()
rng.shuffle(computers)

# Select full Computer groups until we reach target size
selected_groups = []
current_rows = 0
for comp in computers:
    group = df_other[df_other["Computer"] == comp]
    selected_groups.append(group)
    current_rows += len(group)
    if current_rows >= target_rows:
        break

# Combine selected groups
df_other_grouped = pd.concat(selected_groups, ignore_index=True)

# Optional: trim excess rows (still group-consistent)
df_other_grouped = df_other_grouped.iloc[:target_rows]

# Combine with EoRS (balanced)
df_combined_balanced = pd.concat(
    [df_eors, df_other_grouped],
    ignore_index=True
)

# Save balanced dataset
df_combined_balanced.to_csv(Config.LMD_2023_TEST_PATH, index=False)
print(f"\n--- Balanced Dataset ---")
print(f"Saved to: {Config.LMD_2023_TEST_PATH}")
print(f"EoRS rows: {len(df_eors)}")
print(f"Normal rows (grouped): {len(df_other_grouped)}")
print(f"Total rows: {len(df_combined_balanced)}")

# ---- Full Normal Dataset (all normal + all EoRS) ----
df_combined_full = pd.concat(
    [df_eors, df_other],
    ignore_index=True
)

# Save full dataset
df_combined_full.to_csv(Config.LMD_2023_TEST_FULL_PATH, index=False)
print(f"\n--- Full Dataset ---")
print(f"Saved to: {Config.LMD_2023_TEST_FULL_PATH}")
print(f"EoRS rows: {len(df_eors)}")
print(f"Normal rows (all): {len(df_other)}")
print(f"Total rows: {len(df_combined_full)}")