"""
combine_files.py — Combine LMD-2023 EORs + Normal into test sets.

PURPOSE: An alternative/earlier approach to creating the LMD-2023 test set.
         Creates TWO combined datasets:
           1. Balanced: Normal downsampled to match EORs (for RF testing)
           2. Full: All Normal + all EORs (for Isolation Forest testing)

HOW THE BALANCED VERSION WORKS:
  Instead of stratified per-computer downsampling (like prepare_lmd2023_test.py),
  this simpler approach selects WHOLE computers until we have enough rows.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('.'))
from config import Config

# Load both LMD-2023 files
print("Loading EoRS (malicious) data...")
df_eors = pd.read_csv(Config.LMD_2023_EORS_PATH)

print("Loading Normal data...")
df_other = pd.read_csv(Config.LMD_2023_NORMAL_PATH)

# We want the same number of normal rows as malicious rows
target_rows = len(df_eors)

# ── Build balanced set by selecting whole computers ────────────────────
# Get all unique computer names from the normal data
computers = df_other["Computer"].unique()

# Shuffle them randomly
rng = np.random.default_rng(Config.RANDOM_STATE)
rng.shuffle(computers)

# Add whole computers one at a time until we have enough rows
selected_groups = []
current_rows = 0

for comp in computers:
    # Get all rows belonging to this computer
    computer_data = df_other[df_other["Computer"] == comp]
    selected_groups.append(computer_data)
    current_rows = current_rows + len(computer_data)

    # Stop once we have enough
    if current_rows >= target_rows:
        break

# Combine the selected computer groups and trim to exact target
df_other_balanced = pd.concat(selected_groups, ignore_index=True)
df_other_balanced = df_other_balanced.iloc[:target_rows]

# ── Create balanced dataset (50/50 for RF testing) ────────────────────
df_balanced = pd.concat([df_eors, df_other_balanced], ignore_index=True)
df_balanced.to_csv(Config.LMD_2023_TEST_PATH, index=False)

print("\nBalanced dataset:")
print("  EoRS rows:  ", len(df_eors))
print("  Normal rows: ", len(df_other_balanced))
print("  Total:       ", len(df_balanced))
print("  Saved to:    ", Config.LMD_2023_TEST_PATH)

# ── Create full dataset (all data, imbalanced, for IF testing) ────────
df_full = pd.concat([df_eors, df_other], ignore_index=True)
df_full.to_csv(Config.LMD_2023_TEST_FULL_PATH, index=False)

print("\nFull dataset:")
print("  EoRS rows:  ", len(df_eors))
print("  Normal rows: ", len(df_other))
print("  Total:       ", len(df_full))
print("  Saved to:    ", Config.LMD_2023_TEST_FULL_PATH)
