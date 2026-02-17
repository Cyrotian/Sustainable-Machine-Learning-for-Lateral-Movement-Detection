"""
data_loader.py — Load and clean the raw LMD-2022 dataset.

PURPOSE: Take the raw CSV and do minimal cleanup so the rest of the
         pipeline has consistent, valid data to work with.
"""

import pandas as pd


def load_lmd2022(csv_path):
    """
    Load LMD-2022 CSV, drop junk columns, rename label, filter valid rows.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # ── Step 1: Drop auto-generated index columns ──────────────────────
    # When CSVs are saved multiple times, pandas adds "Unnamed: 0" etc.
    # These are meaningless artifacts, so we remove them.
    useful_cols = []
    for col in df.columns:
        if not col.startswith("Unnamed"):
            useful_cols.append(col)

    df = df[useful_cols].copy()

    # ── Step 2: Standardise the label column name ──────────────────────
    # Raw data uses "Label", we rename to "isMalicious" for clarity.
    # 0 = normal (benign), 1 = malicious (lateral movement)
    df.rename(columns={"Label": "isMalicious"}, inplace=True)

    # ── Step 3: Keep only valid binary labels ──────────────────────────
    # Some rows might have missing or non-binary labels — drop those.
    valid_rows = (df["isMalicious"] == "1") | (df["isMalicious"] == "0")
    df = df[valid_rows].copy()

    # ── Summary ────────────────────────────────────────────────────────
    print("Rows after filtering:", len(df))
    print("Columns:", len(df.columns))
    print("\nLabel distribution:")
    print(df["isMalicious"].value_counts())

    return df
