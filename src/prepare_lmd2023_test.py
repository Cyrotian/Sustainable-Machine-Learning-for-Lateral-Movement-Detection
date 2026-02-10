"""
prepare_lmd2023_test.py

Combine LMD-2023_EORs (malicious) and LMD-2023_NORMAL (benign) into a single
balanced 50/50 test CSV with the same column conventions as the LMD-2022 loader.

The Normal class is down-sampled **stratified by Computer** so that every
machine keeps its proportional representation in the final test set.

Usage:
    python prepare_lmd2023_test.py

Outputs:
    data/lmd2023_test_balanced.csv
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from config import Config


# Paths
print(f"Loading EoRS data from: {Config.LMD_2023_EORS_PATH}")
print(f"Loading Normal data from: {Config.LMD_2023_NOMAL_PATH}")

# ============================================================================
# CONFIGURATION
# ============================================================================
EORS_PATH = Config.LMD_2023_EORS_PATH
NORMAL_PATH = Config.LMD_2023_NOMAL_PATH
OUTPUT_PATH = Config.LMD_2023_TEST_PATH

RANDOM_STATE = 42
GROUP_COL = "Computer"


def load_and_tag(csv_path: str, label_value: str) -> pd.DataFrame:
    """
    Load a CSV, drop 'Unnamed' columns, rename Label → isMalicious,
    and ensure every row has the given label value.
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Drop index artifacts
    useful_cols = [c for c in df.columns if not c.startswith("Unnamed")]
    df = df[useful_cols].copy()

    # If a Label column exists, rename it; otherwise create it
    if "Label" in df.columns:
        df.rename(columns={"Label": "isMalicious"}, inplace=True)
    elif "isMalicious" not in df.columns:
        df["isMalicious"] = label_value

    # Force the label to the expected value (safety check)
    df["isMalicious"] = label_value

    print(f"  → {len(df):,} rows, {len(df.columns)} columns")
    return df


def stratified_downsample(df: pd.DataFrame, n_target: int,
                          group_col: str = GROUP_COL,
                          random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Down-sample *df* to *n_target* rows, preserving the proportional
    representation of each group (Computer).

    Each computer gets floor(proportion * n_target) rows, with any
    remainder distributed round-robin to the largest groups.
    """
    counts = df[group_col].value_counts()
    total = len(df)

    # Proportional allocation per computer
    alloc = (counts / total * n_target).apply(np.floor).astype(int)

    # Distribute leftover rows to the largest groups
    shortfall = n_target - alloc.sum()
    if shortfall > 0:
        top_groups = alloc.index[:shortfall]
        alloc.loc[top_groups] += 1

    sampled = []
    for group_name, n in alloc.items():
        group_df = df[df[group_col] == group_name]
        n = min(n, len(group_df))  # can't sample more than available
        sampled.append(group_df.sample(n=n, random_state=random_state))

    result = pd.concat(sampled, ignore_index=True)
    return result


def prepare_lmd2023_test(
    eors_path: str = EORS_PATH,
    normal_path: str = NORMAL_PATH,
    output_path: str = OUTPUT_PATH,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Load both LMD-2023 CSVs, down-sample the larger class (stratified by
    Computer) to match the smaller class (50/50 balance), combine, shuffle,
    and save.

    Returns the balanced DataFrame.
    """

    df_mal = load_and_tag(eors_path, label_value="1")
    df_nor = load_and_tag(normal_path, label_value="0")


    shared_cols = sorted(set(df_mal.columns) & set(df_nor.columns))
    if set(df_mal.columns) != set(df_nor.columns):
        only_mal = set(df_mal.columns) - set(df_nor.columns)
        only_nor = set(df_nor.columns) - set(df_mal.columns)
        if only_mal:
            print(f"\n⚠ Columns only in EORs (dropped): {sorted(only_mal)}")
        if only_nor:
            print(f"⚠ Columns only in NORMAL (dropped): {sorted(only_nor)}")
        df_mal = df_mal[shared_cols]
        df_nor = df_nor[shared_cols]


    n_mal = len(df_mal)
    n_nor = len(df_nor)
    n_target = min(n_mal, n_nor)

    print(f"\nEORs (malicious): {n_mal:,}")
    print(f"Normal (benign):  {n_nor:,}")
    print(f"Target per class: {n_target:,}  (50/50 balance)")

    if n_mal > n_target:
        print(f"\nDown-sampling EORs stratified by {GROUP_COL}...")
        df_mal = stratified_downsample(df_mal, n_target, random_state=random_state)
    if n_nor > n_target:
        print(f"\nDown-sampling Normal stratified by {GROUP_COL}...")
        df_nor = stratified_downsample(df_nor, n_target, random_state=random_state)

    # ── Show per-computer breakdown after balancing ────────────────────────
    print(f"\nComputer distribution after balancing:")
    combined_computers = pd.concat([
        df_mal[GROUP_COL].value_counts().rename("Malicious"),
        df_nor[GROUP_COL].value_counts().rename("Normal"),
    ], axis=1).fillna(0).astype(int)
    combined_computers["Total"] = combined_computers.sum(axis=1)
    print(combined_computers.sort_values("Total", ascending=False).to_string())

    # ── Combine and shuffle ────────────────────────────────────────────────
    df_test = pd.concat([df_mal, df_nor], ignore_index=True)
    df_test = df_test.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # ── Convert isMalicious to consistent type ─────────────────────────────
    df_test["isMalicious"] = df_test["isMalicious"].astype(str)

    print(f"\nFinal test set: {len(df_test):,} rows")
    print(f"Label distribution:")
    print(df_test["isMalicious"].value_counts())
    print(df_test["isMalicious"].value_counts(normalize=True).mul(100).round(1))

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_test.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")

    return df_test


if __name__ == "__main__":
    prepare_lmd2023_test()