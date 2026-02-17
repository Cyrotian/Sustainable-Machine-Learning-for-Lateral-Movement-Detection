"""
preprocessing_pipeline.py — Create all final datasets for model training/testing.

PURPOSE: Take raw LMD CSVs, extract the 5 selected features, apply minimal
         transforms, and save ready-to-use CSVs for each model.

         We already know which 5 features we want from feature selection,
         so there's no need to run the full drop_features + engineer_features
         pipeline. Just grab the columns, apply two transforms, done.

THE 5 DATASETS:
  1. isolation_forest_train.csv        = LMD-2022, normal-only (IF training)
  2. supervised_learning_dataset.csv   = LMD-2022, both classes (RF/LSTM training)
  3. lmd2023_test_dataset.csv          = LMD-2023, imbalanced ~92/8 (ALL models)
  4. lstm_train_sequenced.csv          = LMD-2022 + Computer & time (LSTM training)
  5. lstm_test_sequenced.csv           = LMD-2023 + Computer & time (LSTM testing)

TRANSFORMS APPLIED:
  - Image -> Image_Category (6 security-relevant buckets)
  - SourceImage rare category grouping (<50 occurrences -> "Other_Rare")
  - NaN fill with "0" for categoricals, 0 for numerics
  - The other 3 features (GrantedAccess, RuleName, Signed) pass through as-is
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from config import Config

LABEL_COL = "isMalicious"
GROUP_COL = "Computer"

# The 5 features selected by the feature selection pipeline
SELECTED_FEATURES = [
    "Image_Category",
    "GrantedAccess",
    "RuleName",
    "Signed",
    "SourceImage",
]

# ============================================================================
# FEATURE TRANSFORMS (only the two that matter)
# ============================================================================

ATTACK_TOOLS = ["psexec", "mimikatz", "procdump", "atomicredteam",
                "wmic", "powershell", "cmd.exe"]
INFRA_TOOLS = ["splunk", "streamfwd"]

RARE_THRESHOLD = 50


def categorise_image(path):
    """Categorise executable path into 6 security buckets."""
    s = str(path).lower()

    for tool in ATTACK_TOOLS:
        if tool in s:
            return "AttackTool"

    for tool in INFRA_TOOLS:
        if tool in s:
            return "Infra"

    if "\\windows\\system32\\" in s:
        return "System32"

    if "\\users\\" in s:
        return "UserProfile"

    if "\\temp\\" in s:
        return "Temp"

    return "Other"


def process_dataset(csv_path, selected_features, filter_label=None):
    """
    Load a raw CSV, extract selected features, apply transforms.

    Steps:
      1. Load CSV, clean up column names
      2. Create Image_Category from Image column (if needed)
      3. Extract the 5 selected features + label
      4. Group rare SourceImage values
      5. Fill NaN values with sensible defaults
      6. Optionally filter by label (normal-only for IF)
    """
    print("\nLoading:", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    # Drop Unnamed columns
    useful_cols = []
    for col in df.columns:
        if not col.startswith("Unnamed"):
            useful_cols.append(col)
    df = df[useful_cols].copy()

    # Standardise label column
    if "Label" in df.columns:
        df.rename(columns={"Label": LABEL_COL}, inplace=True)

    if LABEL_COL in df.columns:
        df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
        df = df[df[LABEL_COL].notna()].copy()
        df[LABEL_COL] = df[LABEL_COL].astype(int)

    print("  Raw rows:", len(df))

    # ── Image -> Image_Category ───────────────────────────────────────
    if "Image_Category" not in df.columns and "Image" in df.columns:
        df["Image_Category"] = df["Image"].apply(categorise_image)
        print("  Created Image_Category from Image column")

    # ── Select only the features we need ──────────────────────────────
    result = pd.DataFrame()
    for feat in selected_features:
        if feat in df.columns:
            result[feat] = df[feat].values
        else:
            print("  WARNING:", feat, "not found - filling with default")
            result[feat] = "0"

    result[LABEL_COL] = df[LABEL_COL].values

    # ── SourceImage: group rare categories ────────────────────────────
    if "SourceImage" in result.columns:
        counts = result["SourceImage"].value_counts()
        rare = counts[counts < RARE_THRESHOLD].index
        result["SourceImage"] = result["SourceImage"].where(
            ~result["SourceImage"].isin(rare), other="Other_Rare"
        )

    # ── Fill NaN with sensible defaults ───────────────────────────────
    for col in result.columns:
        if result[col].dtype == "object":
            result[col] = result[col].fillna("0")
        else:
            result[col] = result[col].fillna(0)

    # ── Filter by label if needed ─────────────────────────────────────
    if filter_label is not None:
        result = result[result[LABEL_COL] == filter_label].copy()
        print("  Filtered to label=" + str(filter_label) + ":", len(result), "rows")

    print("  Final:", len(result), "rows x", len(result.columns), "cols")

    return result


def process_dataset_with_sequence_cols(csv_path, selected_features, filter_label=None):
    """
    Same as process_dataset but also keeps Computer + SystemTime for LSTM.

    LSTM needs these to build temporal sliding windows:
      - Computer: group events by machine
      - SystemTime: sort events chronologically
    They are NOT input features — dropped before LSTM training.
    """
    print("\nLoading (with sequence cols):", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    # Drop Unnamed columns
    useful_cols = []
    for col in df.columns:
        if not col.startswith("Unnamed"):
            useful_cols.append(col)
    df = df[useful_cols].copy()

    # Standardise label column
    if "Label" in df.columns:
        df.rename(columns={"Label": LABEL_COL}, inplace=True)

    if LABEL_COL in df.columns:
        df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
        df = df[df[LABEL_COL].notna()].copy()
        df[LABEL_COL] = df[LABEL_COL].astype(int)

    # Save sequence columns before we trim
    has_computer = GROUP_COL in df.columns
    has_time = "SystemTime" in df.columns

    if has_computer:
        saved_computer = df[GROUP_COL].copy()
    if has_time:
        saved_time = df["SystemTime"].copy()

    # Image -> Image_Category
    if "Image_Category" not in df.columns and "Image" in df.columns:
        df["Image_Category"] = df["Image"].apply(categorise_image)

    # Select features
    result = pd.DataFrame()
    for feat in selected_features:
        if feat in df.columns:
            result[feat] = df[feat].values
        else:
            result[feat] = "0"

    result[LABEL_COL] = df[LABEL_COL].values

    # SourceImage rare grouping
    if "SourceImage" in result.columns:
        counts = result["SourceImage"].value_counts()
        rare = counts[counts < RARE_THRESHOLD].index
        result["SourceImage"] = result["SourceImage"].where(
            ~result["SourceImage"].isin(rare), other="Other_Rare"
        )

    # Fill NaN
    for col in result.columns:
        if result[col].dtype == "object":
            result[col] = result[col].fillna("0")
        else:
            result[col] = result[col].fillna(0)

    # Re-attach sequence columns
    if has_computer:
        result[GROUP_COL] = saved_computer.values[:len(result)]
    if has_time:
        result["SystemTime"] = saved_time.values[:len(result)]

    # Filter by label
    if filter_label is not None:
        result = result[result[LABEL_COL] == filter_label].copy()

    # Sort by Computer then time for LSTM windowing
    sort_cols = []
    if has_computer:
        sort_cols.append(GROUP_COL)
    if has_time:
        sort_cols.append("SystemTime")

    if len(sort_cols) > 0:
        result = result.sort_values(sort_cols)
        result = result.reset_index(drop=True)

    print("  Final:", len(result), "rows")

    return result


# ============================================================================
# MAIN: Create all datasets
# ============================================================================

def create_all_datasets(features_csv_path="src/05_final_selected_features.csv"):
    """
    Create all final datasets.

    Reads the selected features from the CSV produced by feature selection,
    or falls back to the hardcoded list.
    """
    if os.path.exists(features_csv_path):
        features_df = pd.read_csv(features_csv_path)
        final_features = features_df["Feature"].tolist()
    else:
        print("WARNING: features CSV not found, using hardcoded list")
        final_features = SELECTED_FEATURES

    output_folder = Path("final_datasets")
    output_folder.mkdir(exist_ok=True)

    print("=" * 80)
    print("CREATING ALL DATASETS")
    print("Selected features:", final_features)
    print("=" * 80)

    # Dataset 1: IF training (LMD-2022, normal-only)
    df = process_dataset(Config.LMD_2022_PATH, final_features, filter_label=0)
    df.to_csv(output_folder / "isolation_forest_train.csv", index=False)

    # Dataset 2: RF + LSTM training (LMD-2022, both classes)
    df = process_dataset(Config.LMD_2022_PATH, final_features, filter_label=None)
    df.to_csv(output_folder / "supervised_learning_dataset.csv", index=False)

    # Dataset 3: SHARED test set (LMD-2023, imbalanced binary, ALL models)
    df = process_dataset(Config.LMD_2023_TEST_PATH, final_features, filter_label=None)
    df.to_csv(output_folder / "lmd2023_test_dataset.csv", index=False)

    # Dataset 4: LSTM training (with sequence columns)
    df = process_dataset_with_sequence_cols(
        Config.LMD_2022_PATH, final_features, filter_label=None)
    df.to_csv(output_folder / "lstm_train_sequenced.csv", index=False)

    # Dataset 5: LSTM testing (with sequence columns)
    df = process_dataset_with_sequence_cols(
        Config.LMD_2023_TEST_PATH, final_features, filter_label=None)
    df.to_csv(output_folder / "lstm_test_sequenced.csv", index=False)

    print("\n" + "=" * 80)
    print("ALL DATASETS CREATED in", output_folder)
    print("=" * 80)


if __name__ == "__main__":
    create_all_datasets()
