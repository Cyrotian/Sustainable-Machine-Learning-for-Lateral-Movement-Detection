"""
prepare_lmd2023_test.py — Create the LMD-2023 test set (imbalanced, binary).

PURPOSE: LMD-2023 comes as a single CSV with 3 classes:
           Label 0 = Normal       (~1.6M rows, 92%)
           Label 1 = EoRS         (~110K rows,  6%)
           Label 2 = EoHT         (~30K rows,   2%)

         We convert to BINARY labels:
           0 = Normal
           1 = Malicious (EoRS + EoHT combined)

         The dataset is kept IMBALANCED (~92% normal, ~8% malicious).

WHY IMBALANCED?
  Isolation Forest is an anomaly detector — it needs rare anomalies
  to flag. A 50/50 balanced set breaks IF because nothing looks "rare".

  For a fair 3-model comparison, ALL models (RF, IF, LSTM) use this
  same imbalanced test set. RF and LSTM handle it fine because:
    - RF uses class_weight="balanced" during training
    - F1-macro weights both classes equally regardless of size

OUTPUT: A single test CSV used by all 3 models.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from config import Config

RANDOM_STATE = 42


def prepare_lmd2023_test():
    """
    Load the combined LMD-2023 CSV, convert 3 classes to binary, save.
    """
    print("=" * 80)
    print("PREPARING LMD-2023 TEST SET")
    print("=" * 80)

    # Load the combined CSV (all 3 classes)
    input_path = Config.LMD_2023_COMBINED_PATH
    print("\nLoading:", input_path)
    df = pd.read_csv(input_path, low_memory=False)

    # Drop auto-generated index columns
    useful_cols = []
    for col in df.columns:
        if not col.startswith("Unnamed"):
            useful_cols.append(col)
    df = df[useful_cols].copy()

    # Standardise label column name
    if "Label" in df.columns:
        df.rename(columns={"Label": "isMalicious"}, inplace=True)

    print("Total rows:", len(df))

    # Show original 3-class distribution
    print("\nOriginal labels:")
    counts = df["isMalicious"].value_counts().sort_index()
    for label_val in counts.index:
        count = counts[label_val]
        pct = round(100 * count / len(df), 1)
        print("  Label", label_val, ":", count, "(" + str(pct) + "%)")

    # Convert to binary: 0 stays 0, everything else becomes 1
    def to_binary(x):
        if int(x) == 0:
            return 0
        else:
            return 1

    df["isMalicious"] = df["isMalicious"].apply(to_binary)

    # Show binary distribution
    print("\nBinary labels:")
    binary_counts = df["isMalicious"].value_counts().sort_index()
    for label_val in binary_counts.index:
        count = binary_counts[label_val]
        pct = round(100 * count / len(df), 1)
        if label_val == 0:
            label_name = "Normal"
        else:
            label_name = "Malicious"
        print("  " + label_name + " (" + str(label_val) + "):", count, "(" + str(pct) + "%)")

    # Shuffle rows
    df = df.sample(frac=1, random_state=RANDOM_STATE)
    df = df.reset_index(drop=True)

    # Save
    output_path = Config.LMD_2023_TEST_PATH
    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nSaved:", output_path)
    print("Rows:", len(df))

    return df


if __name__ == "__main__":
    prepare_lmd2023_test()
