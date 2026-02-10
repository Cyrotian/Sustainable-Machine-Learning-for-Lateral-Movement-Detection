# ============================================================================
# CREATE ALL 3 FINAL DATASETS
# ============================================================================
import os
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, os.path.abspath("."))
from config import Config



SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

SUPERVISED_DATASET_PATH = Config.LMD_2022_PATH
ISOLATION_FOREST_DATASET_PATH = Config.LMD_2022_NORMAL_PATH
TESTING_DATASET_PATH = Config.LMD_2023_TEST_PATH


from feature_engineering import (
    drop_features,
    engineer_features,
    LABEL_COL
)

def process_and_save_dataset(
    df_path: str,
    output_name: str,
    selected_features: list[str],
    filter_label: int = None
):
    """Process a single dataset and save with selected features."""
    
    # Load
    df = pd.read_csv(df_path, low_memory=False)
    if 'Label' in df.columns:
        df = df.rename(columns={'Label': LABEL_COL})
    
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors='coerce')
    df = df[df[LABEL_COL].notna()].copy()

    print(f"\nProcessing: {output_name}")
    print(f"  Loaded: {len(df):,} rows")
    
    # Apply preprocessing (your existing functions!)
    df_clean, _, _ = drop_features(df)
    X, y, groups = engineer_features(df_clean)
    
    # Extract selected features
    available = [f for f in selected_features if f in X.columns]
    missing = [f for f in selected_features if f not in X.columns]
    
    if missing:
        print(f"  ⚠ Missing features: {missing}")
    
    X_selected = X[available].copy()
    X_selected[LABEL_COL] = y.values
    
    # Filter by label if needed
    if filter_label is not None:
        X_selected = X_selected[X_selected[LABEL_COL] == filter_label].copy()
        print(f"  Filtered to label={filter_label}: {len(X_selected):,} rows")
    
    # Save
    output_path = Path("final_datasets")
    output_path.mkdir(exist_ok=True)
    output_file = output_path / output_name
    X_selected.to_csv(output_file, index=False)
    
    print(f"  ✓ Saved: {output_file}")
    print(f"  Features: {available}")
    print(f"  Label distribution: {X_selected[LABEL_COL].value_counts().to_dict()}")
    print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    return X_selected


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Load selected features
selected_features_df = pd.read_csv(SCRIPT_DIR /"05_final_selected_features.csv")
final_features = selected_features_df['Feature'].tolist()

print("="*80)
print("CREATING ALL DATASETS")
print("="*80)
print(f"Selected features ({len(final_features)}): {final_features}\n")

# ── Dataset 1: Normal only (for Isolation Forest) ──────────────────────
print("\n" + "="*80)
print("DATASET 1: LMD2022 NORMAL (ISOLATION FOREST)")
print("="*80)

df1 = process_and_save_dataset(
    df_path=ISOLATION_FOREST_DATASET_PATH,
    output_name="isolation_forest_dataset.csv",
    selected_features=final_features,
    filter_label=0
)

# ── Dataset 2: Full training (for RF/LSTM) ─────────────────────────────
print("\n" + "="*80)
print("DATASET 2: LMD2022 FULL (RF/LSTM TRAINING)")
print("="*80)

df2 = process_and_save_dataset(
    df_path=SUPERVISED_DATASET_PATH,
    output_name="supervised_learning_dataset.csv",
    selected_features=final_features,
    filter_label=None  # All samples
)

# ── Dataset 3: Test set ──────────────────────────────
print("\n" + "="*80)
print("DATASET 3: LMD2023 TEST (ALREADY BALANCED)")
print("="*80)

df3 = process_and_save_dataset(
    df_path=TESTING_DATASET_PATH,
    output_name=f"lmd2023_test_dataset.csv",
    selected_features=final_features,
    filter_label=None
)
