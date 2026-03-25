"""
main.py — Single entry point for the entire dissertation pipeline.

USAGE:
  python main.py preprocess      # Run feature selection + generate plots + create datasets
  python main.py rf              # Train and evaluate Random Forest
  python main.py if              # Train and evaluate Isolation Forest
  python main.py lstm            # Train and evaluate LSTM
  python main.py all             # Run everything

FOLDER STRUCTURE:
  main.py              <- this file (root)
  src/                 <- preprocessing code
    feature_engineering.py
    feature_selection.py
    preprocessing_pipeline.py
    prepare_lmd2023_test.py
    plotting.py
  models/              <- model code
    random_forest.py
    isolation_forest.py
    lstm.py
  final_datasets/      <- generated CSVs (created by preprocess)
  plots/               <- generated charts (created by preprocess)
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from save_metrics import save_model_metrics


# ============================================================================
# PREPROCESSING: Feature selection + plots + dataset creation
# ============================================================================

def preprocess():
    """
    Run the full preprocessing pipeline:
      1. Load raw LMD-2022 data
      2. Drop bad features (Section 1)
      3. Engineer new features (Section 2)
      4. Run feature selection (Sections 3-9) -> saves CSVs
      5. Generate all dissertation plots -> saves PNGs
      6. Prepare LMD-2023 test set (3-class -> binary, imbalanced)
      7. Create 5 final datasets -> saves CSVs
    """
    from src.feature_engineering import  drop_features, engineer_features
    from src.feature_selection import run_feature_selection_pipeline
    from src.plotting import plot_all
    from config import Config

    print("=" * 80)
    print("STEP 1: Load raw LMD-2022 data")
    print("=" * 80)

    df = pd.read_csv(Config.LMD_2022_PATH, low_memory=False)

    # Clean up: drop Unnamed columns, rename label, filter valid rows
    useful_cols = []
    for col in df.columns:
        if not col.startswith("Unnamed"):
            useful_cols.append(col)
    df = df[useful_cols].copy()

    df.rename(columns={"Label": "isMalicious"}, inplace=True)

    valid_rows = (df["isMalicious"] == "1") | (df["isMalicious"] == "0")
    df = df[valid_rows].copy()
    df["isMalicious"] = df["isMalicious"].astype(int)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print(df["isMalicious"].value_counts())

    # Keep a copy of the raw data for EDA plots (before we modify it)
    raw_df = df.copy()

    print("\n" + "=" * 80)
    print("STEP 2: Stratified subset + drop bad features + engineer new ones")
    print("=" * 80)

    # Take a 50k stratified subset for feature selection
    # This produces a better-behaved GroupShuffleSplit than the full dataset

    df_clean, dropped, cols_before = drop_features(df)
    X, y, groups = engineer_features(df_clean)

    print("\n" + "=" * 80)
    print("STEP 3: Feature selection (Sections 3-9)")
    print("=" * 80)

    results = run_feature_selection_pipeline(X, y, groups, output_dir="src")

    print("\nSelected", results["OPTIMAL_N"], "features:")
    for i in range(len(results["final_features"])):
        print("  " + str(i + 1) + ". " + results["final_features"][i])

    print("\n" + "=" * 80)
    print("STEP 4: Generate dissertation plots")
    print("=" * 80)

    plot_all(raw_df, results, output_dir="plots")
 

    print("\n" + "=" * 80)
    print("STEP 6: Create final datasets for all models")
    print("=" * 80)

    from src.preprocessing_pipeline import create_all_datasets
    create_all_datasets(features_csv_path="src/05_final_selected_features.csv")


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def random_forest():
    from models.random_forest import RandomForestModel

    supervised_learning_dataset = pd.read_csv(
        "final_datasets/supervised_learning_dataset.csv", low_memory=False
    )
    testing_dataset = pd.read_csv(
        "final_datasets/lmd2023_test_dataset.csv", low_memory=False
    )

    target_column = "isMalicious"

    X_train = supervised_learning_dataset.drop(columns=[target_column])
    y_train = supervised_learning_dataset[target_column]
    X_test = testing_dataset.drop(columns=[target_column])
    y_test = testing_dataset[target_column]

    random_forest_model = RandomForestModel()
    random_forest_model.train(X_train, y_train)

    random_forest_model.plot_decision_tree(X_train)
    print("\nSanity check:", random_forest_model.evaluate(X_test, y_test))

    print("\n-- Random Forest: 10 Inference Runs --")
    for i in range(10):
        result = random_forest_model.inference_with_metrics(X_test, y_test)
        print("  Run " + str(i + 1) + "/10 |",
              "F1:", round(result["f1"], 4), "|",
              "Time:", round(result["inference_time"], 3), "s |",
              "fpr:", round(result["fpr"], 6), "|",
              "Emissions:", round(result["emissions"], 8), "kg CO2")

    summary = random_forest_model.get_inference_summary()
    print("\n-- Random Forest Inference Summary (mean +/- std) --")
    for key, val in summary.items():
        print("  " + key + ":", round(val, 6))

    save_model_metrics("Random Forest", random_forest_model.get_inference_summary())


def lightgbm():
    from models.lightGBM import LightGBMModel

    supervised_learning_dataset = pd.read_csv(
        "final_datasets/supervised_learning_dataset.csv", low_memory=False
    )
    testing_dataset = pd.read_csv(
        "final_datasets/lmd2023_test_dataset.csv", low_memory=False
    )

    target_column = "isMalicious"

    X = supervised_learning_dataset.drop(columns=[target_column])
    y = supervised_learning_dataset[target_column]

    # Validation split for early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test = testing_dataset.drop(columns=[target_column])
    y_test = testing_dataset[target_column]

    print("Train:", len(X_train), "| Val:", len(X_val), "| Test:", len(X_test))

    lightgbm_model = LightGBMModel(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
    )

    lightgbm_model.train(X_train, y_train, X_val, y_val)

    print("\nSanity check:", lightgbm_model.evaluate(X_test, y_test))

    print("\n-- LightGBM: 10 Inference Runs --")
    for i in range(10):
        result = lightgbm_model.inference_with_metrics(X_test, y_test)
        print(
            "Run " + str(i + 1) + "/10 |",
            "F1:", round(result["f1"], 4), "|",
            "Prec:", round(result["precision"], 4), "|",
            "Rec:", round(result["recall"], 4), "|",
            "fpr:", round(result["fpr"], 6), "|",
            "Time:", round(result["inference_time"], 3), "s |",
            "Emissions:", round(result["emissions"], 8), "kg CO2",
        )

    summary = lightgbm_model.get_inference_summary()
    print("\n-- LightGBM Inference Summary (mean +/- std) --")
    for key, val in summary.items():
        print("  " + key + ":", round(val, 6))

    save_model_metrics("LightGBM", lightgbm_model.get_inference_summary())


def lstm():
    import numpy as np
    import pandas as pd
    from models.lstm import LSTMModel
    from codecarbon import EmissionsTracker

    # Load sequenced datasets (sorted by Computer + SystemTime)
    train_df = pd.read_csv("final_datasets/lstm_train_sequenced.csv", low_memory=False)
    test_df = pd.read_csv("final_datasets/lstm_test_sequenced.csv", low_memory=False)

    # Load model features from feature selection output (excludes metadata cols)
    features_df = pd.read_csv("src/05_final_selected_features.csv")
    feature_cols = features_df["Feature"].tolist()

    # Split training data into train/val BY COMPUTER to avoid leakage
    computers = train_df["Computer"].unique()
    n_val = max(1, int(len(computers) * 0.2))
    rng = np.random.default_rng(42)
    rng.shuffle(computers)

    val_computers = computers[:n_val]
    train_computers = computers[n_val:]

    df_train = train_df[train_df["Computer"].isin(train_computers)].copy()
    df_val = train_df[train_df["Computer"].isin(val_computers)].copy()

    print("Train computers:", len(train_computers), "| Val computers:", len(val_computers))
    print("Train rows:", len(df_train), "| Val rows:", len(df_val))

    carbon_tracker = EmissionsTracker(
        output_dir="emissions",
        project_name="lstm_training",
        log_level="error",
    )

    config = {
        "epochs": 10,
        "batch_size": 256,
        "window_size": 10,
        "threshold": 0.5,   
        "patience": 3,
    }

    lstm_model = LSTMModel(config)

    # Train
    lstm_model.train(df_train, df_val, feature_cols, carbon_tracker)

    print("\nSanity check:", lstm_model.evaluate(test_df, feature_cols))

    # Pre-build test sequences once (don't re-create for each inference run)
    print("\nPreparing test sequences for inference runs...")

    df_test_enc = lstm_model._encode(test_df[feature_cols], fit=False)
    test_scaled = lstm_model._scale(df_test_enc, fit=False)

    df_test_processed = pd.DataFrame(test_scaled, columns=feature_cols)
    df_test_processed["isMalicious"] = test_df["isMalicious"].values
    df_test_processed["Computer"] = test_df["Computer"].values

    X_test_seq, y_test_seq = lstm_model.create_sequences(
        df_test_processed, feature_cols, label_col="isMalicious", group_col="Computer"
    )

    print("\n-- LSTM: 10 Inference Runs --")
    for i in range(10):
        result = lstm_model.inference_with_metrics(X_test_seq, y_test_seq)
        print(
            f"  Run {i+1}/10 | "
            f"F1: {result['f1']:.4f} | "
            f"Prec: {result['precision']:.4f} | "
            f"Rec: {result['recall']:.4f} | "
            f"fpr: {result['fpr']:.6f} | "
            f"Time: {result['inference_time']:.3f}s | "
            f"Emissions: {result['emissions']:.8f} kg CO2"
        )

    summary = lstm_model.get_inference_summary()
    print("\n-- LSTM Inference Summary (mean +/- std) --")
    for key, val in summary.items():
        print(f"  {key}: {val:.6f}")

    save_model_metrics("LSTM", lstm_model.get_inference_summary())


if __name__ == "__main__":
        preprocess()
        #lstm()
        #random_forest()
        #lightgbm()