import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import sys, os
sys.path.insert(0, os.path.abspath('.'))

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
    from src.feature_engineering import create_subset, drop_features, engineer_features
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
    df = create_subset(df)

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
    print("STEP 5: Prepare LMD-2023 test set (3-class -> binary, imbalanced)")
    print("=" * 80)

    from src.prepare_lmd2023_test import prepare_lmd2023_test
    prepare_lmd2023_test()

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
              "Emissions:", round(result["emissions"], 8), "kg CO2")

    summary = random_forest_model.get_inference_summary()
    print("\n-- Random Forest Inference Summary (mean +/- std) --")
    for key, val in summary.items():
        print("  " + key + ":", round(val, 6))

    save_model_metrics("Random Forest", random_forest_model.get_inference_summary())


def lstm():
    from models.lstm import LSTMModel
    from codecarbon import EmissionsTracker

    supervised_learning_dataset = pd.read_csv(
        "final_datasets/supervised_learning_dataset.csv", low_memory=False
    )
    testing_dataset = pd.read_csv(
        "final_datasets/lmd2023_test_dataset.csv", low_memory=False
    )

    target_column = "isMalicious"

    X = supervised_learning_dataset.drop(columns=[target_column])
    y = supervised_learning_dataset[target_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test = testing_dataset.drop(columns=[target_column])
    y_test = testing_dataset[target_column]

    carbon_tracker = EmissionsTracker(
        output_dir="emissions",
        project_name="lstm_training",
        log_level="error",
    )

    config = {"epochs": 10, "batch_size": 512}
    lstm_model = LSTMModel(config)
    lstm_model.train(X_train, y_train, X_val, y_val, carbon_tracker)

    print("\nSanity check:", lstm_model.evaluate(X_test, y_test))

    print("\n-- LSTM: 10 Inference Runs --")
    for i in range(10):
        result = lstm_model.inference_with_metrics(X_test, y_test)
        print("  Run " + str(i + 1) + "/10 |",
              "F1:", round(result["f1"], 4), "|",
              "Time:", round(result["inference_time"], 3), "s |",
              "Emissions:", round(result["emissions"], 8), "kg CO2")

    summary = lstm_model.get_inference_summary()
    print("\n-- LSTM Inference Summary (mean +/- std) --")
    for key, val in summary.items():
        print("  " + key + ":", round(val, 6))


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
        preprocess()