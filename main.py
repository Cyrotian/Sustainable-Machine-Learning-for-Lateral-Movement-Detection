"""
main.py — Main entry point for the dissertation pipeline.

This file controls four main tasks:
1. Preprocessing the raw data
2. Training and evaluating the Random Forest model
3. Training and evaluating the LightGBM model
4. Training and evaluating the LSTM model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from save_metrics import save_model_metrics


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_section(title):
    # Print a clear section heading."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_csv_file(file_path):
    # Load a CSV file with low_memory disabled."""
    return pd.read_csv(file_path, low_memory=False)


def remove_unnamed_columns(dataframe):
    # Remove columns whose names start with 'Unnamed'."""
    useful_columns = []

    for column_name in dataframe.columns:
        if not column_name.startswith("Unnamed"):
            useful_columns.append(column_name)

    return dataframe[useful_columns].copy()


def clean_label_column(dataframe):
    # Rename the label column and keep only valid binary rows.
    # Valid values are '0' and '1'.
  
    cleaned_dataframe = dataframe.copy()

    cleaned_dataframe.rename(columns={"Label": "isMalicious"}, inplace=True)

    valid_rows = (
        (cleaned_dataframe["isMalicious"] == "0")
        | (cleaned_dataframe["isMalicious"] == "1")
    )

    cleaned_dataframe = cleaned_dataframe[valid_rows].copy()
    cleaned_dataframe["isMalicious"] = cleaned_dataframe["isMalicious"].astype(int)

    return cleaned_dataframe


def print_dataset_summary(dataframe, label_column="isMalicious"):
    # Print basic dataset information.
    print("Rows:", len(dataframe))
    print("Columns:", len(dataframe.columns))
    print(dataframe[label_column].value_counts())


def load_supervised_train_and_test():
    # Load the supervised training dataset and the LMD-2023 test dataset
    training_dataframe = load_csv_file("final_datasets/supervised_learning_dataset.csv")
    testing_dataframe = load_csv_file("final_datasets/lmd2023_test_dataset.csv")
    return training_dataframe, testing_dataframe


def split_features_and_labels(dataframe, target_column="isMalicious"):
    # Split a dataframe into X and y
    features = dataframe.drop(columns=[target_column])
    labels = dataframe[target_column]

    return features, labels


def print_inference_summary(model_name, summary_dictionary):
    # Print mean and standard deviation summary values
    print(f"\n-- {model_name} Inference Summary (mean +/- std) --")

    for metric_name, metric_value in summary_dictionary.items():
        print(f"  {metric_name}: {round(metric_value, 6)}")


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess():
    """
    Run the full preprocessing pipeline:
    1. Load raw LMD-2022 data
    2. Clean the dataset
    3. Drop bad features
    4. Engineer new features
    5. Run feature selection
    6. Generate plots
    7. Create final datasets
    """

    from config import Config
    from src.feature_engineering import drop_features, engineer_features
    from src.feature_selection import run_feature_selection_pipeline
    from src.plotting import plot_all
    from src.preprocessing_pipeline import create_all_datasets

    print_section("STEP 1: Load raw LMD-2022 data")

    raw_dataframe = load_csv_file(Config.LMD_2022_PATH)
    raw_dataframe = remove_unnamed_columns(raw_dataframe)
    raw_dataframe = clean_label_column(raw_dataframe)

    print_dataset_summary(raw_dataframe)

    # Keep an untouched copy for plotting
    original_dataframe_for_plots = raw_dataframe.copy()

    print_section("STEP 2: Drop bad features and engineer new ones")

    cleaned_dataframe, dropped_features, original_column_count = drop_features(raw_dataframe)
    feature_matrix, labels, groups = engineer_features(cleaned_dataframe)

    print_section("STEP 3: Run feature selection")

    feature_selection_results = run_feature_selection_pipeline(
        feature_matrix,
        labels,
        groups,
        output_dir="src"
    )

    selected_features = feature_selection_results["final_features"]
    optimal_feature_count = feature_selection_results["OPTIMAL_N"]

    print(f"\nSelected {optimal_feature_count} features:")

    for index, feature_name in enumerate(selected_features, start=1):
        print(f"  {index}. {feature_name}")

    print_section("STEP 4: Generate dissertation plots")

    plot_all(
        original_dataframe_for_plots,
        feature_selection_results,
        output_dir="plots"
    )

    print_section("STEP 5: Create final datasets")

    create_all_datasets(features_csv_path="src/05_final_selected_features.csv")


# ============================================================================
# RANDOM FOREST
# ============================================================================

def random_forest():
    """Train and evaluate the Random Forest model."""
    from models.random_forest import RandomForestModel

    print_section("RANDOM FOREST")

    training_dataframe, testing_dataframe = load_supervised_train_and_test()

    X_train, y_train = split_features_and_labels(training_dataframe)
    X_test, y_test = split_features_and_labels(testing_dataframe)

    random_forest_model = RandomForestModel()

    print("Training Random Forest...")
    random_forest_model.train(X_train, y_train)

    print("Generating decision tree plot...")
    random_forest_model.plot_decision_tree(X_train)

    print("\nSanity check:", random_forest_model.evaluate(X_test, y_test))

    print("\n-- Random Forest: 10 Inference Runs --")
    for run_number in range(10):
        result = random_forest_model.inference_with_metrics(X_test, y_test)

        print(
            f"  Run {run_number + 1}/10 | "
            f"F1: {result['f1']:.4f} | "
            f"Time: {result['inference_time']:.3f}s | "
            f"FPR: {result['fpr']:.6f} | "
            f"Emissions: {result['emissions']:.8f} kg CO2"
        )

    summary = random_forest_model.get_inference_summary()
    print_inference_summary("Random Forest", summary)

    save_model_metrics("Random Forest", summary)


# ============================================================================
# LIGHTGBM
# ============================================================================

def lightgbm():
    """Train and evaluate the LightGBM model."""
    from models.lightGBM import LightGBMModel

    print_section("LIGHTGBM")

    training_dataframe, testing_dataframe = load_supervised_train_and_test()

    X, y = split_features_and_labels(training_dataframe)
    X_test, y_test = split_features_and_labels(testing_dataframe)

    print("Creating validation split for early stopping...")

    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(
        "Train:", len(X_train),
        "| Validation:", len(X_validation),
        "| Test:", len(X_test)
    )

    lightgbm_model = LightGBMModel()

    print("Training LightGBM...")
    lightgbm_model.train(X_train, y_train, X_validation, y_validation)

    print("\nSanity check:", lightgbm_model.evaluate(X_test, y_test))

    print("\n-- LightGBM: 10 Inference Runs --")
    for run_number in range(10):
        result = lightgbm_model.inference_with_metrics(X_test, y_test)

        print(
            f"  Run {run_number + 1}/10 | "
            f"F1: {result['f1']:.4f} | "
            f"Prec: {result['precision']:.4f} | "
            f"Rec: {result['recall']:.4f} | "
            f"FPR: {result['fpr']:.6f} | "
            f"Time: {result['inference_time']:.3f}s | "
            f"Emissions: {result['emissions']:.8f} kg CO2"
        )

    summary = lightgbm_model.get_inference_summary()
    print_inference_summary("LightGBM", summary)

    save_model_metrics("LightGBM", summary)


# ============================================================================
# LSTM
# ============================================================================

def split_lstm_train_and_validation_by_computer(train_dataframe, validation_fraction=0.2):
    """
    Split the LSTM training data by computer to avoid leakage.
    This ensures the validation set contains unseen computers.
    """
    computer_ids = train_dataframe["Computer"].unique()

    number_of_validation_computers = max(1, int(len(computer_ids) * validation_fraction))

    random_generator = np.random.default_rng(42)
    random_generator.shuffle(computer_ids)

    validation_computers = computer_ids[:number_of_validation_computers]
    training_computers = computer_ids[number_of_validation_computers:]

    training_split = train_dataframe[train_dataframe["Computer"].isin(training_computers)].copy()
    validation_split = train_dataframe[train_dataframe["Computer"].isin(validation_computers)].copy()

    return training_split, validation_split, training_computers, validation_computers


def prepare_lstm_test_sequences(lstm_model, test_dataframe, feature_columns):
    """
    Encode, scale, and convert the test set into LSTM sequences.
    This is done once before repeated inference runs.
    """
    encoded_test_features = lstm_model._encode(test_dataframe[feature_columns], fit=False)
    scaled_test_features = lstm_model._scale(encoded_test_features, fit=False)

    processed_test_dataframe = pd.DataFrame(scaled_test_features, columns=feature_columns)
    processed_test_dataframe["isMalicious"] = test_dataframe["isMalicious"].values
    processed_test_dataframe["Computer"] = test_dataframe["Computer"].values

    X_test_sequences, y_test_sequences = lstm_model.create_sequences(
        processed_test_dataframe,
        feature_columns,
        label_col="isMalicious",
        group_col="Computer"
    )

    return X_test_sequences, y_test_sequences


def lstm():
    """Train and evaluate the LSTM model."""
    from models.lstm import LSTMModel
    from codecarbon import EmissionsTracker

    print_section("LSTM")

    train_dataframe = load_csv_file("final_datasets/lstm_train_sequenced.csv")
    test_dataframe = load_csv_file("final_datasets/lstm_test_sequenced.csv")

    selected_features_dataframe = load_csv_file("src/05_final_selected_features.csv")
    feature_columns = selected_features_dataframe["Feature"].tolist()

    training_split, validation_split, training_computers, validation_computers = (
        split_lstm_train_and_validation_by_computer(train_dataframe)
    )

    print(
        "Train computers:", len(training_computers),
        "| Validation computers:", len(validation_computers)
    )
    print(
        "Train rows:", len(training_split),
        "| Validation rows:", len(validation_split)
    )

    carbon_tracker = EmissionsTracker(
        output_dir="emissions",
        project_name="lstm_training",
        log_level="error",
    )

    lstm_config = {
        "epochs": 10,
        "batch_size": 256,
        "window_size": 10,
        "threshold": 0.5,
        "patience": 3,
    }

    lstm_model = LSTMModel(lstm_config)

    print("Training LSTM...")
    lstm_model.train(training_split, validation_split, feature_columns, carbon_tracker)

    print("\nSanity check:", lstm_model.evaluate(test_dataframe, feature_columns))

    print("\nPreparing test sequences...")
    X_test_sequences, y_test_sequences = prepare_lstm_test_sequences(
        lstm_model,
        test_dataframe,
        feature_columns
    )

    print("\n-- LSTM: 10 Inference Runs --")
    for run_number in range(10):
        result = lstm_model.inference_with_metrics(X_test_sequences, y_test_sequences)

        print(
            f"  Run {run_number + 1}/10 | "
            f"F1: {result['f1']:.4f} | "
            f"Prec: {result['precision']:.4f} | "
            f"Rec: {result['recall']:.4f} | "
            f"FPR: {result['fpr']:.6f} | "
            f"Time: {result['inference_time']:.3f}s | "
            f"Emissions: {result['emissions']:.8f} kg CO2"
        )

    summary = lstm_model.get_inference_summary()
    print_inference_summary("LSTM", summary)

    save_model_metrics("LSTM", summary)



# RUN PIPELINE


if __name__ == "__main__":
    preprocess()
    lstm()
    random_forest()
    lightgbm()