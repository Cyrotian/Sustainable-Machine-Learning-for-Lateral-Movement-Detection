# complexity_sweep.py - Measure how model complexity affects performance and emissions.

import time
import warnings
import numpy as np
import pandas as pd

from codecarbon import EmissionsTracker
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_CSV = "final_datasets/supervised_learning_dataset.csv"
TEST_CSV = "final_datasets/lmd2023_test_dataset.csv"
LSTM_TRAIN_CSV = "final_datasets/lstm_train_sequenced.csv"
LSTM_TEST_CSV = "final_datasets/lstm_test_sequenced.csv"

FEATURES_CSV = "src/05_final_selected_features.csv"
OUTPUT_CSV = "results/complexity_sweep.csv"

LABEL_COLUMN = "isMalicious"
GROUP_COLUMN = "Computer"
RANDOM_STATE = 42

RF_ESTIMATOR_VALUES = [10, 25, 50, 75, 100, 150, 200, 300]
LGBM_ESTIMATOR_VALUES = [10, 25, 50, 75, 100, 150, 200, 300]
LSTM_HIDDEN_UNIT_VALUES = [16, 32, 64, 128, 256]

LSTM_WINDOW_SIZE = 10
LSTM_BATCH_SIZE = 256
LSTM_EPOCHS = 10
LSTM_PATIENCE = 3

selected_features_dataframe = pd.read_csv(FEATURES_CSV)
FEATURE_COLUMNS = selected_features_dataframe["Feature"].tolist()


# ============================================================================
# GENERAL HELPER FUNCTIONS
# ============================================================================

def print_section(title):
    # Print a section heading.
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_csv_file(file_path):
    # Load a CSV file.
    return pd.read_csv(file_path, low_memory=False)


def replace_unknown_category(value, known_classes, fallback_class):
    # Replace unseen categories with a fallback class.
    if value in known_classes:
        return value
    return fallback_class


def record_result(
    results,
    model_name,
    parameter_name,
    parameter_value,
    f1,
    precision,
    recall,
    false_positive_rate,
    training_time,
    training_emissions,
    inference_time,
    inference_emissions,
):
    # Store one model result in the results list.
    results.append(
        {
            "model": model_name,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "fpr": false_positive_rate,
            "train_time": training_time,
            "train_emissions": training_emissions,
            "inference_time": inference_time,
            "inference_emissions": inference_emissions,
        }
    )


def calculate_classification_metrics(true_labels, predicted_labels):
    # Calculate the main evaluation metrics.
    f1 = f1_score(true_labels, predicted_labels, average="binary")
    precision = precision_score(true_labels, predicted_labels, average="binary", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="binary", zero_division=0)

    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    return f1, precision, recall, false_positive_rate


def run_with_emissions_tracking(function_to_run, project_name, *args, **kwargs):
    # Run a function while measuring time and emissions.
    #
    # Returns:
    # - function result
    # - elapsed time
    # - emissions
    tracker = EmissionsTracker(project_name=project_name, log_level="error")
    tracker.start()

    start_time = time.time()
    function_result = function_to_run(*args, **kwargs)
    elapsed_time = time.time() - start_time

    emissions = tracker.stop()

    return function_result, elapsed_time, emissions


def save_results_to_csv(results, output_csv_path):
    # Save all results to CSV.
    results_dataframe = pd.DataFrame(results)
    results_dataframe.to_csv(output_csv_path, index=False)
    return results_dataframe


# ============================================================================
# SUPERVISED DATA HELPERS
# ============================================================================

def load_supervised_data():
    # Load the train and test datasets used by Random Forest and LightGBM.
 
    print("Loading supervised train and test data...")

    training_dataframe = load_csv_file(TRAIN_CSV)
    testing_dataframe = load_csv_file(TEST_CSV)

    X_train = training_dataframe.drop(columns=[LABEL_COLUMN])
    y_train = training_dataframe[LABEL_COLUMN].values

    X_test = testing_dataframe.drop(columns=[LABEL_COLUMN])
    y_test = testing_dataframe[LABEL_COLUMN].values

    print(f"  Train rows: {len(X_train):,} | Features: {X_train.shape[1]}")
    print(f"  Test rows:  {len(X_test):,} | Features: {X_test.shape[1]}")

    return X_train, y_train, X_test, y_test


def encode_categorical_columns(X_train, X_test):
    # Label-encode categorical columns using training data only.
    # Unknown test categories are mapped to the first known class.
    print("Encoding categorical columns...")

    encoded_X_train = X_train.copy()
    encoded_X_test = X_test.copy()
    encoders_by_column = {}

    categorical_columns = X_train.select_dtypes(include=["object"]).columns.tolist()

    for column_name in categorical_columns:
        label_encoder = LabelEncoder()

        encoded_X_train[column_name] = label_encoder.fit_transform(
            X_train[column_name].astype(str)
        )

        known_classes = set(label_encoder.classes_)
        fallback_class = label_encoder.classes_[0]

        encoded_X_test[column_name] = X_test[column_name].astype(str).apply(
            replace_unknown_category,
            args=(known_classes, fallback_class),
        )
        encoded_X_test[column_name] = label_encoder.transform(encoded_X_test[column_name])

        encoders_by_column[column_name] = label_encoder

    print(f"  Encoded {len(categorical_columns)} columns")

    return encoded_X_train.values, encoded_X_test.values, encoders_by_column


# ============================================================================
# RANDOM FOREST
# ============================================================================

def train_random_forest_model(X_train, y_train, number_of_estimators):
    # Train a Random Forest model with a specific number of trees.
    model = RandomForestClassifier(
        n_estimators=number_of_estimators,
        class_weight="balanced",
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict_with_model(model, X_test):
    # Run inference with a trained model.
    return model.predict(X_test)


def sweep_random_forest(X_train, y_train, X_test, y_test, results):
    # Test Random Forest across multiple tree counts.
    print_section("RANDOM FOREST SWEEP")

    for number_of_estimators in RF_ESTIMATOR_VALUES:
        print(f"\nTesting Random Forest with n_estimators = {number_of_estimators}")

        trained_model, training_time, training_emissions = run_with_emissions_tracking(
            train_random_forest_model,
            f"rf_train_{number_of_estimators}",
            X_train,
            y_train,
            number_of_estimators,
        )

        predictions, inference_time, inference_emissions = run_with_emissions_tracking(
            predict_with_model,
            f"rf_infer_{number_of_estimators}",
            trained_model,
            X_test,
        )

        f1, precision, recall, false_positive_rate = calculate_classification_metrics(
            y_test, predictions
        )

        print(
            f"  F1: {f1:.4f} | "
            f"Train: {training_time:.2f}s ({training_emissions * 1e6:.1f}µg) | "
            f"Infer: {inference_time:.2f}s ({inference_emissions * 1e6:.1f}µg)"
        )

        record_result(
            results=results,
            model_name="Random Forest",
            parameter_name="n_estimators",
            parameter_value=number_of_estimators,
            f1=f1,
            precision=precision,
            recall=recall,
            false_positive_rate=false_positive_rate,
            training_time=training_time,
            training_emissions=training_emissions,
            inference_time=inference_time,
            inference_emissions=inference_emissions,
        )


# ============================================================================
# LIGHTGBM
# ============================================================================

def train_lightgbm_model(X_train, y_train, number_of_estimators):
    # Train a LightGBM model with a specific number of trees.
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=number_of_estimators,
        is_unbalance=True,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def sweep_lightgbm(X_train, y_train, X_test, y_test, results):
    # Test LightGBM across multiple tree counts.
    print_section("LIGHTGBM SWEEP")

    for number_of_estimators in LGBM_ESTIMATOR_VALUES:
        print(f"\nTesting LightGBM with n_estimators = {number_of_estimators}")

        trained_model, training_time, training_emissions = run_with_emissions_tracking(
            train_lightgbm_model,
            f"lgbm_train_{number_of_estimators}",
            X_train,
            y_train,
            number_of_estimators,
        )

        predictions, inference_time, inference_emissions = run_with_emissions_tracking(
            predict_with_model,
            f"lgbm_infer_{number_of_estimators}",
            trained_model,
            X_test,
        )

        f1, precision, recall, false_positive_rate = calculate_classification_metrics(
            y_test, predictions
        )

        print(
            f"  F1: {f1:.4f} | "
            f"Train: {training_time:.2f}s ({training_emissions * 1e6:.1f}µg) | "
            f"Infer: {inference_time:.2f}s ({inference_emissions * 1e6:.1f}µg)"
        )

        record_result(
            results=results,
            model_name="LightGBM",
            parameter_name="n_estimators",
            parameter_value=number_of_estimators,
            f1=f1,
            precision=precision,
            recall=recall,
            false_positive_rate=false_positive_rate,
            training_time=training_time,
            training_emissions=training_emissions,
            inference_time=inference_time,
            inference_emissions=inference_emissions,
        )


# ============================================================================
# LSTM DATA PREPARATION
# ============================================================================

def split_lstm_data_by_computer(train_dataframe, validation_fraction=0.2):
    # Split LSTM training data by Computer so validation uses unseen machines.
    computer_ids = train_dataframe[GROUP_COLUMN].unique()

    number_of_validation_computers = max(1, int(len(computer_ids) * validation_fraction))

    random_generator = np.random.default_rng(RANDOM_STATE)
    random_generator.shuffle(computer_ids)

    validation_computers = computer_ids[:number_of_validation_computers]
    training_computers = computer_ids[number_of_validation_computers:]

    training_split = train_dataframe[
        train_dataframe[GROUP_COLUMN].isin(training_computers)
    ].copy()

    validation_split = train_dataframe[
        train_dataframe[GROUP_COLUMN].isin(validation_computers)
    ].copy()

    return training_split, validation_split, training_computers, validation_computers


def encode_and_scale_lstm_data(training_dataframe, other_dataframe, feature_columns):
    # Label-encode categorical columns and MinMax-scale numeric values.
    # Fit on training data only, then apply to another dataset.
    processed_training_dataframe = training_dataframe.copy()
    processed_other_dataframe = other_dataframe.copy()

    label_encoders = {}
    scaler = MinMaxScaler()

    categorical_columns = (
        training_dataframe[feature_columns]
        .select_dtypes(include=["object"])
        .columns
        .tolist()
    )

    for column_name in categorical_columns:
        label_encoder = LabelEncoder()

        processed_training_dataframe[column_name] = label_encoder.fit_transform(
            training_dataframe[column_name].astype(str)
        )

        known_classes = set(label_encoder.classes_)
        fallback_class = label_encoder.classes_[0]

        processed_other_dataframe[column_name] = other_dataframe[column_name].astype(str).apply(
            replace_unknown_category,
            args=(known_classes, fallback_class),
        )
        processed_other_dataframe[column_name] = label_encoder.transform(
            processed_other_dataframe[column_name]
        )

        label_encoders[column_name] = label_encoder

    processed_training_dataframe[feature_columns] = scaler.fit_transform(
        processed_training_dataframe[feature_columns]
    )
    processed_other_dataframe[feature_columns] = scaler.transform(
        processed_other_dataframe[feature_columns]
    )

    return processed_training_dataframe, processed_other_dataframe


def create_lstm_sequences(
    dataframe,
    feature_columns,
    window_size,
    label_column=LABEL_COLUMN,
    group_column=GROUP_COLUMN,
):
    # Create sliding window sequences for the LSTM.
    # Each sequence stays within one computer.
    sequence_features = []
    sequence_labels = []

    computer_ids = dataframe[group_column].unique()
    print(f"  Building sequences from {len(computer_ids)} computers")

    for computer_id in computer_ids:
        computer_rows = dataframe[dataframe[group_column] == computer_id]

        feature_values = computer_rows[feature_columns].values
        label_values = computer_rows[label_column].values

        
        for start_index in range(len(feature_values) - window_size + 1):
            end_index = start_index + window_size

            window = feature_values[start_index:end_index]
            label = label_values[end_index - 1]

            sequence_features.append(window)
            sequence_labels.append(label)

    X_sequences = np.array(sequence_features)
    y_sequences = np.array(sequence_labels)

    print(f"  Total sequences: {X_sequences.shape[0]:,}")
    print(f"  Sequence shape: {X_sequences.shape}")

    return X_sequences, y_sequences


def load_and_prepare_lstm_datasets():
    # Load, split, encode, scale, and sequence the LSTM data.
    print("Loading LSTM datasets...")

    full_training_dataframe = load_csv_file(LSTM_TRAIN_CSV)
    full_test_dataframe = load_csv_file(LSTM_TEST_CSV)

    training_split, validation_split, training_computers, validation_computers = (
        split_lstm_data_by_computer(full_training_dataframe)
    )

    print(
        f"  Train computers: {len(training_computers)} | "
        f"Validation computers: {len(validation_computers)}"
    )
    print(
        f"  Train rows: {len(training_split):,} | "
        f"Validation rows: {len(validation_split):,}"
    )

    encoded_training_split, encoded_test_dataframe = encode_and_scale_lstm_data(
        training_split,
        full_test_dataframe,
        FEATURE_COLUMNS,
    )

    _, encoded_validation_split = encode_and_scale_lstm_data(
        training_split,
        validation_split,
        FEATURE_COLUMNS,
    )

    print("\nBuilding LSTM training sequences...")
    X_train, y_train = create_lstm_sequences(
        encoded_training_split,
        FEATURE_COLUMNS,
        LSTM_WINDOW_SIZE,
    )

    print("Building LSTM validation sequences...")
    X_validation, y_validation = create_lstm_sequences(
        encoded_validation_split,
        FEATURE_COLUMNS,
        LSTM_WINDOW_SIZE,
    )

    print("Building LSTM test sequences...")
    X_test, y_test = create_lstm_sequences(
        encoded_test_dataframe,
        FEATURE_COLUMNS,
        LSTM_WINDOW_SIZE,
    )

    y_train = y_train.astype(int)
    y_validation = y_validation.astype(int)
    y_test = y_test.astype(int)

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def calculate_lstm_class_weights(y_train):
    # Calculate balanced class weights for the LSTM.
    unique_classes = np.unique(y_train)
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y_train,
    )

    class_weights = dict(zip(unique_classes.astype(int), class_weight_values))
    return class_weights


# ============================================================================
# LSTM MODEL
# ============================================================================

def build_lstm_model(number_of_hidden_units, number_of_features):
    # Create an LSTM model with the given hidden layer size.
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential(
        [
            LSTM(
                number_of_hidden_units,
                input_shape=(LSTM_WINDOW_SIZE, number_of_features),
                return_sequences=False,
            ),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", "recall"],
    )

    return model


def train_lstm_model(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    class_weights,
):
    # Train the LSTM model with early stopping.
    from tensorflow.keras.callbacks import EarlyStopping

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=LSTM_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=0,
    )

    return history


def predict_with_lstm(model, X_test, threshold=0.5):
    # Generate binary predictions from LSTM probabilities.
    predicted_probabilities = model.predict(X_test, verbose=0)
    predicted_labels = (predicted_probabilities.flatten() >= threshold).astype(int)
    return predicted_labels


def sweep_lstm(results):
    # Test LSTM across multiple hidden-unit sizes.
    import tensorflow as tf

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")

    print_section("LSTM SWEEP")

    X_train, y_train, X_validation, y_validation, X_test, y_test = (
        load_and_prepare_lstm_datasets()
    )

    class_weights = calculate_lstm_class_weights(y_train)
    print(f"  Class weights: {class_weights}")

    for number_of_hidden_units in LSTM_HIDDEN_UNIT_VALUES:
        print(f"\nTesting LSTM with hidden_units = {number_of_hidden_units}")

        tf.keras.backend.clear_session()
        tf.random.set_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        lstm_model = build_lstm_model(
            number_of_hidden_units=number_of_hidden_units,
            number_of_features=len(FEATURE_COLUMNS),
        )

        history, training_time, training_emissions = run_with_emissions_tracking(
            train_lstm_model,
            f"lstm_train_{number_of_hidden_units}",
            lstm_model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            class_weights,
        )

        actual_number_of_epochs = len(history.history["loss"])

        predictions, inference_time, inference_emissions = run_with_emissions_tracking(
            predict_with_lstm,
            f"lstm_infer_{number_of_hidden_units}",
            lstm_model,
            X_test,
            0.5,
        )

        f1, precision, recall, false_positive_rate = calculate_classification_metrics(
            y_test, predictions
        )

        print(
            f"  F1: {f1:.4f} | "
            f"Epochs: {actual_number_of_epochs} | "
            f"Train: {training_time:.1f}s ({training_emissions * 1e6:.1f}µg) | "
            f"Infer: {inference_time:.1f}s ({inference_emissions * 1e6:.1f}µg)"
        )

        record_result(
            results=results,
            model_name="LSTM",
            parameter_name="hidden_units",
            parameter_value=number_of_hidden_units,
            f1=f1,
            precision=precision,
            recall=recall,
            false_positive_rate=false_positive_rate,
            training_time=training_time,
            training_emissions=training_emissions,
            inference_time=inference_time,
            inference_emissions=inference_emissions,
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Run the full complexity sweep and save the results.
    os.makedirs("results", exist_ok=True)
    all_results = []

    X_train, y_train, X_test, y_test = load_supervised_data()
    encoded_X_train, encoded_X_test, encoders = encode_categorical_columns(X_train, X_test)

    sweep_random_forest(encoded_X_train, y_train, encoded_X_test, y_test, all_results)
    sweep_lightgbm(encoded_X_train, y_train, encoded_X_test, y_test, all_results)
    sweep_lstm(all_results)

    results_dataframe = save_results_to_csv(all_results, OUTPUT_CSV)

    print_section("SWEEP COMPLETE")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Total configurations tested: {len(results_dataframe)}")

    print("\nSummary:")
    print(results_dataframe.to_string(index=False))


if __name__ == "__main__":
    main()