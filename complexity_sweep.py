"""
complexity_sweep.py — Map complexity parameters to emissions and F1.

PURPOSE: For each model, vary the key complexity parameter while holding
         everything else constant. Record F1 and CO₂ at each step.

         Random Forest:  n_estimators  (10, 25, 50, 75, 100, 150, 200, 300)
         LightGBM:       n_estimators  (10, 25, 50, 75, 100, 150, 200, 300)
         LSTM:           hidden_units  (16, 32, 64, 128, 256)

         All parameters directly affect inference cost:
           - More trees = more predictions to aggregate
           - More LSTM units = larger weight matrices to compute

OUTPUT:  results/complexity_sweep.csv

USAGE:   python complexity_sweep.py
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from codecarbon import EmissionsTracker
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION — UPDATE THESE PATHS TO MATCH YOUR PROJECT
# ============================================================================

TRAIN_CSV = "final_datasets/supervised_learning_dataset.csv"
TEST_CSV = "final_datasets/lmd2023_test_dataset.csv"
LSTM_TRAIN_CSV = "final_datasets/lstm_train_sequenced.csv"
LSTM_TEST_CSV = "final_datasets/lstm_test_sequenced.csv"

OUTPUT_CSV = "results/complexity_sweep.csv"

RANDOM_STATE = 42
LABEL_COL = "isMalicious"

RF_ESTIMATORS = [10, 25, 50, 75, 100, 150, 200, 300]
LGBM_ESTIMATORS = [10, 25, 50, 75, 100, 150, 200, 300]
LSTM_UNITS = [16, 32, 64, 128, 256]

LSTM_WINDOW_SIZE = 10
LSTM_BATCH_SIZE = 256
LSTM_EPOCHS = 10
LSTM_PATIENCE = 3

features_df = pd.read_csv("src/05_final_selected_features.csv")
FEATURE_COLS = features_df["Feature"].tolist()


# ============================================================================
# HELPERS
# ============================================================================

def load_supervised_data():
    """Load supervised learning train/test data as DataFrames."""
    print("Loading supervised data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    y_train = train[LABEL_COL].values
    X_train = train.drop(columns=[LABEL_COL])

    y_test = test[LABEL_COL].values
    X_test = test.drop(columns=[LABEL_COL])

    print(f"  Train: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]:,} rows, {X_test.shape[1]} features")

    return X_train, y_train, X_test, y_test


def encode_data(X_train, X_test):
    """
    Label-encode categorical columns, matching the model implementations.
    Fitted on training data only. Unknown test categories mapped to the
    first known class (same behaviour as RandomForestModel._encode).
    """
    print("Encoding categorical features...")
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    encoders = {}

    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        X_train_enc[col] = le.fit_transform(X_train[col].astype(str))

        known = set(le.classes_)
        X_test_enc[col] = X_test[col].astype(str).apply(
            lambda x: x if x in known else le.classes_[0]
        )
        X_test_enc[col] = le.transform(X_test_enc[col])

        encoders[col] = le

    print(f"  Encoded {len(categorical_cols)} categorical columns: {categorical_cols}")

    return X_train_enc.values, X_test_enc.values, encoders


def compute_metrics(y_true, y_pred):
    """Compute F1, precision, recall, FPR."""
    f1 = f1_score(y_true, y_pred, average="binary")
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return f1, prec, rec, fpr


def append_result(results, model, param_name, param_value,
                  f1, prec, rec, fpr,
                  train_time, train_emissions,
                  inference_time, inference_emissions):
    """Append one sweep result."""
    results.append({
        "model": model,
        "parameter_name": param_name,
        "parameter_value": param_value,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "train_time": train_time,
        "train_emissions": train_emissions,
        "inference_time": inference_time,
        "inference_emissions": inference_emissions,
    })


# ============================================================================
# RANDOM FOREST SWEEP
# ============================================================================

def sweep_rf(X_train, y_train, X_test, y_test, results):
    """Sweep Random Forest over n_estimators."""
    print("\n" + "=" * 60)
    print("RANDOM FOREST SWEEP: n_estimators")
    print("=" * 60)

    for n in RF_ESTIMATORS:
        print(f"\n  n_estimators = {n}")

        tracker = EmissionsTracker(project_name=f"rf_sweep_{n}", log_level="error")
        tracker.start()
        t0 = time.time()

        model = RandomForestClassifier(
            n_estimators=n,
            class_weight="balanced",
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        train_time = time.time() - t0
        train_emissions = tracker.stop()

        tracker = EmissionsTracker(project_name=f"rf_infer_{n}", log_level="error")
        tracker.start()
        t0 = time.time()

        y_pred = model.predict(X_test)

        inference_time = time.time() - t0
        inference_emissions = tracker.stop()

        f1, prec, rec, fpr = compute_metrics(y_test, y_pred)

        print(f"    F1: {f1:.4f} | Train: {train_time:.2f}s ({train_emissions*1e6:.1f}µg) "
              f"| Infer: {inference_time:.2f}s ({inference_emissions*1e6:.1f}µg)")

        append_result(results, "Random Forest", "n_estimators", n,
                      f1, prec, rec, fpr,
                      train_time, train_emissions,
                      inference_time, inference_emissions)


# ============================================================================
# LIGHTGBM SWEEP
# ============================================================================

def sweep_lgbm(X_train, y_train, X_test, y_test, results):
    """Sweep LightGBM over n_estimators."""
    from lightgbm import LGBMClassifier

    print("\n" + "=" * 60)
    print("LIGHTGBM SWEEP: n_estimators")
    print("=" * 60)

    for n in LGBM_ESTIMATORS:
        print(f"\n  n_estimators = {n}")

        tracker = EmissionsTracker(project_name=f"lgbm_sweep_{n}", log_level="error")
        tracker.start()
        t0 = time.time()

        model = LGBMClassifier(
            n_estimators=n,
            is_unbalance=True,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        train_time = time.time() - t0
        train_emissions = tracker.stop()

        tracker = EmissionsTracker(project_name=f"lgbm_infer_{n}", log_level="error")
        tracker.start()
        t0 = time.time()

        y_pred = model.predict(X_test)

        inference_time = time.time() - t0
        inference_emissions = tracker.stop()

        f1, prec, rec, fpr = compute_metrics(y_test, y_pred)

        print(f"    F1: {f1:.4f} | Train: {train_time:.2f}s ({train_emissions*1e6:.1f}µg) "
              f"| Infer: {inference_time:.2f}s ({inference_emissions*1e6:.1f}µg)")

        append_result(results, "LightGBM", "n_estimators", n,
                      f1, prec, rec, fpr,
                      train_time, train_emissions,
                      inference_time, inference_emissions)


# ============================================================================
# LSTM SWEEP
# ============================================================================

def encode_and_scale_lstm(df_train, df_test, feature_cols):
    """
    LabelEncode + MinMaxScale the LSTM data, matching LSTMModel._encode
    and LSTMModel._scale. Fitted on training data only.
    """
    print("Encoding and scaling LSTM features...")
    label_encoders = {}
    scaler = MinMaxScaler()

    train_enc = df_train.copy()
    test_enc = df_test.copy()

    categorical_cols = df_train[feature_cols].select_dtypes(include=["object"]).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        train_enc[col] = le.fit_transform(df_train[col].astype(str))

        known = set(le.classes_)
        test_enc[col] = df_test[col].astype(str).apply(
            lambda x: x if x in known else le.classes_[0]
        )
        test_enc[col] = le.transform(test_enc[col])

        label_encoders[col] = le

    train_enc[feature_cols] = scaler.fit_transform(train_enc[feature_cols])
    test_enc[feature_cols] = scaler.transform(test_enc[feature_cols])

    print(f"  Encoded {len(categorical_cols)} columns, scaled to [0, 1]")

    return train_enc, test_enc


def build_sequences(df, feature_cols, window_size,
                    label_col="isMalicious", group_col="Computer"):
    """
    Build sliding windows grouped by Computer, matching LSTMModel.create_sequences.
    No zero-padding — windows start at index (window_size - 1).
    """
    X_windows = []
    y_windows = []

    computers = df[group_col].unique()
    print(f"  Creating sequences from {len(computers)} computers, window_size={window_size}")

    for comp in computers:
        comp_data = df[df[group_col] == comp]
        features = comp_data[feature_cols].values
        labels = comp_data[label_col].values

        for i in range(len(features) - window_size + 1):
            window = features[i : i + window_size]
            label = labels[i + window_size - 1]
            X_windows.append(window)
            y_windows.append(label)

    X_seq = np.array(X_windows)
    y_seq = np.array(y_windows)

    print(f"  Sequences: {X_seq.shape[0]:,} | Shape: {X_seq.shape}")

    return X_seq, y_seq


def sweep_lstm(results):
    """Sweep LSTM over hidden units."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")

    print("\n" + "=" * 60)
    print("LSTM SWEEP: hidden_units")
    print("=" * 60)

    # Load raw sequenced data
    print("Loading LSTM sequenced data...")
    train_df = pd.read_csv(LSTM_TRAIN_CSV, low_memory=False)
    test_df = pd.read_csv(LSTM_TEST_CSV, low_memory=False)

    # Split training into train/val by Computer
    computers = train_df["Computer"].unique()
    n_val = max(1, int(len(computers) * 0.2))
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(computers)
    val_computers = computers[:n_val]
    train_computers = computers[n_val:]

    df_train = train_df[train_df["Computer"].isin(train_computers)].copy()
    df_val = train_df[train_df["Computer"].isin(val_computers)].copy()

    print(f"  Train computers: {len(train_computers)} | Val computers: {len(val_computers)}")
    print(f"  Train rows: {len(df_train):,} | Val rows: {len(df_val):,}")

    # Encode + scale (fitted on train only)
    df_train_enc, test_df_enc = encode_and_scale_lstm(df_train, test_df, FEATURE_COLS)
    _, df_val_enc = encode_and_scale_lstm(df_train, df_val, FEATURE_COLS)

    # Build sequences once (shared across all unit configurations)
    print("\nBuilding training sequences...")
    X_train, y_train = build_sequences(df_train_enc, FEATURE_COLS, LSTM_WINDOW_SIZE)

    print("Building validation sequences...")
    X_val, y_val = build_sequences(df_val_enc, FEATURE_COLS, LSTM_WINDOW_SIZE)

    print("Building test sequences...")
    X_test, y_test = build_sequences(test_df_enc, FEATURE_COLS, LSTM_WINDOW_SIZE)

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)

    # Class weights
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes.astype(int), weights))
    print(f"  Class weights: {class_weights}")

    # Sweep over hidden units
    for units in LSTM_UNITS:
        print(f"\n  hidden_units = {units}")

        tf.keras.backend.clear_session()
        tf.random.set_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        # Architecture matching LSTMModel.build_model, varying first LSTM layer
        model = Sequential([
            LSTM(units, input_shape=(LSTM_WINDOW_SIZE, len(FEATURE_COLS)),
                 return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "recall"],
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=LSTM_PATIENCE,
            restore_best_weights=True,
        )

        # Train with early stopping (matching your actual training)
        tracker = EmissionsTracker(project_name=f"lstm_sweep_{units}", log_level="error")
        tracker.start()
        t0 = time.time()

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0,
        )

        train_time = time.time() - t0
        train_emissions = tracker.stop()

        actual_epochs = len(history.history["loss"])

        # Inference
        tracker = EmissionsTracker(project_name=f"lstm_infer_{units}", log_level="error")
        tracker.start()
        t0 = time.time()

        y_prob = model.predict(X_test, verbose=0)
        y_pred = (y_prob.flatten() >= 0.5).astype(int)

        inference_time = time.time() - t0
        inference_emissions = tracker.stop()

        f1, prec, rec, fpr = compute_metrics(y_test, y_pred)

        print(f"    F1: {f1:.4f} | Epochs: {actual_epochs} | "
              f"Train: {train_time:.1f}s ({train_emissions*1e6:.1f}µg) "
              f"| Infer: {inference_time:.1f}s ({inference_emissions*1e6:.1f}µg)")

        append_result(results, "LSTM", "hidden_units", units,
                      f1, prec, rec, fpr,
                      train_time, train_emissions,
                      inference_time, inference_emissions)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    results = []

    # Load and encode supervised data for RF and LightGBM sweeps
    X_train, y_train, X_test, y_test = load_supervised_data()
    X_train_enc, X_test_enc, encoders = encode_data(X_train, X_test)

    sweep_rf(X_train_enc, y_train, X_test_enc, y_test, results)
    sweep_lgbm(X_train_enc, y_train, X_test_enc, y_test, results)
    sweep_lstm(results)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{'=' * 60}")
    print(f"Results saved to {OUTPUT_CSV}")
    print(f"Total configurations tested: {len(results)}")
    print(f"{'=' * 60}")
    print("\nSummary:")
    print(df.to_string(index=False))