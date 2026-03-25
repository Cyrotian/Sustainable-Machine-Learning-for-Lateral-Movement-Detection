#LSTM model with temporal sliding windows.

import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from codecarbon import EmissionsTracker
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


WINDOW_SIZE = 10


class LSTMModel:
    def __init__(self, config):
        self.config = config
        self.window_size = config.get("window_size", WINDOW_SIZE)
        self.model = None
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.metrics = {}
        self.training_emissions = None
        self.inference_results = []

    def _encode(self, X, fit=False):
        """LabelEncode categorical columns to numeric."""
        X_encoded = X.copy()
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                known = set(le.classes_)
                X_encoded[col] = X[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                X_encoded[col] = le.transform(X_encoded[col])

        return X_encoded

    def _scale(self, X, fit=False):
        """MinMax scale all features to [0, 1]."""
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def create_sequences(self, df, feature_cols, label_col="isMalicious",
                         group_col="Computer"):
        """Create sliding windows of events for LSTM input."""
        X_windows = []
        y_windows = []

        computers = df[group_col].unique()
        print("  Creating sequences from", len(computers), "computers,",
              "window_size=" + str(self.window_size))

        for comp in computers:
            comp_data = df[df[group_col] == comp]

            features = comp_data[feature_cols].values
            labels = comp_data[label_col].values

            # Slide window across this computer's events
            for i in range(len(features) - self.window_size + 1):
                window = features[i : i + self.window_size]
                label = labels[i + self.window_size - 1]
                X_windows.append(window)
                y_windows.append(label)

        X_seq = np.array(X_windows)
        y_seq = np.array(y_windows)

        print("  Sequences created:", X_seq.shape[0],
              "| Shape:", X_seq.shape)

        return X_seq, y_seq

    def build_model(self, input_shape):
        self.model = keras.Sequential([
            layers.LSTM(64, input_shape=input_shape, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid")
        ])

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "recall"]
        )

    def train(self, df_train, df_val, feature_cols, carbon_tracker,
              label_col="isMalicious", group_col="Computer"):
        """
        Train LSTM on sequenced data.
        """
        # Encode categoricals
        print("Encoding features...")
        df_train_enc = self._encode(df_train[feature_cols], fit=True)
        df_val_enc = self._encode(df_val[feature_cols], fit=False)

        # Scale
        print("Scaling features...")
        train_scaled = self._scale(df_train_enc, fit=True)
        val_scaled = self._scale(df_val_enc, fit=False)

        # Rebuild DataFrames with scaled values + metadata columns
        df_train_processed = pd.DataFrame(train_scaled, columns=feature_cols)
        df_train_processed[label_col] = df_train[label_col].values
        df_train_processed[group_col] = df_train[group_col].values

        df_val_processed = pd.DataFrame(val_scaled, columns=feature_cols)
        df_val_processed[label_col] = df_val[label_col].values
        df_val_processed[group_col] = df_val[group_col].values

        # Create sliding windows
        print("\nCreating training sequences...")
        X_train, y_train = self.create_sequences(
            df_train_processed, feature_cols, label_col, group_col)

        print("Creating validation sequences...")
        X_val, y_val = self.create_sequences(
            df_val_processed, feature_cols, label_col, group_col)

        # Build and train
        input_shape = (self.window_size, len(feature_cols))
        self.build_model(input_shape)

        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        print("  Class weights:", class_weights)

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        carbon_tracker.start()

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.get("epochs", 10),
            batch_size=self.config.get("batch_size", ),
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=1
        )

        self.training_emissions = carbon_tracker.stop()

        return history

    def predict_from_sequences(self, X_seq):
        """Predict on pre-built sequences."""
        probs = self.model.predict(X_seq, verbose=0)
        return (probs > 0.5).astype(int).flatten()

    def evaluate(self, df_test, feature_cols, label_col="isMalicious",
                 group_col="Computer"):  
        # Encode + scale
        df_test_enc = self._encode(df_test[feature_cols], fit=False)
        test_scaled = self._scale(df_test_enc, fit=False)

        df_test_processed = pd.DataFrame(test_scaled, columns=feature_cols)
        df_test_processed[label_col] = df_test[label_col].values
        df_test_processed[group_col] = df_test[group_col].values

        X_test, y_test = self.create_sequences(
            df_test_processed, feature_cols, label_col, group_col)

        predictions = self.predict_from_sequences(X_test)

        tn, fp, _, _ = confusion_matrix(y_test, predictions).ravel()
        self.metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        }

        return self.metrics

    def inference_with_metrics(self, X_seq, y_seq):
        """
        Single tracked inference run on pre-built sequences.
        Call this 10 times for statistical reliability.

        Pass pre-built X_seq and y_seq to avoid re-creating sequences
        each run (sequence creation is preprocessing, not inference).
        """
        tracker = EmissionsTracker(
            output_dir="emissions",
            project_name="lstm_inference",
            log_level="error"
        )

        tracker.start()
        start_time = time.perf_counter()

        probs = self.model.predict(X_seq, verbose=0)

        end_time = time.perf_counter()
        emissions = tracker.stop()

        predictions = (probs > 0.5).astype(int).flatten()
        inference_time = end_time - start_time

        tn, fp, _, _ = confusion_matrix(y_seq, predictions).ravel()
        result = {
            "inference_time": inference_time,
            "emissions": emissions,
            "f1": f1_score(y_seq, predictions, zero_division=0),
            "accuracy": accuracy_score(y_seq, predictions),
            "precision": precision_score(y_seq, predictions, zero_division=0),
            "recall": recall_score(y_seq, predictions, zero_division=0),
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        }

        self.inference_results.append(result)
        return result

    def get_inference_summary(self):
        """Call after all 10 runs to get mean/std across runs."""
        if not self.inference_results:
            print("No inference runs recorded yet.")
            return None

        keys = ["inference_time", "emissions", "f1", "accuracy", "precision", "recall", "fpr"]
        summary = {}

        for key in keys:
            values = [r[key] for r in self.inference_results]
            summary[key + "_mean"] = np.mean(values)
            summary[key + "_std"] = np.std(values)

        return summary