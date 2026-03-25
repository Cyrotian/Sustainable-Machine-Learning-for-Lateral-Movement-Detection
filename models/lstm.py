# LSTM model with temporal sliding windows.

import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from codecarbon import EmissionsTracker
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


WINDOW_SIZE = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_THRESHOLD = 0.5


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

    # =========================================================================
    # ENCODING AND SCALING HELPERS
    # =========================================================================

    def _get_categorical_columns(self, dataframe):
        """Return the categorical column names."""
        return dataframe.select_dtypes(include=["object"]).columns.tolist()

    def _replace_unknown_category(self, value, known_classes, fallback_class):
        # Replace unseen categories with a fallback value
        if value in known_classes:
            return value
        return fallback_class

    def _fit_and_encode_column(self, dataframe, column_name):
        # Fit a LabelEncoder on one training column and transform it.
        label_encoder = LabelEncoder()

        dataframe[column_name] = label_encoder.fit_transform(
            dataframe[column_name].astype(str)
        )
        self.label_encoders[column_name] = label_encoder

    def _transform_column_with_existing_encoder(self, dataframe, column_name):
        # Transform one column using a stored LabelEncoder
        label_encoder = self.label_encoders[column_name]
        known_classes = set(label_encoder.classes_)
        fallback_class = label_encoder.classes_[0]

        dataframe[column_name] = dataframe[column_name].astype(str).apply(
            self._replace_unknown_category,
            args=(known_classes, fallback_class),
        )

        dataframe[column_name] = label_encoder.transform(dataframe[column_name])

    def _encode(self, X, fit=False):
        # Label-encode categorical columns
        encoded_dataframe = X.copy()
        categorical_columns = self._get_categorical_columns(X)

        for column_name in categorical_columns:
            if fit:
                self._fit_and_encode_column(encoded_dataframe, column_name)
            else:
                self._transform_column_with_existing_encoder(
                    encoded_dataframe,
                    column_name,
                )

        return encoded_dataframe

    def _scale(self, X, fit=False):
        # Scale all features to the range [0, 1]
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def _prepare_features(self, dataframe, feature_columns, fit=False):
        #Encode and scale the feature columns.
        encoded_features = self._encode(dataframe[feature_columns], fit=fit)
        scaled_features = self._scale(encoded_features, fit=fit)
        return scaled_features

    def _build_processed_dataframe(
        self,
        scaled_features,
        original_dataframe,
        feature_columns,
        label_column,
        group_column,
    ):
        """
        Rebuild a dataframe containing scaled features plus label and group columns.
        """
        processed_dataframe = pd.DataFrame(scaled_features, columns=feature_columns)
        processed_dataframe[label_column] = original_dataframe[label_column].values
        processed_dataframe[group_column] = original_dataframe[group_column].values
        return processed_dataframe

    # =========================================================================
    # SEQUENCE HELPERS
    # =========================================================================

    def create_sequences(
        self,
        dataframe,
        feature_columns,
        label_col="isMalicious",
        group_col="Computer",
    ):
        # Create sliding windows for LSTM input
        sequence_features = []
        sequence_labels = []

        computer_ids = dataframe[group_col].unique()

        print(
            "  Creating sequences from",
            len(computer_ids),
            "computers,",
            "window_size=" + str(self.window_size),
        )

        for computer_id in computer_ids:
            computer_rows = dataframe[dataframe[group_col] == computer_id]

            feature_values = computer_rows[feature_columns].values
            label_values = computer_rows[label_col].values

            for start_index in range(len(feature_values) - self.window_size + 1):
                end_index = start_index + self.window_size

                window = feature_values[start_index:end_index]
                label = label_values[end_index - 1]

                sequence_features.append(window)
                sequence_labels.append(label)

        X_sequences = np.array(sequence_features)
        y_sequences = np.array(sequence_labels)

        print(
            "  Sequences created:",
            X_sequences.shape[0],
            "| Shape:",
            X_sequences.shape,
        )

        return X_sequences, y_sequences

    def _prepare_sequenced_dataset(
        self,
        dataframe,
        feature_columns,
        fit=False,
        label_col="isMalicious",
        group_col="Computer",
    ):
    
        # Encode, scale, rebuild, and convert a dataframe into LSTM sequences.
        
        scaled_features = self._prepare_features(
            dataframe,
            feature_columns,
            fit=fit,
        )

        processed_dataframe = self._build_processed_dataframe(
            scaled_features,
            dataframe,
            feature_columns,
            label_col,
            group_col,
        )

        return self.create_sequences(
            processed_dataframe,
            feature_columns,
            label_col,
            group_col,
        )

 
    # MODEL BUILDING
  

    def build_model(self, input_shape):
        # Build the LSTM network 
        hidden_units = self.config.get("hidden_units", 64)

        self.model = keras.Sequential(
            [
                layers.LSTM(
                    hidden_units,
                    input_shape=input_shape,
                    return_sequences=False,
                ),
                layers.Dropout(0.2),
                layers.Dense(32, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "recall"],
        )

    def _calculate_class_weights(self, y_train):
        # Calculate balanced class weights from the training labels
        unique_classes = np.unique(y_train)
        class_weight_values = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=y_train,
        )

        return dict(zip(unique_classes, class_weight_values))

    def _create_early_stopping_callback(self):
        """Create an EarlyStopping callback."""
        patience = self.config.get("patience", 3)

        return keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )

    # =========================================================================
    # METRIC HELPERS
    # =========================================================================

    def _calculate_false_positive_rate(self, y_true, predictions):
        # Calculate false positive rate 
        tn, fp, _, _ = confusion_matrix(y_true, predictions).ravel()

        if (fp + tn) > 0:
            return float(fp / (fp + tn))

        return 0.0

    def _calculate_metrics(self, y_true, predictions):
        # Calculate classification metrics
        false_positive_rate = self._calculate_false_positive_rate(y_true, predictions)

        return {
            "accuracy": accuracy_score(y_true, predictions),
            "f1": f1_score(y_true, predictions, zero_division=0),
            "precision": precision_score(y_true, predictions, zero_division=0),
            "recall": recall_score(y_true, predictions, zero_division=0),
            "fpr": false_positive_rate,
        }

    # =========================================================================
    # TRAINING
    # =========================================================================

    def train(
        self,
        df_train,
        df_val,
        feature_columns,
        carbon_tracker,
        label_col="isMalicious",
        group_col="Computer",
    ):
        # Train the LSTM on sequenced data
        print("Encoding and scaling training data...")
        X_train, y_train = self._prepare_sequenced_dataset(
            df_train,
            feature_columns,
            fit=True,
            label_col=label_col,
            group_col=group_col,
        )

        print("\nEncoding and scaling validation data...")
        X_val, y_val = self._prepare_sequenced_dataset(
            df_val,
            feature_columns,
            fit=False,
            label_col=label_col,
            group_col=group_col,
        )

        input_shape = (self.window_size, len(feature_columns))
        self.build_model(input_shape)

        class_weights = self._calculate_class_weights(y_train)
        print("  Class weights:", class_weights)

        early_stopping = self._create_early_stopping_callback()

        carbon_tracker.start()

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.get("epochs", DEFAULT_EPOCHS),
            batch_size=self.config.get("batch_size", DEFAULT_BATCH_SIZE),
            class_weight=class_weights,
            callbacks=[early_stopping],
            verbose=1,
        )

        self.training_emissions = carbon_tracker.stop()

        return history

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict_probabilities_from_sequences(self, X_sequences):
        # Predict probabilities from pre-built sequences
        return self.model.predict(X_sequences, verbose=0)

    def predict_from_sequences(self, X_sequences):
        # Predict binary labels from pre-built sequences."""
        threshold = self.config.get("threshold", DEFAULT_THRESHOLD)
        predicted_probabilities = self.predict_probabilities_from_sequences(X_sequences)
        predicted_labels = (predicted_probabilities >= threshold).astype(int).flatten()
        return predicted_labels

    # =========================================================================
    # EVALUATION
    # =========================================================================

    def evaluate(
        self,
        df_test,
        feature_columns,
        label_col="isMalicious",
        group_col="Computer",
    ):
        # Evaluate the LSTM on a test dataframe."""
        X_test, y_test = self._prepare_sequenced_dataset(
            df_test,
            feature_columns,
            fit=False,
            label_col=label_col,
            group_col=group_col,
        )

        predictions = self.predict_from_sequences(X_test)
        self.metrics = self._calculate_metrics(y_test, predictions)

        return self.metrics

    def inference_with_metrics(self, X_sequences, y_sequences):
        
        # Run one tracked inference pass on pre-built sequences.
        
        tracker = EmissionsTracker(
            output_dir="emissions",
            project_name="lstm_inference",
            log_level="error",
        )

        tracker.start()
        start_time = time.perf_counter()

        predicted_probabilities = self.predict_probabilities_from_sequences(X_sequences)

        end_time = time.perf_counter()
        emissions = tracker.stop()

        threshold = self.config.get("threshold", DEFAULT_THRESHOLD)
        predictions = (predicted_probabilities >= threshold).astype(int).flatten()
        inference_time = end_time - start_time

        metric_results = self._calculate_metrics(y_sequences, predictions)

        result = {
            "inference_time": inference_time,
            "emissions": emissions,
            "f1": metric_results["f1"],
            "accuracy": metric_results["accuracy"],
            "precision": metric_results["precision"],
            "recall": metric_results["recall"],
            "fpr": metric_results["fpr"],
        }

        self.inference_results.append(result)
        return result

    def get_inference_summary(self):
        # Return mean and standard deviation across all recorded inference runs."""
        if not self.inference_results:
            print("No inference runs recorded yet.")
            return None

        metric_names = [
            "inference_time",
            "emissions",
            "f1",
            "accuracy",
            "precision",
            "recall",
            "fpr",
        ]

        summary = {}

        for metric_name in metric_names:
            metric_values = [result[metric_name] for result in self.inference_results]
            summary[metric_name + "_mean"] = np.mean(metric_values)
            summary[metric_name + "_std"] = np.std(metric_values)

        return summary