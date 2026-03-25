from lightgbm import LGBMClassifier, early_stopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from codecarbon import EmissionsTracker
import numpy as np
import time


class LightGBMModel:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=42,
    ):
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

        self.label_encoders = {}
        self.metrics = {}
        self.training_emissions = None
        self.training_duration = None
        self.inference_results = []

    # =========================================================================
    # ENCODING HELPERS
    # =========================================================================

    def _get_categorical_columns(self, dataframe):
        """Return the categorical column names."""
        return dataframe.select_dtypes(include=["object"]).columns.tolist()

    def _replace_unknown_category(self, value, known_classes, fallback_class):
        """Replace unseen categories with a safe fallback value."""
        if value in known_classes:
            return value
        return fallback_class

    def _fit_and_encode_column(self, dataframe, column_name):
        """Fit a LabelEncoder on one training column and transform it."""
        label_encoder = LabelEncoder()

        dataframe[column_name] = label_encoder.fit_transform(
            dataframe[column_name].astype(str)
        )

        self.label_encoders[column_name] = label_encoder

    def _transform_column_with_existing_encoder(self, dataframe, column_name):
        """Transform one column using a previously fitted LabelEncoder."""
        label_encoder = self.label_encoders[column_name]
        known_classes = set(label_encoder.classes_)
        fallback_class = label_encoder.classes_[0]

        dataframe[column_name] = dataframe[column_name].astype(str).apply(
            self._replace_unknown_category,
            args=(known_classes, fallback_class),
        )

        dataframe[column_name] = label_encoder.transform(dataframe[column_name])

    def _encode(self, X, fit=False):
        """
        Encode categorical columns.

        If fit=True:
        - fit encoders on the data
        - store them for later use

        If fit=False:
        - reuse stored encoders
        - map unseen categories to the first known class
        """
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

    # =========================================================================
    # METRIC HELPERS
    # =========================================================================

    def _calculate_false_positive_rate(self, y_true, predictions):
        """Calculate the false positive rate."""
        tn, fp, _, _ = confusion_matrix(y_true, predictions).ravel()

        if (fp + tn) > 0:
            return float(fp / (fp + tn))

        return 0.0

    def _calculate_metrics(self, y_true, predictions):
        """Calculate standard classification metrics."""
        false_positive_rate = self._calculate_false_positive_rate(y_true, predictions)

        return {
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision_score(y_true, predictions, zero_division=0),
            "recall": recall_score(y_true, predictions, zero_division=0),
            "f1": f1_score(y_true, predictions, zero_division=0),
            "fpr": false_positive_rate,
        }

    # =========================================================================
    # TRAINING HELPERS
    # =========================================================================

    def _fit_without_validation(self, X_train_encoded, y_train):
        """Train LightGBM without a validation set."""
        self.model.fit(X_train_encoded, y_train)

    def _fit_with_validation(self, X_train_encoded, y_train, X_val, y_val):
        """Train LightGBM with validation and early stopping."""
        X_validation_encoded = self._encode(X_val, fit=False)

        self.model.fit(
            X_train_encoded,
            y_train,
            eval_set=[(X_validation_encoded, y_val)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(50, verbose=False)],
        )

    # =========================================================================
    # TRAINING
    # =========================================================================

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the LightGBM model and track training emissions."""
        print("Encoding categorical features...")
        X_train_encoded = self._encode(X_train, fit=True)

        emissions_tracker = EmissionsTracker(
            output_dir="emissions",
            project_name="lgbm_training",
            log_level="error",
        )

        emissions_tracker.start()
        start_time = time.perf_counter()

        if X_val is not None and y_val is not None:
            self._fit_with_validation(X_train_encoded, y_train, X_val, y_val)
        else:
            self._fit_without_validation(X_train_encoded, y_train)

        end_time = time.perf_counter()

        self.training_emissions = emissions_tracker.stop()
        self.training_duration = end_time - start_time

        print("✓ Training complete!")
        print(f"  Duration:  {self.training_duration:.2f}s")
        print(f"  Emissions: {self.training_emissions:.8f} kg CO2")

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict(self, X):
        """Generate predictions for new data."""
        encoded_X = self._encode(X, fit=False)
        return self.model.predict(encoded_X)

    # =========================================================================
    # EVALUATION
    # =========================================================================

    def evaluate(self, X_test, y_test):
        """Run a quick evaluation without emissions tracking."""
        predictions = self.predict(X_test)
        self.metrics = self._calculate_metrics(y_test, predictions)
        return self.metrics

    def inference_with_metrics(self, X, y_true):
        """Run one tracked inference pass and return the results."""
        encoded_X = self._encode(X, fit=False)

        emissions_tracker = EmissionsTracker(
            output_dir="emissions",
            project_name="lgbm_inference",
            log_level="error",
        )

        emissions_tracker.start()
        start_time = time.perf_counter()

        predictions = self.model.predict(encoded_X)

        end_time = time.perf_counter()
        emissions = emissions_tracker.stop()

        metric_results = self._calculate_metrics(y_true, predictions)

        result = {
            "inference_time": end_time - start_time,
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
        """Return mean and standard deviation across all recorded inference runs."""
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
            summary[f"{metric_name}_mean"] = float(np.mean(metric_values))
            summary[f"{metric_name}_std"] = float(np.std(metric_values))

        return summary