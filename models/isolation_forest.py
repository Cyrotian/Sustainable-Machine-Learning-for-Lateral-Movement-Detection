import os
import sys
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from codecarbon import EmissionsTracker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('..'))
from config import Config


class IsolationForestModel:

    def __init__(self, contamination, random_state=Config.RANDOM_STATE):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            max_samples=1024,
            n_estimators=300,
        )
        self.is_fitted = False
        self.encoder = None
        self.categorical_cols = []
        self.numeric_cols = []
        self.metrics = {}
        self.training_emissions = None
        self.training_duration = None
        self.inference_results = []

    def _encode(self, X, fit=False):
        """
        OneHotEncode categorical features instead of LabelEncode.

        WHY: Isolation Forest measures distance to isolate anomalies.
        LabelEncoder creates fake numeric distances (e.g. System32=3, Other=4
        implies they're "close" when they're unrelated categories).
        OneHotEncoder creates binary columns so each category is equidistant.
        """
        X_copy = X.copy()

        if fit:
            self.categorical_cols = X_copy.select_dtypes(include=['object']).columns.tolist()
            self.numeric_cols = X_copy.select_dtypes(exclude=['object']).columns.tolist()

            self.encoder = OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
            )
            self.encoder.fit(X_copy[self.categorical_cols].astype(str))

        # Encode categoricals
        if len(self.categorical_cols) > 0:
            encoded = self.encoder.transform(X_copy[self.categorical_cols].astype(str))
        else:
            encoded = np.empty((len(X_copy), 0))

        # Get numeric columns
        if len(self.numeric_cols) > 0:
            numeric = X_copy[self.numeric_cols].values
        else:
            numeric = np.empty((len(X_copy), 0))

        # Combine
        result = np.hstack([encoded, numeric])
        return result

    def train(self, X):
        print("Encoding categorical features (OneHot)...")
        X_encoded = self._encode(X, fit=True)
        print("  Encoded shape:", X_encoded.shape)

        print("Training Isolation Forest on", X_encoded.shape[0], "normal samples...")

        tracker = EmissionsTracker(
            output_dir='emissions',
            project_name='if_training',
            log_level='error'
        )

        tracker.start()
        start_time = time.perf_counter()

        self.model.fit(X_encoded)

        end_time = time.perf_counter()
        self.training_emissions = tracker.stop()
        self.training_duration = end_time - start_time

        self.is_fitted = True
        print("Training complete!")
        print("  Duration: ", round(self.training_duration, 2), "s")
        print("  Emissions:", round(self.training_emissions, 8), "kg CO2")

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        X_encoded = self._encode(X, fit=False)
        return self.model.predict(X_encoded)

    def predict_binary(self, X):
        raw = self.predict(X)
        return np.where(raw == -1, 1, 0)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        X_encoded = self._encode(X, fit=False)
        return -self.model.score_samples(X_encoded)

    def evaluate(self, X_test, y_test):
        """Quick evaluation with no tracking - sanity check only."""
        predictions = self.predict_binary(X_test)

        self.metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1': f1_score(y_test, predictions, zero_division=0)
        }

        return self.metrics

    def inference_with_metrics(self, X, y_true):
        """Single tracked inference run - call this 10 times for statistical reliability."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")

        X_encoded = self._encode(X, fit=False)

        tracker = EmissionsTracker(
            output_dir='emissions',
            project_name='if_inference',
            log_level='error'
        )

        tracker.start()
        start_time = time.perf_counter()

        raw = self.model.predict(X_encoded)

        end_time = time.perf_counter()
        emissions = tracker.stop()

        predictions = np.where(raw == -1, 1, 0)
        inference_time = end_time - start_time

        result = {
            'inference_time': inference_time,
            'emissions': emissions,
            'f1': f1_score(y_true, predictions, zero_division=0),
            'accuracy': accuracy_score(y_true, predictions),
            'precision': precision_score(y_true, predictions, zero_division=0),
            'recall': recall_score(y_true, predictions, zero_division=0),
        }

        self.inference_results.append(result)
        return result

    def get_inference_summary(self):
        """Call after all 10 runs to get mean/std across runs."""
        if not self.inference_results:
            print("No inference runs recorded yet.")
            return None

        keys = ['inference_time', 'emissions', 'f1', 'accuracy', 'precision', 'recall']
        summary = {}

        for key in keys:
            values = [r[key] for r in self.inference_results]
            summary[key + '_mean'] = np.mean(values)
            summary[key + '_std'] = np.std(values)

        return summary