import os
import sys
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from codecarbon import EmissionsTracker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('..'))
from config import Config


class IsolationForestModel:

    def __init__(self, contamination, random_state: int = Config.RANDOM_STATE):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            max_samples=256,
            n_estimators=100
        )
        self.is_fitted = False
        self.label_encoders = {}
        self.metrics = {}
        self.training_emissions = None
        self.training_duration  = None
        self.inference_results  = []

    def _encode(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X_encoded = X.copy()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

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

        return X_encoded.values

    def train(self, X: pd.DataFrame) -> None:
        print("Encoding categorical features...")
        X_encoded = self._encode(X, fit=True)

        print(f"Training Isolation Forest on {X_encoded.shape[0]:,} normal samples...")

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
        self.training_duration  = end_time - start_time

        self.is_fitted = True
        print("✓ Training complete!")
        print(f"  Duration:  {self.training_duration:.2f}s")
        print(f"  Emissions: {self.training_emissions:.8f} kg CO₂")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        X_encoded = self._encode(X, fit=False)
        return self.model.predict(X_encoded)

    def predict_binary(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict(X)
        return np.where(raw == -1, 1, 0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        X_encoded = self._encode(X, fit=False)
        return -self.model.score_samples(X_encoded)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Quick evaluation with no tracking - sanity check only."""
        predictions = self.predict_binary(X_test)

        self.metrics = {
            'accuracy':  accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall':    recall_score(y_test, predictions, zero_division=0),
            'f1':        f1_score(y_test, predictions, zero_division=0)
        }

        return self.metrics

    def inference_with_metrics(self, X: pd.DataFrame, y_true: pd.Series) -> dict:
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

        predictions    = np.where(raw == -1, 1, 0)
        inference_time = end_time - start_time

        result = {
            'inference_time': inference_time,
            'emissions':      emissions,
            'f1':             f1_score(y_true, predictions, zero_division=0),
            'accuracy':       accuracy_score(y_true, predictions),
            'precision':      precision_score(y_true, predictions, zero_division=0),
            'recall':         recall_score(y_true, predictions, zero_division=0),
        }

        self.inference_results.append(result)
        return result

    def get_inference_summary(self) -> dict:
        """Call after all 10 runs to get mean/std across runs."""
        if not self.inference_results:
            print("No inference runs recorded yet.")
            return None

        keys = ['inference_time', 'emissions', 'f1', 'accuracy', 'precision', 'recall']
        summary = {}

        for key in keys:
            values = [r[key] for r in self.inference_results]
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std']  = np.std(values)

        return summary