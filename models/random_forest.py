from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import numpy as np
import time


class RandomForestModel:
    def __init__(self, n_estimators=200, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.label_encoders = {}
        self.metrics = {}
        self.training_emissions = None
        self.training_duration  = None
        self.training_energy    = None
        self.inference_results  = []

    def _encode(self, X, fit=False):
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

        return X_encoded

    def train(self, X_train, y_train):
        print("Encoding categorical features...")
        X_encoded = self._encode(X_train, fit=True)

        print(f"Training Random Forest on {X_encoded.shape[0]:,} samples...")

        tracker = EmissionsTracker(
            output_dir='emissions',
            project_name='rf_training',
            log_level='error'
        )

        tracker.start()
        start_time = time.perf_counter()

        self.model.fit(X_encoded, y_train)

        end_time = time.perf_counter()
        self.training_emissions = tracker.stop()
        self.training_duration  = end_time - start_time

        print("✓ Training complete!")
        print(f"  Duration:  {self.training_duration:.2f}s")
        print(f"  Emissions: {self.training_emissions:.8f} kg CO2")

    def predict(self, X):
        X_encoded = self._encode(X, fit=False)
        return self.model.predict(X_encoded)

    def evaluate(self, X_test, y_test):
        """Quick evaluation with no tracking - sanity check only."""
        predictions = self.predict(X_test)

        self.metrics = {
            'accuracy':  accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall':    recall_score(y_test, predictions, zero_division=0),
            'f1':        f1_score(y_test, predictions, zero_division=0)
        }

        return self.metrics

    def inference_with_metrics(self, X, y_true):
        """Single tracked inference run - call this 10 times for statistical reliability."""
        X_encoded = self._encode(X, fit=False)

        tracker = EmissionsTracker(
            output_dir='emissions',
            project_name='rf_inference',
            log_level='error'
        )

        tracker.start()
        start_time = time.perf_counter()

        predictions = self.model.predict(X_encoded)

        end_time = time.perf_counter()
        emissions = tracker.stop()

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

    def get_inference_summary(self):
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

    def plot_decision_tree(self, X_train, output_path='figures/rf_decision_tree.png'):
        """
        Plot the first estimator from the ensemble at max_depth=3.
        Call after train() - uses stored encoders to get correct feature names.
        """
        import os
        os.makedirs('figures', exist_ok=True)

        X_encoded = self._encode(X_train, fit=False)
        feature_names = X_encoded.columns.tolist()
        tree = self.model.estimators_[0]

        fig, ax = plt.subplots(figsize=(20, 8))

        plot_tree(
            tree,
            feature_names=feature_names,
            class_names=['Normal', 'Malicious'],
            max_depth=3,
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax
        )

        ax.set_title(
            'Random Forest: Single Estimator Decision Tree (max depth = 3)',
            fontsize=14, fontweight='bold', pad=15
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Decision tree saved to {output_path}")