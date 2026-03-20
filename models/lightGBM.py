from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from codecarbon import EmissionsTracker
import numpy as np
import time


class LightGBMModel:
    def __init__(self,n_estimators=100,learning_rate=0.05,num_leaves=31,max_depth=-1,random_state=42):
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced" 
        )

        self.label_encoders = {}
        self.metrics = {}
        self.training_emissions = None
        self.training_duration = None
        self.inference_results = []

    def _encode(self, X, fit=False):
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

    def train(self, X_train, y_train, X_val=None, y_val=None):
        print("Encoding categorical features...")
        X_encoded = self._encode(X_train, fit=True)

        tracker = EmissionsTracker(
            output_dir="emissions",
            project_name="lgbm_training",
            log_level="error"
        )

        tracker.start()
        start_time = time.perf_counter()


        if X_val is not None and y_val is not None:
            X_val_enc = self._encode(X_val, fit=False)
            self.model.fit(
                X_encoded, y_train,
                eval_set=[(X_val_enc, y_val)],
                eval_metric="binary_logloss",
                callbacks=[
                    # stops if no improvement for 50 rounds
                    __import__("lightgbm").early_stopping(50, verbose=False)
                ]
            )
        else:
            self.model.fit(X_encoded, y_train)

        end_time = time.perf_counter()
        self.training_emissions = tracker.stop()
        self.training_duration = end_time - start_time

        print("✓ Training complete!")
        print(f"  Duration:  {self.training_duration:.2f}s")
        print(f"  Emissions: {self.training_emissions:.8f} kg CO2")

    def predict(self, X):
        X_encoded = self._encode(X, fit=False)
        return self.model.predict(X_encoded)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)

        tn, fp, _, _ = confusion_matrix(y_test, preds).ravel()
        self.metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        }
        return self.metrics

    def inference_with_metrics(self, X, y_true):
        X_encoded = self._encode(X, fit=False)

        tracker = EmissionsTracker(
            output_dir="emissions",
            project_name="lgbm_inference",
            log_level="error"
        )

        tracker.start()
        start_time = time.perf_counter()

        preds = self.model.predict(X_encoded)

        end_time = time.perf_counter()
        emissions = tracker.stop()

        tn, fp, _, _ = confusion_matrix(y_true, preds).ravel()
        result = {
            "inference_time": end_time - start_time,
            "emissions": emissions,
            "f1": f1_score(y_true, preds, zero_division=0),
            "accuracy": accuracy_score(y_true, preds),
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        }

        self.inference_results.append(result)


        return result

    def get_inference_summary(self):
        if not self.inference_results:
            print("No inference runs recorded yet.")
            return None

        keys = ["inference_time", "emissions", "f1", "accuracy", "precision", "recall", "fpr"]
        summary = {}

        for key in keys:
            values = [r[key] for r in self.inference_results]
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))

        return summary
