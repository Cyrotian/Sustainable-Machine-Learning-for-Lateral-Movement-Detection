import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from codecarbon import EmissionsTracker


class LSTMModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.metrics = {}
        self.training_emissions = None
        self.inference_results = []

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

    def _scale(self, X, fit=False):
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def _reshape(self, X):
        """Reshape to (samples, timesteps, features) for LSTM"""
        return X.reshape((X.shape[0], 1, X.shape[1]))

    def build_model(self, input_shape):
        self.model = keras.Sequential([
            layers.LSTM(64, input_shape=input_shape, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, X_val, y_val, carbon_tracker):
        X_train_enc = self._encode(X_train, fit=True)
        X_val_enc   = self._encode(X_val,   fit=False)

        X_train_scaled = self._scale(X_train_enc, fit=True)
        X_val_scaled   = self._scale(X_val_enc,   fit=False)

        X_train_reshaped = self._reshape(X_train_scaled)
        X_val_reshaped   = self._reshape(X_val_scaled)

        input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
        self.build_model(input_shape)

        carbon_tracker.start()

        history = self.model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=self.config.get('epochs', 10),
            batch_size=self.config.get('batch_size', 512),
            verbose=1
        )

        self.training_emissions = carbon_tracker.stop()

        return history

    def predict(self, X):
        X_enc      = self._encode(X, fit=False)
        X_scaled   = self._scale(X_enc, fit=False)
        X_reshaped = self._reshape(X_scaled)

        probs = self.model.predict(X_reshaped, verbose=0)
        return (probs > 0.5).astype(int).flatten()

    def evaluate(self, X_test, y_test):
        """Quick evaluation - sanity check only, no tracking."""
        predictions = self.predict(X_test)

        self.metrics = {
            'accuracy':  accuracy_score(y_test, predictions),
            'f1':        f1_score(y_test, predictions, zero_division=0),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall':    recall_score(y_test, predictions, zero_division=0),
        }

        return self.metrics

    def inference_with_metrics(self, X, y_true):
        """Single inference run - call this 10 times for statistical reliability."""
        X_enc      = self._encode(X, fit=False)
        X_scaled   = self._scale(X_enc, fit=False)
        X_reshaped = self._reshape(X_scaled)

        tracker = EmissionsTracker(
            output_dir='emissions',
            project_name='lstm_inference',
            log_level='error'
        )

        tracker.start()
        start_time = time.perf_counter()

        probs = self.model.predict(X_reshaped, verbose=0)

        end_time = time.perf_counter()
        emissions = tracker.stop()

        predictions    = (probs > 0.5).astype(int).flatten()
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