import os
import sys
from sklearn.ensemble import IsolationForest
import numpy as np
from typing import Tuple
sys.path.insert(0, os.path.abspath('..'))
from config import Config

class IsolationForestModel:
    """Isolation Forest for anomaly detection."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = Config.RANDOM_STATE):
        """
        Initialize the Isolation Forest model.
        
        Args:
            contamination: Expected proportion of anomalies in the dataset
            random_state: Random seed for reproducibility
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.is_fitted = False
    
    def train(self, X: np.ndarray) -> None:
        """Train the model on normal data."""
        self.model.fit(X)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Returns:
            Array of predictions (1 for normal, -1 for anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        return -self.model.score_samples(X)