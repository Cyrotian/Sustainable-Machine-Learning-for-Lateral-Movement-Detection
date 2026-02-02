import os
from dataclasses import dataclass

@dataclass
class Config:
    # Paths
    LMD_2022_PATH = "../data/lmd2022/LMD-2022_NormalVsMalicious.csv"
    LMD_2023_PATH = "../data/lmd2023/test_set.csv"
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_INFERENCE_RUNS = 10
    
    # Random Forest
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = None
    
    # Isolation Forest
    IF_CONTAMINATION = 0.35  # Based on your 35% normal, 65% malicious split
    
    # LSTM
    LSTM_UNITS = 64
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32
    
    # Carbon tracking
    UK_REGION = "GB"
    CARBON_API_URL = "https://api.carbonintensity.org.uk/intensity"