import os

from pathlib import Path

class Config:

    CONFIG_DIR = Path(__file__).parent.resolve(strict=True)  # .../src/
    PROJECT_ROOT = Path("/home/mike/Documents/Sustainable-Machine-Learning-for-Lateral-Movement-Detection")

    
    

    DATA_DIR = PROJECT_ROOT / "data"
    LMD_2022_DIR = DATA_DIR / "lmd2022"
    LMD_2023_DIR = DATA_DIR / "lmd2023"
    
   
    LMD_2022_PATH = str(LMD_2022_DIR / "LMD-2022_NormalVsMalicious.csv")
    LMD_2022_NORMAL_PATH = str(LMD_2022_DIR / "LMD-2022_Normal.csv")
    LMD_2023_EORS_PATH = str(LMD_2023_DIR / "LMD-2023_EoRS.csv")
    LMD_2023_NORMAL_PATH = str(LMD_2023_DIR / "LMD-2023_Normal.csv")
    LMD_2023_TEST_PATH = str(LMD_2023_DIR / "lmd2023_test_file.csv")
    
    LMD_2023_TEST_FULL_PATH = str(LMD_2023_DIR / "LMD-2023_IF_TEST.csv")
    OUTPUT_DIR = str(PROJECT_ROOT / "final_datasets")

    
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_INFERENCE_RUNS = 10
    
    # Random Forest
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = None
    
    # Isolation Forest
    IF_CONTAMINATION = 0.35


    # LSTM
    LSTM_UNITS = 64
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32


    EMMISSIONS_OUTPUT_DIR = str(PROJECT_ROOT / "emissions")
    
    # Carbon tracking
    UK_REGION = "GB"
    CARBON_API_URL = "https://api.carbonintensity.org.uk/intensity"