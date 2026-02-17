"""
save_metrics.py — Log model performance metrics to a shared CSV.

Saves F1, precision, recall (mean + std) from each model's 10 inference runs
to emissions/model_metrics.csv. Charts are generated separately by plot_all.py.

USAGE in main.py (after each model's inference loop):

    from save_metrics import save_model_metrics

    save_model_metrics("Random Forest", rf_model.get_inference_summary())
    save_model_metrics("LightGBM", lightgbm_model.get_inference_summary())
    save_model_metrics("LSTM", lstm_model.get_inference_summary())
"""

import pandas as pd
import os

METRICS_FILE = "emissions/model_metrics.csv"


def save_model_metrics(model_name, summary_dict):
    """
    Save a model's inference summary to the shared metrics CSV.

    summary_dict comes from model.get_inference_summary() and contains
    keys like f1_mean, f1_std, precision_mean, precision_std, etc.

    If the model already exists in the CSV, it gets replaced.
    """
    row = {"model": model_name}
    row.update(summary_dict)

    df_new = pd.DataFrame([row])

    if os.path.exists(METRICS_FILE):
        df_existing = pd.read_csv(METRICS_FILE)
        df_existing = df_existing[df_existing["model"] != model_name]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        os.makedirs("emissions", exist_ok=True)
        df_out = df_new

    df_out.to_csv(METRICS_FILE, index=False)
    print(f"  Saved {model_name} metrics to {METRICS_FILE}")