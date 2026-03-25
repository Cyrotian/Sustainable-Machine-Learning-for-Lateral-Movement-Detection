import pandas as pd
import os

METRICS_FILE = "emissions/model_metrics.csv"


def save_model_metrics(model_name, summary_dict):
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