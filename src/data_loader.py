"""Load and prepare the raw LMD-2022 dataset."""

import pandas as pd


def load_lmd2022(csv_path: str) -> pd.DataFrame:
    """
    Load LMD-2022 CSV, drop 'Unnamed' columns, rename Label → isMalicious,
    and keep only rows where isMalicious ∈ {0, 1}.

    Parameters
    ----------
    csv_path : str
        Path to the LMD-2022 CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with 'isMalicious' as the label column.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    useful_cols = [col for col in df.columns if not col.startswith("Unnamed")]
    print(f"Useful columns (excluding Unnamed): {len(useful_cols)}")

    lmd2022_df = df[useful_cols].copy()
    lmd2022_df.rename(columns={"Label": "isMalicious"}, inplace=True)
    lmd2022_df = lmd2022_df[
        (lmd2022_df["isMalicious"] == "1") | (lmd2022_df["isMalicious"] == "0")
    ].copy()

    print(f"Rows after filtering valid labels: {len(lmd2022_df):,}")
    print(f"Columns: {len(lmd2022_df.columns)}")
    print(f"\nLabel distribution:")
    print(lmd2022_df["isMalicious"].value_counts())
    print(lmd2022_df["isMalicious"].value_counts(normalize=True) * 100)

    return lmd2022_df
