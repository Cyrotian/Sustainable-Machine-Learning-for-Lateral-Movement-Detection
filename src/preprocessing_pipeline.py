"""
preprocessing_pipeline.py — Create final datasets for model training and testing.

This script:
1. Loads the raw LMD datasets
2. Builds the selected features
3. Applies a small number of feature transformations
4. Saves the final CSV files used by the models

Transforms applied:
- Image -> Image_Category
- SourceImage rare-category grouping
- Missing-value filling
- Optional retention of Computer and SystemTime for LSTM sequence building
"""

import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from config import Config


LABEL_COLUMN = "isMalicious"
GROUP_COLUMN = "Computer"
TIME_COLUMN = "SystemTime"

SELECTED_FEATURES = [
    "Image_Category",
    "GrantedAccess",
    "RuleName",
    "SourceImage",
    "Signature",
]

ATTACK_TOOLS = [
    "mimikatz",
    "mimidrv",
    "lazagne",
    "wcry",
    "wannacry",
    "smbexec",
    "psexec",
    "psexesvc",
    "powershell",
    "cmd.exe",
]

INFRA_TOOLS = ["splunk", "streamfwd"]

RARE_THRESHOLD = 50


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def print_section(title):
    """Print a clear section heading."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_csv_file(csv_path):
    """Load a CSV file."""
    return pd.read_csv(csv_path, low_memory=False)


def remove_unnamed_columns(dataframe):
    """Remove columns whose names start with 'Unnamed'."""
    useful_columns = []

    for column_name in dataframe.columns:
        if not column_name.startswith("Unnamed"):
            useful_columns.append(column_name)

    return dataframe[useful_columns].copy()


def standardise_label_column(dataframe):
    """
    Rename Label to isMalicious and keep only valid numeric label rows.
    """
    cleaned_dataframe = dataframe.copy()

    if "Label" in cleaned_dataframe.columns:
        cleaned_dataframe.rename(columns={"Label": LABEL_COLUMN}, inplace=True)

    if LABEL_COLUMN in cleaned_dataframe.columns:
        cleaned_dataframe[LABEL_COLUMN] = pd.to_numeric(
            cleaned_dataframe[LABEL_COLUMN],
            errors="coerce",
        )
        cleaned_dataframe = cleaned_dataframe[cleaned_dataframe[LABEL_COLUMN].notna()].copy()
        cleaned_dataframe[LABEL_COLUMN] = cleaned_dataframe[LABEL_COLUMN].astype(int)

    return cleaned_dataframe


def load_and_clean_raw_dataset(csv_path):
    """Load a raw dataset and apply the standard cleanup steps."""
    print("\nLoading:", csv_path)

    dataframe = load_csv_file(csv_path)
    dataframe = remove_unnamed_columns(dataframe)
    dataframe = standardise_label_column(dataframe)

    print("  Raw rows:", len(dataframe))
    return dataframe


# ============================================================================
# FEATURE TRANSFORMS
# ============================================================================

def categorise_image(image_path):
    """Convert an executable path into a security-relevant category."""
    image_path_text = str(image_path).lower()

    for tool_name in ATTACK_TOOLS:
        if tool_name in image_path_text:
            return "AttackTool"

    for tool_name in INFRA_TOOLS:
        if tool_name in image_path_text:
            return "Infra"

    if "\\windows\\system32\\" in image_path_text:
        return "System32"

    if "\\users\\" in image_path_text:
        return "UserProfile"

    if "\\temp\\" in image_path_text:
        return "Temp"

    return "Other"


def create_image_category_if_needed(dataframe):
    """Create Image_Category from Image when it is not already present."""
    updated_dataframe = dataframe.copy()

    if "Image_Category" not in updated_dataframe.columns and "Image" in updated_dataframe.columns:
        updated_dataframe["Image_Category"] = updated_dataframe["Image"].apply(categorise_image)
        print("  Created Image_Category from Image column")

    return updated_dataframe


def select_required_features(dataframe, selected_features):
    """
    Create a new dataframe containing only the selected features plus the label.
    Missing features are filled with a default value.
    """
    result_dataframe = pd.DataFrame()

    for feature_name in selected_features:
        if feature_name in dataframe.columns:
            result_dataframe[feature_name] = dataframe[feature_name].values
        else:
            print(f"  WARNING: {feature_name} not found - filling with default")
            result_dataframe[feature_name] = "0"

    result_dataframe[LABEL_COLUMN] = dataframe[LABEL_COLUMN].values

    return result_dataframe


def group_rare_sourceimage_values(dataframe):
    """Replace rare SourceImage categories with 'Other_Rare'."""
    updated_dataframe = dataframe.copy()

    if "SourceImage" in updated_dataframe.columns:
        value_counts = updated_dataframe["SourceImage"].value_counts()
        rare_values = value_counts[value_counts < RARE_THRESHOLD].index

        updated_dataframe["SourceImage"] = updated_dataframe["SourceImage"].where(
            ~updated_dataframe["SourceImage"].isin(rare_values),
            other="Other_Rare",
        )

    return updated_dataframe


def fill_missing_values(dataframe):
    """Fill missing values using simple defaults."""
    filled_dataframe = dataframe.copy()

    for column_name in filled_dataframe.columns:
        if filled_dataframe[column_name].dtype == "object":
            filled_dataframe[column_name] = filled_dataframe[column_name].fillna("0")
        else:
            filled_dataframe[column_name] = filled_dataframe[column_name].fillna(0)

    return filled_dataframe


def filter_by_label_if_needed(dataframe, filter_label=None):
    """Optionally keep only rows with one label value."""
    if filter_label is None:
        return dataframe

    filtered_dataframe = dataframe[dataframe[LABEL_COLUMN] == filter_label].copy()
    print(f"  Filtered to label={filter_label}: {len(filtered_dataframe)} rows")

    return filtered_dataframe


# ============================================================================
# CORE DATASET BUILDING
# ============================================================================

def build_base_dataset(dataframe, selected_features, filter_label=None):
    """
    Apply the shared preprocessing steps used by all final datasets.
    """
    processed_dataframe = create_image_category_if_needed(dataframe)
    processed_dataframe = select_required_features(processed_dataframe, selected_features)
    processed_dataframe = group_rare_sourceimage_values(processed_dataframe)
    processed_dataframe = fill_missing_values(processed_dataframe)
    processed_dataframe = filter_by_label_if_needed(processed_dataframe, filter_label)

    print(
        "  Final:",
        len(processed_dataframe),
        "rows x",
        len(processed_dataframe.columns),
        "cols",
    )

    return processed_dataframe


def process_dataset(csv_path, selected_features, filter_label=None):
    """
    Load a dataset and apply the shared preprocessing steps.

    This version is used for the standard tabular model datasets.
    """
    raw_dataframe = load_and_clean_raw_dataset(csv_path)
    return build_base_dataset(raw_dataframe, selected_features, filter_label)


# ============================================================================
# LSTM-SPECIFIC DATASET BUILDING
# ============================================================================

def save_sequence_columns(dataframe):
    """Extract Computer and SystemTime if they exist."""
    saved_columns = {}

    if GROUP_COLUMN in dataframe.columns:
        saved_columns[GROUP_COLUMN] = dataframe[GROUP_COLUMN].copy()

    if TIME_COLUMN in dataframe.columns:
        saved_columns[TIME_COLUMN] = dataframe[TIME_COLUMN].copy()

    return saved_columns


def reattach_sequence_columns(dataframe, saved_columns):
    """Add back Computer and SystemTime after feature trimming."""
    updated_dataframe = dataframe.copy()

    if GROUP_COLUMN in saved_columns:
        updated_dataframe[GROUP_COLUMN] = saved_columns[GROUP_COLUMN].values[: len(updated_dataframe)]

    if TIME_COLUMN in saved_columns:
        updated_dataframe[TIME_COLUMN] = saved_columns[TIME_COLUMN].values[: len(updated_dataframe)]

    return updated_dataframe


def sort_for_lstm_sequence_building(dataframe):
    """Sort by Computer and SystemTime so LSTM windows are chronological."""
    sorted_dataframe = dataframe.copy()
    sort_columns = []

    if GROUP_COLUMN in sorted_dataframe.columns:
        sort_columns.append(GROUP_COLUMN)

    if TIME_COLUMN in sorted_dataframe.columns:
        sort_columns.append(TIME_COLUMN)

    if len(sort_columns) > 0:
        sorted_dataframe = sorted_dataframe.sort_values(sort_columns)
        sorted_dataframe = sorted_dataframe.reset_index(drop=True)

    return sorted_dataframe


def process_dataset_with_sequence_columns(csv_path, selected_features, filter_label=None):
    """
    Load a dataset, apply the shared preprocessing, and keep sequence columns.

    This version is used for LSTM datasets because the LSTM needs:
    - Computer: to keep events grouped by machine
    - SystemTime: to keep events in time order
    """
    print("\nLoading (with sequence columns):", csv_path)

    raw_dataframe = load_csv_file(csv_path)
    raw_dataframe = remove_unnamed_columns(raw_dataframe)
    raw_dataframe = standardise_label_column(raw_dataframe)

    saved_sequence_columns = save_sequence_columns(raw_dataframe)

    processed_dataframe = build_base_dataset(
        raw_dataframe,
        selected_features,
        filter_label=filter_label,
    )

    processed_dataframe = reattach_sequence_columns(
        processed_dataframe,
        saved_sequence_columns,
    )

    processed_dataframe = sort_for_lstm_sequence_building(processed_dataframe)

    print("  Final:", len(processed_dataframe), "rows")
    return processed_dataframe


# ============================================================================
# FEATURE LIST HELPERS
# ============================================================================

def load_selected_features(features_csv_path):
    """
    Load the selected feature list from CSV.
    Fall back to the hardcoded list if the file is missing.
    """
    if os.path.exists(features_csv_path):
        features_dataframe = pd.read_csv(features_csv_path)
        return features_dataframe["Feature"].tolist()

    print("WARNING: features CSV not found, using hardcoded list")
    return SELECTED_FEATURES


# ============================================================================
# MAIN
# ============================================================================

def create_all_datasets(features_csv_path="05_final_selected_features.csv"):
    """
    Create all final datasets used by the models.
    """
    final_features = load_selected_features(features_csv_path)

    output_folder = Path("final_datasets")
    output_folder.mkdir(exist_ok=True)

    print_section("CREATING ALL DATASETS")
    print("Selected features:", final_features)

    supervised_learning_dataframe = process_dataset(
        Config.LMD_2022_PATH,
        final_features,
        filter_label=None,
    )
    supervised_learning_dataframe.to_csv(
        output_folder / "supervised_learning_dataset.csv",
        index=False,
    )

    test_dataframe = process_dataset(
        Config.LMD_2023_TEST_PATH,
        final_features,
        filter_label=None,
    )
    test_dataframe.to_csv(
        output_folder / "lmd2023_test_dataset.csv",
        index=False,
    )

    lstm_train_dataframe = process_dataset_with_sequence_columns(
        Config.LMD_2022_PATH,
        final_features,
        filter_label=None,
    )
    lstm_train_dataframe.to_csv(
        output_folder / "lstm_train_sequenced.csv",
        index=False,
    )

    lstm_test_dataframe = process_dataset_with_sequence_columns(
        Config.LMD_2023_TEST_PATH,
        final_features,
        filter_label=None,
    )
    lstm_test_dataframe.to_csv(
        output_folder / "lstm_test_sequenced.csv",
        index=False,
    )

    print_section(f"ALL DATASETS CREATED IN {output_folder}")


if __name__ == "__main__":
    create_all_datasets()