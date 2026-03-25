"""
feature_engineering.py — Remove weak features and create better ones.

This script has two jobs:
1. Remove raw features that would hurt the model
2. Create more meaningful features from raw Sysmon columns

Result:
- starts with many raw Sysmon fields
- ends with a cleaner feature matrix for feature selection
"""

import pandas as pd
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
LABEL_COLUMN = "isMalicious"
GROUP_COLUMN = "Computer"
RARE_CATEGORY_THRESHOLD = 50

ATTACK_TOOLS = [
    "mimikatz",
    "mimidrv",
    "lazagne",
    "wcry",
    "wannacry",
    "smbexec",
    "psexec",
    "psexesvc",
]

INFRA_TOOLS = ["splunk", "streamfwd", "sysmon"]

HIGH_CARDINALITY_CATEGORICAL_COLUMNS = [
    "ImageLoaded",
    "ParentImage",
    "TargetImage",
    "SourceImage",
    "CurrentDirectory",
    "QueryName",
]

METADATA_COLUMNS = [
    "SystemTime",
    "UtcTime",
    "CreationUtcTime",
    "PreviousCreationUtcTime",
    "Guid",
    "ProcessGuid",
    "SourceProcessGUID",
    "TargetProcessGUID",
    "ParentProcessGuid",
    "SourceProcessGuid",
    "TargetProcessGuid",
    "LogonGuid",
    "ProcessID",
    "ProcessId",
    "ThreadID",
    "Execution_ProcessID",
    "EventRecordID",
    "SourceProcessId",
    "SourceThreadId",
    "TargetProcessId",
    "User",
    "UserID",
    "Name",
    "Version",
    "StartFunction",
]

LEAK_RISK_COLUMNS = [
    "CommandLine",
    "ParentCommandLine",
    "Hashes",
    "Hash",
    "CallTrace",
    "TargetFilename",
    "TargetObject",
    "Details",
    "Contents",
    "QueryResults",
    "ConfigurationFileHash",
]

REDUNDANT_COLUMNS = [
    "Task",
    "SignatureStatus",
    "DestinationIsIpv6",
]


# ============================================================================
# DISPLAY HELPER
# ============================================================================

def h(title):
    """Print a section heading."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================================
# SECTION 1: FEATURE REMOVAL
# ============================================================================

def find_near_zero_variance_columns(dataframe):
    """
    Find columns where one value appears in more than 99.9% of rows.
    """
    near_zero_variance_columns = []

    for column_name in dataframe.columns:
        if column_name == LABEL_COLUMN or column_name == GROUP_COLUMN:
            continue

        value_frequencies = dataframe[column_name].value_counts(
            dropna=False,
            normalize=True,
        )

        most_common_frequency = value_frequencies.iloc[0]

        if most_common_frequency > 0.999:
            near_zero_variance_columns.append(column_name)

    return near_zero_variance_columns


def combine_columns_to_drop(dataframe, near_zero_variance_columns):
    """
    Combine all feature-removal lists and keep only columns that actually exist.
    """
    all_candidate_columns = (
        near_zero_variance_columns
        + METADATA_COLUMNS
        + LEAK_RISK_COLUMNS
        + REDUNDANT_COLUMNS
    )

    unique_candidate_columns = list(set(all_candidate_columns))
    final_columns_to_drop = []

    for column_name in unique_candidate_columns:
        if column_name in dataframe.columns:
            if column_name != LABEL_COLUMN and column_name != GROUP_COLUMN:
                final_columns_to_drop.append(column_name)

    return final_columns_to_drop


def drop_features(dataframe):
    """
    Remove features that are likely to harm generalisation.

    Removed feature types:
    1. Near-zero variance columns
    2. Metadata and identifier columns
    3. High-cardinality leak-risk columns
    4. Redundant columns

    Returns:
    - cleaned dataframe
    - dropped column list
    - original feature count
    """
    h("SECTION 1: FEATURE REMOVAL")

    near_zero_variance_columns = find_near_zero_variance_columns(dataframe)

    print("1a. Near-zero variance columns:", len(near_zero_variance_columns))
    print("1b. Metadata / identifiers:", len(METADATA_COLUMNS))
    print("1c. High-cardinality leak risk:", len(LEAK_RISK_COLUMNS))
    print("1d. Redundant:", len(REDUNDANT_COLUMNS))

    final_columns_to_drop = combine_columns_to_drop(
        dataframe,
        near_zero_variance_columns,
    )

    original_feature_count = len(dataframe.columns) - 2
    cleaned_dataframe = dataframe.drop(columns=final_columns_to_drop)
    remaining_feature_count = len(cleaned_dataframe.columns) - 2

    print("\nFeatures before:", original_feature_count)
    print("Features after: ", remaining_feature_count)
    print("Total dropped:  ", len(final_columns_to_drop))

    return cleaned_dataframe, final_columns_to_drop, original_feature_count


# ============================================================================
# SECTION 2: FEATURE ENGINEERING HELPERS
# ============================================================================

def categorise_image(image_path):
    """
    Convert a raw executable path into a broader security-relevant category.
    """
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


def bin_port(port_series):
    """
    Convert raw port numbers into broader categories:
    - NotPopulated: 0 or missing
    - WellKnown: 1–1023
    - Registered: 1024–49151
    - Ephemeral: 49152–65536
    """
    port_numbers = pd.to_numeric(port_series, errors="coerce").fillna(0).astype(int)

    def classify_port(port_number):
        if port_number == 0:
            return "NotPopulated"
        elif 1 <= port_number <= 1023:
            return "WellKnown"
        elif 1024 <= port_number <= 49151:
            return "Registered"
        else:
            return "Ephemeral"

    return port_numbers.apply(classify_port)


def create_image_category(dataframe):
    """Replace Image with Image_Category."""
    updated_dataframe = dataframe.copy()

    if "Image" in updated_dataframe.columns:
        updated_dataframe["Image_Category"] = updated_dataframe["Image"].apply(categorise_image)
        updated_dataframe["Image_Category"] = updated_dataframe["Image_Category"].astype(str)
        updated_dataframe = updated_dataframe.drop(columns=["Image"])
        print("Done: Image -> Image_Category (6 categories)")

    return updated_dataframe


def create_port_bins(dataframe):
    """Replace raw port columns with port-range categories."""
    updated_dataframe = dataframe.copy()

    if "SourcePort" in updated_dataframe.columns:
        updated_dataframe["SourcePort_Bin"] = bin_port(updated_dataframe["SourcePort"])
        updated_dataframe = updated_dataframe.drop(columns=["SourcePort"])
        print("Done: SourcePort -> SourcePort_Bin (4 bins)")

    if "DestinationPort" in updated_dataframe.columns:
        updated_dataframe["DestinationPort_Bin"] = bin_port(updated_dataframe["DestinationPort"])
        updated_dataframe = updated_dataframe.drop(columns=["DestinationPort"])
        print("Done: DestinationPort -> DestinationPort_Bin (4 bins)")

    return updated_dataframe


def group_rare_categories(dataframe, rare_threshold):
    """
    Group rare values in high-cardinality categorical columns into Other_Rare.
    """
    updated_dataframe = dataframe.copy()

    for column_name in HIGH_CARDINALITY_CATEGORICAL_COLUMNS:
        if column_name not in updated_dataframe.columns:
            continue

        value_counts = updated_dataframe[column_name].value_counts()
        rare_categories = value_counts[value_counts < rare_threshold].index

        updated_dataframe[column_name] = updated_dataframe[column_name].where(
            ~updated_dataframe[column_name].isin(rare_categories),
            other="Other_Rare",
        )

        print(
            "Done:",
            column_name,
            "- grouped",
            len(rare_categories),
            "rare categories",
        )

    return updated_dataframe


def create_logonid_flag(dataframe):
    """Replace LogonId with a simple populated / not-populated flag."""
    updated_dataframe = dataframe.copy()

    if "LogonId" in updated_dataframe.columns:
        is_populated = updated_dataframe["LogonId"].astype(str) != "0"
        updated_dataframe["LogonId_Populated"] = is_populated.astype(int)
        updated_dataframe = updated_dataframe.drop(columns=["LogonId"])
        print("Done: LogonId -> LogonId_Populated (binary)")

    return updated_dataframe


def drop_parent_process_id(dataframe):
    """Drop ParentProcessId because it behaves like an identifier."""
    updated_dataframe = dataframe.copy()

    if "ParentProcessId" in updated_dataframe.columns:
        updated_dataframe = updated_dataframe.drop(columns=["ParentProcessId"])
        print("Done: Dropped ParentProcessId (identifier)")

    return updated_dataframe


def print_feature_summary(feature_dataframe):
    """Print a short summary of the final engineered feature matrix."""
    number_of_categorical_columns = feature_dataframe.select_dtypes(
        exclude=["number"]
    ).shape[1]

    number_of_numeric_columns = feature_dataframe.select_dtypes(
        include=["number"]
    ).shape[1]

    print("\nFinal X shape:", feature_dataframe.shape)
    print("Categorical columns:", number_of_categorical_columns)
    print("Numeric columns:    ", number_of_numeric_columns)


# ============================================================================
# SECTION 2: FEATURE ENGINEERING
# ============================================================================

def engineer_features(cleaned_dataframe, rare_threshold=RARE_CATEGORY_THRESHOLD):
    """
    Apply all feature-engineering steps.

    Transforms:
    1. Image -> Image_Category
    2. SourcePort -> SourcePort_Bin
    3. DestinationPort -> DestinationPort_Bin
    4. Group rare categories in selected high-cardinality columns
    5. LogonId -> LogonId_Populated
    6. Drop ParentProcessId

    Returns:
    - X: feature matrix
    - y: label vector
    - groups: computer IDs for group-based splitting
    """
    h("SECTION 2: FEATURE ENGINEERING")

    y = cleaned_dataframe[LABEL_COLUMN].astype(int)
    groups = cleaned_dataframe[GROUP_COLUMN].astype(str)

    X = cleaned_dataframe.drop(
        columns=[LABEL_COLUMN, GROUP_COLUMN],
        errors="ignore",
    )

    X = create_image_category(X)
    X = create_port_bins(X)
    X = group_rare_categories(X, rare_threshold)
    X = create_logonid_flag(X)
    X = drop_parent_process_id(X)

    print_feature_summary(X)

    return X, y, groups