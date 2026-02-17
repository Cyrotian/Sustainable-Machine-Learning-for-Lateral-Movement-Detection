"""
feature_engineering.py — Drop bad features + engineer new ones.

PURPOSE: Transform raw Sysmon log columns into a clean feature matrix.
         Two stages:
           Section 1: Remove features that would hurt the model
           Section 2: Engineer domain-specific features from raw columns

RESULT:  ~90 raw columns -> 5 final features
"""

import pandas as pd
import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

RANDOM_STATE = 42
LABEL_COL = "isMalicious"       # Target variable
GROUP_COL = "Computer"          # Group column (for group-based splitting)
SUBSET_N_ROWS = 50_000          # Stratified subset size for feature selection
RARE_CATEGORY_THRESHOLD = 50    # Categories with <50 rows become "Other_Rare"


def h(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================================
# SECTION 0: STRATIFIED SUBSET
# ============================================================================

def create_subset(df, subset_n_rows=SUBSET_N_ROWS):
    """
    Take a stratified sample of the dataset for feature selection.

    WHY NOT use the full dataset?
      With GroupShuffleSplit on only a few computers, the full dataset
      can produce validation splits where the held-out computer has
      completely different patterns. A 50k stratified subset gives a
      better-behaved split for ranking features.

    WHY STRATIFIED?
      A random subset might accidentally drop all malicious rows
      (since they're the minority class). Stratified sampling
      preserves the original class ratio.

    Set subset_n_rows=None to use the full dataset.
    """
    h("SECTION 0: STRATIFIED SUBSET")

    # Convert label to numeric if stored as string
    if df[LABEL_COL].dtype == "object":
        df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
        df = df[df[LABEL_COL].notna()].copy()
        df[LABEL_COL] = df[LABEL_COL].astype(int)

    total_rows = len(df)

    if subset_n_rows is not None and subset_n_rows < total_rows:
        from sklearn.model_selection import train_test_split

        # train_test_split with stratify keeps the class ratio intact
        # We throw away the first half and keep the subset
        _, subset = train_test_split(
            df,
            test_size=subset_n_rows,
            random_state=RANDOM_STATE,
            stratify=df[LABEL_COL],
        )
        df = subset.reset_index(drop=True)
        print("Stratified subset:", len(df), "rows from", total_rows)
    else:
        df = df.reset_index(drop=True)
        print("Using full dataset:", len(df), "rows")

    return df


# ============================================================================
# SECTION 1: DROP BAD FEATURES
# ============================================================================

def drop_features(df):
    """
    Remove features that would hurt model performance. Four categories:

    1a. NEAR-ZERO VARIANCE: >99.9% of values are the same.
        No discriminative power.

    1b. METADATA / IDENTIFIERS: Unique per-row values.
        Model would memorise specific log entries instead of learning patterns.

    1c. HIGH-CARDINALITY LEAK RISK: Free-text or hash fields.
        Too many unique values + risk of data leakage.

    1d. REDUNDANT: Information already captured by another feature.

    Returns: (cleaned_df, list_of_dropped_columns, original_column_count)
    """
    h("SECTION 1: FEATURE REMOVAL")

    # ── 1a. Find near-zero variance columns ────────────────────────────
    # If one value appears in >99.9% of rows, the column is useless.
    near_zero_cols = []
    for col in df.columns:
        if col == LABEL_COL or col == GROUP_COL:
            continue
        frequencies = df[col].value_counts(dropna=False, normalize=True)
        most_common_frequency = frequencies.iloc[0]
        if most_common_frequency > 0.999:
            near_zero_cols.append(col)

    print("1a. Near-zero variance columns:", len(near_zero_cols))

    # ── 1b. Metadata / identifiers (hardcoded list) ───────────────────
    # These are unique per-row or per-event — the model would just
    # memorise them instead of learning real patterns.
    metadata_cols = [
        "SystemTime", "UtcTime", "CreationUtcTime", "PreviousCreationUtcTime",
        "Guid", "ProcessGuid", "SourceProcessGUID", "TargetProcessGUID",
        "ParentProcessGuid", "SourceProcessGuid", "TargetProcessGuid", "LogonGuid",
        "ProcessID", "ProcessId", "ThreadID", "Execution_ProcessID", "EventRecordID",
        "SourceProcessId", "SourceThreadId", "TargetProcessId",
        "User", "UserID", "Name", "Version", "StartFunction",
    ]
    print("1b. Metadata / identifiers:", len(metadata_cols))

    # ── 1c. High-cardinality / leak risk (hardcoded list) ─────────────
    # Free-text fields with thousands of unique values.
    # CommandLine is especially dangerous — the model could learn
    # to detect specific commands rather than general attack patterns.
    leak_cols = [
        "CommandLine", "ParentCommandLine", "Hashes", "Hash", "CallTrace",
        "TargetFilename", "TargetObject", "Details", "Contents",
        "QueryResults", "ConfigurationFileHash",
    ]
    print("1c. High-cardinality leak risk:", len(leak_cols))

    # ── 1d. Redundant columns ─────────────────────────────────────────
    # Task = same info as EventID
    # SignatureStatus = same info as Signed
    # DestinationIsIpv6 = redundant with IP address format
    redundant_cols = ["Task", "SignatureStatus", "DestinationIsIpv6"]
    print("1d. Redundant:", len(redundant_cols))

    # ── Combine all columns to drop ───────────────────────────────────
    all_to_drop = near_zero_cols + metadata_cols + leak_cols + redundant_cols

    # Remove duplicates
    all_to_drop = list(set(all_to_drop))

    # Only drop columns that actually exist and aren't the label/group
    final_drops = []
    for col in all_to_drop:
        if col in df.columns and col != LABEL_COL and col != GROUP_COL:
            final_drops.append(col)

    cols_before = len(df.columns) - 2  # minus label and group
    df_clean = df.drop(columns=final_drops)
    cols_after = len(df_clean.columns) - 2

    print("\nFeatures before:", cols_before)
    print("Features after: ", cols_after)
    print("Total dropped:  ", len(final_drops))

    return df_clean, final_drops, cols_before


# ============================================================================
# SECTION 2: FEATURE ENGINEERING
# ============================================================================

# Known attack tools — used to categorise executable paths
ATTACK_TOOLS = ["psexec", "mimikatz", "procdump", "atomicredteam",
                "wmic", "powershell", "cmd.exe"]
INFRA_TOOLS = ["splunk", "streamfwd"]


def categorise_image(path):
    """
    Categorise an executable path into a security-relevant bucket.

    WHY: The raw "Image" column has thousands of unique file paths.
    A model can't learn from raw paths — but it CAN learn that
    "AttackTool" executables are suspicious.

    Categories:
      AttackTool   = known offensive tools (mimikatz, psexec, etc.)
      Infra        = monitoring infrastructure (splunk, etc.)
      System32     = standard Windows system executables
      UserProfile  = executables in user folders (unusual for system processes)
      Temp         = executables in temp folders (common for malware)
      Other        = everything else
    """
    s = str(path).lower()

    for tool in ATTACK_TOOLS:
        if tool in s:
            return "AttackTool"

    for tool in INFRA_TOOLS:
        if tool in s:
            return "Infra"

    if "\\windows\\system32\\" in s:
        return "System32"

    if "\\users\\" in s:
        return "UserProfile"

    if "\\temp\\" in s:
        return "Temp"

    return "Other"


def bin_port(port_series):
    """
    Bin raw port numbers (0-65535) into 4 security-meaningful categories.

    WHY: Raw port numbers have no ordinal meaning for ML.
    But the PORT RANGE tells you a lot:
      NotPopulated (0)        = no network activity for this event
      WellKnown (1-1023)      = standard services (HTTP=80, SMB=445, etc.)
      Registered (1024-49151) = application-specific ports
      Ephemeral (49152+)      = temporary client-side ports
    """
    port = pd.to_numeric(port_series, errors="coerce")
    port = port.fillna(0)
    port = port.astype(int)

    binned = pd.cut(
        port,
        bins=[-1, 0, 1023, 49151, 65536],
        labels=["NotPopulated", "WellKnown", "Registered", "Ephemeral"],
    )
    return binned.astype(str)


# Columns with too many unique categories to use directly
HIGH_CARD_CATEGORICALS = [
    "ImageLoaded",
    "ParentImage",
    "TargetImage",
    "SourceImage",
    "CurrentDirectory",
    "QueryName",
]


def engineer_features(df_clean, rare_threshold=RARE_CATEGORY_THRESHOLD):
    """
    Apply all feature engineering transforms.

    ALL transforms must run even though only 5 features are ultimately
    selected — because the feature SELECTION sweep (Sections 3-9)
    evaluates all ~35 features and needs them properly engineered
    to produce valid rankings and a valid sweep curve.

    Transforms:
      2a. Image path -> Image_Category (6 security buckets)
      2b. Port numbers -> 4 bins each
      2c. Rare category grouping for high-cardinality categoricals
      2d. LogonId -> binary populated flag
      2e. Drop ParentProcessId (identifier)

    Returns:
      X      = feature matrix (no label, no group column)
      y      = binary label series (0/1)
      groups = Computer ID series (for group-based splitting)
    """
    h("SECTION 2: FEATURE ENGINEERING")

    # Separate label and group from features
    y = df_clean[LABEL_COL].astype(int)
    groups = df_clean[GROUP_COL].astype(str)
    X = df_clean.drop(columns=[LABEL_COL, GROUP_COL], errors="ignore")

    # ── 2a. Image path -> 6 security categories ──────────────────────
    if "Image" in X.columns:
        X["Image_Category"] = X["Image"].apply(categorise_image)
        X["Image_Category"] = X["Image_Category"].astype(str)
        X = X.drop(columns=["Image"])
        print("Done: Image -> Image_Category (6 categories)")

    # ── 2b. Port numbers -> 4 bins each ──────────────────────────────
    if "SourcePort" in X.columns:
        X["SourcePort_Bin"] = bin_port(X["SourcePort"])
        X = X.drop(columns=["SourcePort"])
        print("Done: SourcePort -> SourcePort_Bin (4 bins)")

    if "DestinationPort" in X.columns:
        X["DestinationPort_Bin"] = bin_port(X["DestinationPort"])
        X = X.drop(columns=["DestinationPort"])
        print("Done: DestinationPort -> DestinationPort_Bin (4 bins)")

    # ── 2c. Group rare categories -> "Other_Rare" ────────────────────
    # For columns like QueryName with 5000+ unique DNS names,
    # group anything appearing <50 times into one "Other_Rare" bucket.
    for col in HIGH_CARD_CATEGORICALS:
        if col not in X.columns:
            continue

        counts = X[col].value_counts()
        rare_categories = counts[counts < rare_threshold].index
        X[col] = X[col].where(~X[col].isin(rare_categories), other="Other_Rare")
        print("Done:", col, "- grouped", len(rare_categories), "rare categories")

    # ── 2d. LogonId -> binary flag ────────────────────────────────────
    if "LogonId" in X.columns:
        is_populated = X["LogonId"].astype(str) != "0"
        X["LogonId_Populated"] = is_populated.astype(int)
        X = X.drop(columns=["LogonId"])
        print("Done: LogonId -> LogonId_Populated (binary)")

    # ── 2e. Drop ParentProcessId (it's just an identifier) ───────────
    if "ParentProcessId" in X.columns:
        X = X.drop(columns=["ParentProcessId"])
        print("Done: Dropped ParentProcessId (identifier)")

    # ── Summary ───────────────────────────────────────────────────────
    num_categorical = X.select_dtypes(exclude=["number"]).shape[1]
    num_numeric = X.select_dtypes(include=["number"]).shape[1]

    print("\nFinal X shape:", X.shape)
    print("Categorical columns:", num_categorical)
    print("Numeric columns:    ", num_numeric)

    return X, y, groups
