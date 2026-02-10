"""
Feature engineering pipeline for LMD-2022 lateral movement detection.

Covers:
  - Pipeline hyperparameters / constants
  - Section 0: optional stratified subset
  - Section 1: drop constants, identifiers, leak-prone features
  - Section 2: feature engineering (Image_Category, port binning, rare-category
    grouping, LogonId binarisation, ParentProcessId drop)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================================
# PIPELINE HYPERPARAMETERS
# ============================================================================

RANDOM_STATE = 42
LABEL_COL = "isMalicious"
GROUP_COL = "Computer"

# Set to None to use the full dataset; set to e.g. 50_000 for fast iteration
SUBSET_N_ROWS = 50_000

TEST_SIZE = 0.20

# Rare-category threshold: categories with fewer than this many rows → "Other"
RARE_CATEGORY_THRESHOLD = 50

# RF for feature importance ranking (fitted once for importance scores)
RF_SELECTOR_PARAMS = dict(
    n_estimators=100,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
    max_features="sqrt",
    min_samples_leaf=2,
)

# RF for feature-count validation (slightly stronger)
RF_EVAL_PARAMS = dict(
    n_estimators=150,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
    max_features="sqrt",
    min_samples_leaf=1,
)

# Original-feature counts to sweep (how many top features to keep)
FEATURE_COUNTS_TO_TEST = [3, 5, 7, 10, 12, 15, 20, 25, 30, 41]


def h(title: str):
    """Section header printer."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================================
# SECTION 0: OPTIONAL STRATIFIED SUBSET
# ============================================================================

def create_subset(lmd2022_df: pd.DataFrame,
                  subset_n_rows: int | None = SUBSET_N_ROWS) -> pd.DataFrame:
    """
    Optionally take a stratified subset of the full dataframe.

    Parameters
    ----------
    lmd2022_df : pd.DataFrame
        Full cleaned dataframe (must contain LABEL_COL and GROUP_COL).
    subset_n_rows : int or None
        If not None and less than total rows, take a stratified sample of
        this size. Use None for the full dataset.

    Returns
    -------
    pd.DataFrame
        The (possibly subsetted) dataframe, reset-indexed.
    """
    h("SECTION 0: LOAD DATA")

    assert LABEL_COL in lmd2022_df.columns, f"Missing label column: {LABEL_COL}"
    assert GROUP_COL in lmd2022_df.columns, f"Missing group column: {GROUP_COL}"


    if lmd2022_df[LABEL_COL].dtype == "object":
        lmd2022_df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors='coerce')
        lmd2022_df = df[df[LABEL_COL].notna()].copy()
        lmd2022_df[LABEL_COL] = lmd2022_df[LABEL_COL].astype(int)

    n_total = len(lmd2022_df)

    if subset_n_rows is not None and subset_n_rows < n_total:
        _, df = train_test_split(
            lmd2022_df,
            test_size=subset_n_rows,
            random_state=RANDOM_STATE,
            stratify=lmd2022_df[LABEL_COL],
        )
        df = df.reset_index(drop=True)
        print(f"Stratified subset: {len(df):,} rows from {n_total:,}")
    else:
        df = lmd2022_df.copy().reset_index(drop=True)
        print(f"Using full dataset: {len(df):,} rows")

    print(f"\nLabel distribution:")
    print(df[LABEL_COL].value_counts())
    print((df[LABEL_COL].value_counts(normalize=True) * 100).round(2))
    print(f"\nComputers: {df[GROUP_COL].nunique()}")

    return df


# ============================================================================
# SECTION 1: DROP CONSTANTS, IDENTIFIERS, AND LEAK-PRONE FEATURES
# ============================================================================

def drop_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    """
    Drop constant, identifier, leak-prone, and redundant features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe after optional subsetting (must contain LABEL_COL and GROUP_COL).

    Returns
    -------
    df_clean : pd.DataFrame
        Dataframe with problematic columns removed.
    all_drops : list[str]
        Names of all dropped columns.
    cols_before : int
        Number of feature columns before dropping (excludes label + group).
    """
    h("SECTION 1: FEATURE REMOVAL (STATISTICALLY JUSTIFIED)")

    # ── 1a. Constant / near-constant (>99.9% same value) ───────────────────
    near_zero_cols = []
    for col in df.columns:
        if col in [LABEL_COL, GROUP_COL]:
            continue
        top_freq = df[col].value_counts(dropna=False, normalize=True).iloc[0]
        if top_freq > 0.999:
            near_zero_cols.append(col)

    print(f"\n1a. Near-zero variance ({len(near_zero_cols)} features):")
    for c in sorted(near_zero_cols):
        print(f"    {c}")

    # ── 1b. Metadata / identifiers ─────────────────────────────────────────
    metadata_cols = [
        "SystemTime", "UtcTime", "CreationUtcTime", "PreviousCreationUtcTime",
        "Guid", "ProcessGuid", "SourceProcessGUID", "TargetProcessGUID",
        "ParentProcessGuid", "SourceProcessGuid", "TargetProcessGuid", "LogonGuid",
        "ProcessID", "ProcessId", "ThreadID", "Execution_ProcessID", "EventRecordID",
        "SourceProcessId", "SourceThreadId", "TargetProcessId",
        "User", "UserID",
        "Name", "Version", "StartFunction",
    ]

    print(f"\n1b. Metadata / identifiers ({len(metadata_cols)} features):")
    for c in sorted(metadata_cols):
        print(f"    {c}")

    # ── 1c. High-cardinality / leak risk ───────────────────────────────────
    leak_cols = [
        "CommandLine", "ParentCommandLine",
        "Hashes", "Hash",
        "CallTrace",
        "TargetFilename", "TargetObject",
        "Details", "Contents",
        "QueryResults",
        "ConfigurationFileHash",
    ]

    print(f"\n1c. High-cardinality leak risk ({len(leak_cols)} features):")
    for c in sorted(leak_cols):
        print(f"    {c}")

    # ── 1d. Redundant features ─────────────────────────────────────────────
    redundant_cols = ["Task"]
    print(f"\n1d. Redundant ({len(redundant_cols)} features): {redundant_cols}")

    # ── 1e. Redundant within signing group ─────────────────────────────────
    signing_redundant = ["SignatureStatus"]

    # ── 1f. Redundant IPv6 flags ───────────────────────────────────────────
    ipv6_redundant = ["DestinationIsIpv6"]

    print(f"\n1e–f. Redundant within groups ({len(signing_redundant) + len(ipv6_redundant)} features):")
    print(f"    Signing: {signing_redundant}")
    print(f"    IPv6:    {ipv6_redundant}")

    # ── Apply all drops ────────────────────────────────────────────────────
    all_drops = sorted(set(
        near_zero_cols + metadata_cols + leak_cols +
        redundant_cols + signing_redundant + ipv6_redundant
    ))
    all_drops = [c for c in all_drops if c in df.columns and c not in [LABEL_COL, GROUP_COL]]

    cols_before = len(df.columns) - 2  # exclude label + group
    df_clean = df.drop(columns=all_drops)
    cols_after = len(df_clean.columns) - 2

    print(f"\nFeatures before: {cols_before}")
    print(f"Features after:  {cols_after}")
    print(f"Total dropped:   {len(all_drops)}")

    return df_clean, all_drops, cols_before


# ============================================================================
# SECTION 2: FEATURE ENGINEERING
# ============================================================================

# ── Image categorisation helpers ───────────────────────────────────────────
ATTACK_TOOLS = ["psexec", "mimikatz", "procdump", "atomicredteam",
                "wmic", "powershell", "cmd.exe"]
INFRA = ["splunk", "streamfwd"]


def categorise_image(path: str) -> str:
    """Categorise an executable image path into a security-relevant bucket."""
    s = str(path).lower()
    if any(t in s for t in ATTACK_TOOLS):
        return "AttackTool"
    if any(t in s for t in INFRA):
        return "Infra"
    if "\\windows\\system32\\" in s:
        return "System32"
    if "\\users\\" in s:
        return "UserProfile"
    if "\\temp\\" in s or "\\appdata\\local\\temp\\" in s:
        return "Temp"
    return "Other"


def bin_port(port_series: pd.Series) -> pd.Series:
    """Bin port numbers into security-relevant categories."""
    port = pd.to_numeric(port_series, errors="coerce").fillna(0).astype(int)
    return pd.cut(
        port,
        bins=[-1, 0, 1023, 49151, 65536],
        labels=["NotPopulated", "WellKnown", "Registered", "Ephemeral"],
    ).astype(str)


# High-cardinality categoricals to group rare values
HIGH_CARD_CATEGORICALS = [
    "ImageLoaded",
    "ParentImage",
    "TargetImage",
    "SourceImage",
    "CurrentDirectory",
    "QueryName",
]


def engineer_features(
    df_clean: pd.DataFrame,
    rare_threshold: int = RARE_CATEGORY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Apply all feature engineering transforms.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Dataframe after feature removal (must contain LABEL_COL and GROUP_COL).
    rare_threshold : int
        Categories with fewer than this many rows are grouped into "Other_Rare".

    Returns
    -------
    X : pd.DataFrame
        Engineered feature matrix (no label or group columns).
    y : pd.Series
        Binary label (0/1).
    groups : pd.Series
        Computer group identifiers.
    """
    h("SECTION 2: FEATURE ENGINEERING")

    y = df_clean[LABEL_COL].astype(int)
    groups = df_clean[GROUP_COL].astype(str)
    X = df_clean.drop(columns=[LABEL_COL, GROUP_COL], errors="ignore")

    # ── 2a. Image → Image_Category ─────────────────────────────────────────
    if "Image" in X.columns:
        X["Image_Category"] = X["Image"].map(categorise_image).astype(str)
        X = X.drop(columns=["Image"])
        print("✓ Image → Image_Category (6 categories)")

    # ── 2b. Port binning ───────────────────────────────────────────────────
    for port_col in ["SourcePort", "DestinationPort"]:
        if port_col in X.columns:
            X[port_col + "_Bin"] = bin_port(X[port_col])
            X = X.drop(columns=[port_col])
            print(f"✓ {port_col} → {port_col}_Bin (4 bins)")

    # ── 2c. Rare-category grouping ─────────────────────────────────────────
    for col in HIGH_CARD_CATEGORICALS:
        if col not in X.columns:
            continue
        vc = X[col].value_counts()
        rare_cats = vc[vc < rare_threshold].index
        n_before = X[col].nunique()
        X[col] = X[col].where(~X[col].isin(rare_cats), other="Other_Rare")
        n_after = X[col].nunique()
        print(f"✓ {col}: {n_before} → {n_after} categories (grouped {len(rare_cats)} rare)")

    # ── 2d. LogonId → binary populated flag ────────────────────────────────
    if "LogonId" in X.columns:
        X["LogonId_Populated"] = (X["LogonId"].astype(str) != "0").astype(int)
        X = X.drop(columns=["LogonId"])
        print("✓ LogonId → LogonId_Populated (binary)")

    # ── 2e. Drop ParentProcessId (identifier) ─────────────────────────────
    if "ParentProcessId" in X.columns:
        X = X.drop(columns=["ParentProcessId"])
        print("✓ Dropped ParentProcessId (identifier, 544 unique)")

    print(f"\nFinal X shape: {X.shape}")
    print(f"Categorical: {X.select_dtypes(exclude=['number']).shape[1]}")
    print(f"Numeric:     {X.select_dtypes(include=['number']).shape[1]}")

    return X, y, groups
