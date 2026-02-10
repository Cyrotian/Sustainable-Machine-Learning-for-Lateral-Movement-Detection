"""
Feature selection pipeline for LMD-2022 lateral movement detection.

Covers:
  - Section 3: group-based train/validation split
  - Section 4: one-hot encoding + variance filter
  - Section 5: mutual information (train-only)
  - Section 6: RF importance ranking
  - Section 7: aggregate to original (pre-encoding) features
  - Section 8: feature-count sweep
  - Section 9: optimal feature count selection (Pareto)
"""

import time
import inspect

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, classification_report

from feature_engineering import (
    h,
    RANDOM_STATE,
    TEST_SIZE,
    RF_SELECTOR_PARAMS,
    RF_EVAL_PARAMS,
    FEATURE_COUNTS_TO_TEST,
)


# ============================================================================
# KNOWN ORIGINAL FEATURES (for encoded → original name mapping)
# ============================================================================

KNOWN_ORIGINAL_FEATURES = [
    "DestinationPort_Bin", "SourcePort_Bin", "Image_Category",
    "RuleName", "FileVersion", "Description", "Product", "Company",
    "OriginalFileName", "CurrentDirectory", "TerminalSessionId",
    "IntegrityLevel", "ParentImage", "ParentUser", "Protocol",
    "Initiated", "SourceIsIpv6", "SourceIp", "SourceHostname",
    "SourcePortName", "DestinationIp", "DestinationHostname",
    "DestinationPortName", "ImageLoaded", "Signed", "Signature",
    "SourceImage", "TargetImage", "GrantedAccess", "SourceUser",
    "TargetUser", "EventType", "QueryName", "IsExecutable",
    "Archived", "NewThreadId", "StartAddress", "StartModule",
]


def get_original_feature(encoded_name: str) -> str:
    """
    Map encoded column name back to original feature.

    Examples
    --------
    >>> get_original_feature("cat__Image_Category_System32")
    'Image_Category'
    >>> get_original_feature("num__EventID")
    'EventID'
    """
    if encoded_name.startswith("cat__"):
        rest = encoded_name[5:]
        for feat in sorted(KNOWN_ORIGINAL_FEATURES, key=len, reverse=True):
            if rest.startswith(feat + "_") or rest == feat:
                return feat
        return rest.rsplit("_", 1)[0] if "_" in rest else rest
    elif encoded_name.startswith("num__"):
        return encoded_name[5:]
    return encoded_name


# ============================================================================
# SECTION 3: GROUP-BASED TRAIN/VALIDATION SPLIT
# ============================================================================

def group_split(X, y, groups):
    """
    Split data using GroupShuffleSplit so validation contains unseen computers.

    Returns
    -------
    X_train, X_val, y_train, y_val, groups_train, groups_val
    """
    h("SECTION 3: GROUP-BASED TRAIN/VAL SPLIT")

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()
    groups_train = groups.iloc[train_idx]
    groups_val = groups.iloc[val_idx]

    print(f"Train: {len(X_train):,} samples | Val: {len(X_val):,} samples")
    print(f"Train computers: {sorted(groups_train.unique())}")
    print(f"Val computers:   {sorted(groups_val.unique())}")
    print(f"\nTrain label %:")
    print((y_train.value_counts(normalize=True) * 100).round(2))
    print(f"\nVal label %:")
    print((y_val.value_counts(normalize=True) * 100).round(2))

    return X_train, X_val, y_train, y_val, groups_train, groups_val


# ============================================================================
# SECTION 4: ONE-HOT ENCODING + VARIANCE FILTER
# ============================================================================

def encode_and_filter(X_train, X_val):
    """
    One-hot encode categorical columns and remove zero-variance encoded features.

    Returns
    -------
    X_train_var : array
        Encoded + filtered training matrix.
    X_val_var : array
        Encoded + filtered validation matrix.
    feature_names_var : np.ndarray
        Names of surviving encoded features.
    preprocess : ColumnTransformer
        Fitted encoder (for later reuse).
    """
    h("SECTION 4: ONE-HOT ENCODING + VARIANCE FILTER")

    cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    print(f"Numeric columns ({len(num_cols)}):     {num_cols}")

    # Version-safe OneHotEncoder kwargs
    OHE_KWARGS = dict(handle_unknown="ignore")
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        OHE_KWARGS["sparse_output"] = True
    else:
        OHE_KWARGS["sparse"] = True

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(**OHE_KWARGS), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    print("Fitting encoder on train...")
    X_train_enc = preprocess.fit_transform(X_train)
    X_val_enc = preprocess.transform(X_val)
    feature_names = preprocess.get_feature_names_out()

    # Remove constant columns created by encoding
    var_sel = VarianceThreshold(threshold=0.0)
    X_train_var = var_sel.fit_transform(X_train_enc)
    X_val_var = var_sel.transform(X_val_enc)
    feature_names_var = feature_names[var_sel.get_support()]

    print(f"Encoded features: {X_train_enc.shape[1]:,}")
    print(f"After variance filter: {X_train_var.shape[1]:,}")
    print(f"Removed: {X_train_enc.shape[1] - X_train_var.shape[1]:,} constant encoded columns")

    return X_train_var, X_val_var, feature_names_var, preprocess


# ============================================================================
# SECTION 5: MUTUAL INFORMATION (TRAIN-ONLY)
# ============================================================================

def compute_mutual_information(X_train_var, y_train, feature_names_var,
                               output_dir: str = "."):
    """
    Compute mutual information scores on training data.

    Returns
    -------
    mi_df : pd.DataFrame
        DataFrame with Feature and MI_Score columns, sorted descending.
    """
    h("SECTION 5: MUTUAL INFORMATION (TRAIN-ONLY)")

    mi_scores = mutual_info_classif(
        X_train_var,
        y_train,
        discrete_features="auto",
        random_state=RANDOM_STATE,
    )

    mi_df = (
        pd.DataFrame({"Feature": feature_names_var, "MI_Score": mi_scores})
        .sort_values("MI_Score", ascending=False)
        .reset_index(drop=True)
    )

    print("Top 30 features by Mutual Information:")
    print(mi_df.head(30).to_string(index=False))

    out_path = f"{output_dir}/01_mutual_information_scores.csv"
    mi_df.to_csv(out_path, index=False)
    print(f"✓ Saved: {out_path}")

    return mi_df


# ============================================================================
# SECTION 6: RF IMPORTANCE RANKING
# ============================================================================

def compute_rf_importance(X_train_var, y_train, feature_names_var,
                          output_dir: str = "."):
    """
    Fit a Random Forest and extract feature importances.

    Returns
    -------
    imp_df : pd.DataFrame
        DataFrame with Feature and RF_Importance columns, sorted descending.
    """
    h("SECTION 6: RF IMPORTANCE RANKING")

    t0 = time.time()

    rf_selector = RandomForestClassifier(**RF_SELECTOR_PARAMS)
    print("Fitting selector RF for feature importances...")
    rf_selector.fit(X_train_var, y_train)

    importances = rf_selector.feature_importances_
    ranked_idx = np.argsort(importances)[::-1]

    imp_df = pd.DataFrame({
        "Feature": feature_names_var[ranked_idx],
        "RF_Importance": importances[ranked_idx],
    }).reset_index(drop=True)

    out_path = f"{output_dir}/02_rf_importance_ranking.csv"
    imp_df.to_csv(out_path, index=False)

    elapsed = (time.time() - t0) / 60
    print(f"✓ Saved: {out_path}")
    print(f"✓ RF fitting runtime: {elapsed:.2f} minutes")

    return imp_df


# ============================================================================
# SECTION 7: AGGREGATE TO ORIGINAL (PRE-ENCODING) FEATURES
# ============================================================================

def aggregate_to_original(imp_df, mi_df, output_dir: str = "."):
    """
    Aggregate encoded-level RF importance and MI scores back to original features.

    Returns
    -------
    summary : pd.DataFrame
        Aggregated importance summary with RF and MI scores per original feature.
    """
    h("SECTION 7: AGGREGATE TO ORIGINAL FEATURES")

    imp_df = imp_df.copy()
    mi_df = mi_df.copy()

    imp_df["Original_Feature"] = imp_df["Feature"].apply(get_original_feature)
    rf_agg = (
        imp_df.groupby("Original_Feature")["RF_Importance"]
        .agg(["sum", "max", "count"])
        .rename(columns={"sum": "RF_Importance_Sum", "max": "RF_Importance_Max",
                          "count": "Encoded_Columns"})
        .sort_values("RF_Importance_Sum", ascending=False)
        .reset_index()
    )

    mi_df["Original_Feature"] = mi_df["Feature"].apply(get_original_feature)
    mi_agg = (
        mi_df.groupby("Original_Feature")["MI_Score"]
        .agg(["sum", "max"])
        .rename(columns={"sum": "MI_Score_Sum", "max": "MI_Score_Max"})
        .sort_values("MI_Score_Sum", ascending=False)
        .reset_index()
    )

    summary = rf_agg.merge(mi_agg, on="Original_Feature", how="outer").fillna(0)
    summary = summary.sort_values("RF_Importance_Sum", ascending=False).reset_index(drop=True)
    summary.index = summary.index + 1
    summary.index.name = "Rank"

    print("ORIGINAL FEATURE IMPORTANCE SUMMARY")
    print("(Aggregated from encoded features)\n")
    print(summary.to_string())

    out_path = f"{output_dir}/03_original_feature_importance_summary.csv"
    summary.to_csv(out_path)
    print(f"\n✓ Saved: {out_path}")

    return summary


# ============================================================================
# SECTION 8: FEATURE-COUNT SWEEP
# ============================================================================

def feature_count_sweep(summary, X_train_var, X_val_var, y_train, y_val,
                        feature_names_var, output_dir: str = "."):
    """
    For each candidate feature count, train RF and measure validation F1.

    Returns
    -------
    sweep_df : pd.DataFrame
        Results for each feature count tested.
    ranked_original : list[str]
        Original features ranked by RF importance (used by Pareto selection).
    """
    h("SECTION 8: FEATURE-COUNT SWEEP (ORIGINAL FEATURES)")

    ranked_original = summary["Original_Feature"].tolist()
    sweep_results = []

    for i, k in enumerate(FEATURE_COUNTS_TO_TEST, 1):
        k = min(k, len(ranked_original))
        top_k_originals = ranked_original[:k]

        mask_k = np.array([
            get_original_feature(fname) in top_k_originals
            for fname in feature_names_var
        ])
        n_encoded = mask_k.sum()

        print(f"  [{i}/{len(FEATURE_COUNTS_TO_TEST)}] top-{k:2d} original features "
              f"({n_encoded} encoded)...", end=" ", flush=True)

        rf_eval = RandomForestClassifier(**RF_EVAL_PARAMS)
        rf_eval.fit(X_train_var[:, mask_k], y_train)

        y_pred_val = rf_eval.predict(X_val_var[:, mask_k])
        y_pred_train = rf_eval.predict(X_train_var[:, mask_k])

        f1_val = f1_score(y_val, y_pred_val, average="macro")
        f1_train = f1_score(y_train, y_pred_train, average="macro")
        f1_class0 = f1_score(y_val, y_pred_val, pos_label=0, average="binary")
        f1_class1 = f1_score(y_val, y_pred_val, pos_label=1, average="binary")

        sweep_results.append({
            "Original_Features": k,
            "Encoded_Features": n_encoded,
            "Train_F1": round(f1_train, 4),
            "Val_F1": round(f1_val, 4),
            "Val_F1_Normal": round(f1_class0, 4),
            "Val_F1_Attack": round(f1_class1, 4),
            "Overfit_Gap": round(f1_train - f1_val, 4),
            "Selected_Features": ", ".join(top_k_originals),
        })
        print(f"Val F1={f1_val:.4f}  Train={f1_train:.4f}  Gap={f1_train - f1_val:+.4f}")

    sweep_df = pd.DataFrame(sweep_results)
    out_path = f"{output_dir}/04_feature_count_sweep_results.csv"
    sweep_df.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")

    print("\nAll results:")
    print(sweep_df[["Original_Features", "Encoded_Features", "Train_F1", "Val_F1",
                     "Val_F1_Normal", "Val_F1_Attack", "Overfit_Gap"]].to_string(index=False))

    return sweep_df, ranked_original


# ============================================================================
# SECTION 9: OPTIMAL FEATURE COUNT SELECTION (PARETO)
# ============================================================================

def select_optimal_features(sweep_df, ranked_original, summary,
                            X_train_var, X_val_var, y_train, y_val,
                            feature_names_var, output_dir: str = "."):
    """
    Select optimal feature count via Pareto criterion and produce final report.

    Returns
    -------
    final_features : list[str]
        Names of the selected original features.
    OPTIMAL_N : int
        Number of original features selected.
    pareto_threshold : float
        The F1 threshold used for Pareto selection.
    """
    h("SECTION 9: OPTIMAL FEATURE COUNT (PARETO)")

    best_idx = sweep_df["Val_F1"].idxmax()
    best_n = int(sweep_df.loc[best_idx, "Original_Features"])
    best_f1 = float(sweep_df.loc[best_idx, "Val_F1"])

    pareto_threshold = best_f1 * 1.00
    pareto_candidates = sweep_df[sweep_df["Val_F1"] >= pareto_threshold]
    OPTIMAL_N = int(pareto_candidates["Original_Features"].min())

    print(f"Best Val F1:     {best_n} original features → {best_f1:.4f}")
    print(f"99% threshold:   {pareto_threshold:.4f}")
    print(f"Pareto choice:   {OPTIMAL_N} original features")

    final_features = ranked_original[:OPTIMAL_N]
    final_df = pd.DataFrame({
        "Rank": range(1, len(final_features) + 1),
        "Feature": final_features,
        "RF_Importance_Sum": [
            summary.loc[summary["Original_Feature"] == f, "RF_Importance_Sum"].values[0]
            for f in final_features
        ],
        "MI_Score_Sum": [
            summary.loc[summary["Original_Feature"] == f, "MI_Score_Sum"].values[0]
            for f in final_features
        ],
    })

    out_path = f"{output_dir}/05_final_selected_features.csv"
    final_df.to_csv(out_path, index=False)

    print(f"\n✓ Saved: {out_path}")
    print(f"\nSelected {OPTIMAL_N} original features:")
    for _, row in final_df.iterrows():
        print(f"  {row['Rank']:2d}. {row['Feature']:25s}  RF={row['RF_Importance_Sum']:.4f}  MI={row['MI_Score_Sum']:.4f}")

    # Final classification report
    mask_final = np.array([
        get_original_feature(fname) in final_features
        for fname in feature_names_var
    ])

    rf_final = RandomForestClassifier(**RF_EVAL_PARAMS)
    rf_final.fit(X_train_var[:, mask_final], y_train)
    y_pred_val = rf_final.predict(X_val_var[:, mask_final])

    print(f"\nValidation Classification Report ({OPTIMAL_N} original features, unseen computers):")
    print(classification_report(y_val, y_pred_val, target_names=["Normal", "Attack"], digits=4))

    return final_features, OPTIMAL_N, pareto_threshold


# ============================================================================
# CONVENIENCE: RUN FULL PIPELINE
# ============================================================================

def run_feature_selection_pipeline(X, y, groups, output_dir: str = "."):
    """
    Run the complete feature selection pipeline (Sections 3–9).

    Parameters
    ----------
    X : pd.DataFrame
        Engineered feature matrix.
    y : pd.Series
        Binary labels.
    groups : pd.Series
        Computer group identifiers.
    output_dir : str
        Directory for output CSVs.

    Returns
    -------
    dict with keys:
        X_train_var, X_val_var, y_train, y_val, feature_names_var,
        mi_df, imp_df, summary, sweep_df, ranked_original,
        final_features, OPTIMAL_N, pareto_threshold
    """
    X_train, X_val, y_train, y_val, groups_train, groups_val = group_split(X, y, groups)
    X_train_var, X_val_var, feature_names_var, preprocess = encode_and_filter(X_train, X_val)
    mi_df = compute_mutual_information(X_train_var, y_train, feature_names_var, output_dir)
    imp_df = compute_rf_importance(X_train_var, y_train, feature_names_var, output_dir)
    summary = aggregate_to_original(imp_df, mi_df, output_dir)
    sweep_df, ranked_original = feature_count_sweep(
        summary, X_train_var, X_val_var, y_train, y_val, feature_names_var, output_dir
    )
    final_features, OPTIMAL_N, pareto_threshold = select_optimal_features(
        sweep_df, ranked_original, summary,
        X_train_var, X_val_var, y_train, y_val, feature_names_var, output_dir
    )

    return {
        "X_train_var": X_train_var,
        "X_val_var": X_val_var,
        "y_train": y_train,
        "y_val": y_val,
        "feature_names_var": feature_names_var,
        "mi_df": mi_df,
        "imp_df": imp_df,
        "summary": summary,
        "sweep_df": sweep_df,
        "ranked_original": ranked_original,
        "final_features": final_features,
        "OPTIMAL_N": OPTIMAL_N,
        "pareto_threshold": pareto_threshold,
    }
