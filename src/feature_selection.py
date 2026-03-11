"""
feature_selection.py — Select the optimal subset of features.

NOTE: This file is a ONE-TIME pipeline. It was run once to determine
      the best 5 features, saved to 05_final_selected_features.csv.
      It is NOT imported by the final dataset creation pipeline.
      It's kept here for reproducibility and to explain the methodology.

PURPOSE: Start with ~35 engineered features, use two ranking methods
         (Mutual Information + Random Forest Importance), then sweep
         different feature counts to find the fewest features that
         still achieve peak performance.

RESULT:  ~35 features -> 5 features (88% dimensionality reduction)

PIPELINE:
  Section 3: Split data by Computer (group-based)
  Section 4: One-hot encode categoricals + remove zero-variance
  Section 5: Rank features by Mutual Information
  Section 6: Rank features by Random Forest Importance
  Section 7: Aggregate encoded-level scores back to original features
  Section 8: Sweep feature counts (3, 5, 6, ..., 13) and measure F1
  Section 9: Pick optimal count via Pareto criterion
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, classification_report

from src.feature_engineering import h

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.20
FEATURE_COUNTS_TO_TEST = [3, 5, 6, 7, 8, 9, 10, 11, 12, 13]


# ============================================================================
# HELPER: Build Random Forest models
# ============================================================================

def build_selector_rf():
    """RF used for ranking feature importance (100 trees)."""
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )
    return rf


def build_eval_rf():
    """Stronger RF for evaluating feature subsets (150 trees)."""
    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=1,
    )
    return rf


# ============================================================================
# HELPER: Map encoded column names back to original feature names
# ============================================================================

# After one-hot encoding, "Image_Category" becomes multiple columns like:
#   "cat__Image_Category_AttackTool", "cat__Image_Category_System32", etc.
# We need to map these BACK to "Image_Category" to aggregate importance.

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


def get_original_feature(encoded_name):
    """
    Map an encoded column name back to its original feature.

    Examples:
      "cat__Image_Category_System32" -> "Image_Category"
      "num__EventID"                 -> "EventID"
    """
    if encoded_name.startswith("cat__"):
        rest = encoded_name[5:]
        sorted_features = sorted(KNOWN_ORIGINAL_FEATURES, key=len, reverse=True)
        for feat in sorted_features:
            if rest.startswith(feat + "_") or rest == feat:
                return feat
        if "_" in rest:
            parts = rest.rsplit("_", 1)
            return parts[0]
        return rest

    if encoded_name.startswith("num__"):
        return encoded_name[5:]

    return encoded_name


# ============================================================================
# SECTION 3: GROUP-BASED TRAIN/VALIDATION SPLIT
# ============================================================================

def group_split(X, y, groups):
    """
    Split data so that validation contains UNSEEN COMPUTERS.

    WHY NOT random split?
      A random split would mix rows from the same computer across
      train and val. The model could then "recognise" a computer's
      baseline rather than learning actual attack patterns.

    GroupShuffleSplit ensures:
      - 80% of COMPUTERS go to training
      - 20% of COMPUTERS go to validation (never seen during training)

    Returns: X_train, X_val, y_train, y_val, groups_train, groups_val
    """
    h("SECTION 3: GROUP-BASED TRAIN/VAL SPLIT")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    split_result = splitter.split(X, y, groups=groups)
    train_idx, val_idx = next(split_result)

    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_val = y.iloc[val_idx].copy()
    groups_train = groups.iloc[train_idx]
    groups_val = groups.iloc[val_idx]

    print("Train samples:", len(X_train))
    print("Val samples:  ", len(X_val))

    return X_train, X_val, y_train, y_val, groups_train, groups_val


# ============================================================================
# SECTION 4: ONE-HOT ENCODING + VARIANCE FILTER
# ============================================================================

def encode_and_filter(X_train, X_val):
    """
    Convert categorical features to numbers, then remove useless columns.

    ONE-HOT ENCODING: "Image_Category = AttackTool" becomes:
        Image_Category_AttackTool: 1, Image_Category_System32: 0, etc.
    Needed because sklearn's RF can't handle string inputs.

    VARIANCE FILTER: Remove encoded columns that are all-zero.

    IMPORTANT: Encoder is fit on TRAIN only to prevent data leakage.

    Returns: X_train_filtered, X_val_filtered, feature_names, encoder
    """
    h("SECTION 4: ONE-HOT ENCODING + VARIANCE FILTER")

    cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", encoder, cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    X_train_encoded = preprocess.fit_transform(X_train)
    X_val_encoded = preprocess.transform(X_val)
    feature_names = preprocess.get_feature_names_out()

    variance_filter = VarianceThreshold(threshold=0.0)
    X_train_filtered = variance_filter.fit_transform(X_train_encoded)
    X_val_filtered = variance_filter.transform(X_val_encoded)
    feature_names_filtered = feature_names[variance_filter.get_support()]

    print("Encoded features:", X_train_encoded.shape[1])
    print("After variance filter:", X_train_filtered.shape[1])

    return X_train_filtered, X_val_filtered, feature_names_filtered, preprocess


# ============================================================================
# SECTION 5: MUTUAL INFORMATION (Train-only)
# ============================================================================

def compute_mutual_information(X_train_filtered, y_train, feature_names, output_dir="."):
    """
    Rank features by Mutual Information with the label.

    MI measures how much knowing a feature reduces uncertainty about
    the label. MI = 0 means useless, MI > 0 means informative.
    Model-agnostic — captures any statistical dependency.

    Returns: DataFrame with Feature and MI_Score columns.
    """
    h("SECTION 5: MUTUAL INFORMATION")

    mi_scores = mutual_info_classif(
        X_train_filtered, y_train,
        discrete_features="auto",
        random_state=RANDOM_STATE,
    )

    mi_df = pd.DataFrame({
        "Feature": feature_names,
        "MI_Score": mi_scores,
    })
    mi_df = mi_df.sort_values("MI_Score", ascending=False)
    mi_df = mi_df.reset_index(drop=True)

    mi_df.to_csv(output_dir + "/01_mutual_information_scores.csv", index=False)
    return mi_df


# ============================================================================
# SECTION 6: RANDOM FOREST IMPORTANCE
# ============================================================================

def compute_rf_importance(X_train_filtered, y_train, feature_names, output_dir="."):
    """
    Rank features by RF Gini importance. Features that create the best
    splits get higher scores. Complements MI by capturing interactions.

    Returns: DataFrame with Feature and RF_Importance columns.
    """
    h("SECTION 6: RF IMPORTANCE RANKING")

    rf = build_selector_rf()
    rf.fit(X_train_filtered, y_train)

    importances = rf.feature_importances_
    ranked_indices = np.argsort(importances)[::-1]

    imp_df = pd.DataFrame({
        "Feature": feature_names[ranked_indices],
        "RF_Importance": importances[ranked_indices],
    })
    imp_df = imp_df.reset_index(drop=True)

    imp_df.to_csv(output_dir + "/02_rf_importance_ranking.csv", index=False)
    return imp_df


# ============================================================================
# SECTION 7: AGGREGATE TO ORIGINAL FEATURES
# ============================================================================

def aggregate_to_original(imp_df, mi_df, output_dir="."):
    """
    Sum encoded-level scores back to original features.

    Example: cat__Image_Category_AttackTool (0.15) +
             cat__Image_Category_System32 (0.08) = Image_Category (0.23)

    Returns: Summary DataFrame with one row per original feature.
    """
    h("SECTION 7: AGGREGATE TO ORIGINAL FEATURES")

    imp_copy = imp_df.copy()
    mi_copy = mi_df.copy()

    # Map each encoded name back to its original
    rf_originals = []
    for name in imp_copy["Feature"]:
        rf_originals.append(get_original_feature(name))
    imp_copy["Original_Feature"] = rf_originals

    mi_originals = []
    for name in mi_copy["Feature"]:
        mi_originals.append(get_original_feature(name))
    mi_copy["Original_Feature"] = mi_originals

    # Sum per original feature
    rf_grouped = imp_copy.groupby("Original_Feature")["RF_Importance"]
    rf_agg = rf_grouped.agg(["sum", "max", "count"])
    rf_agg = rf_agg.rename(columns={
        "sum": "RF_Importance_Sum",
        "max": "RF_Importance_Max",
        "count": "Encoded_Columns",
    })
    rf_agg = rf_agg.sort_values("RF_Importance_Sum", ascending=False)
    rf_agg = rf_agg.reset_index()

    mi_grouped = mi_copy.groupby("Original_Feature")["MI_Score"]
    mi_agg = mi_grouped.agg(["sum", "max"])
    mi_agg = mi_agg.rename(columns={
        "sum": "MI_Score_Sum",
        "max": "MI_Score_Max",
    })
    mi_agg = mi_agg.sort_values("MI_Score_Sum", ascending=False)
    mi_agg = mi_agg.reset_index()

    summary = rf_agg.merge(mi_agg, on="Original_Feature", how="outer")
    summary = summary.fillna(0)
    summary = summary.sort_values("RF_Importance_Sum", ascending=False)
    summary = summary.reset_index(drop=True)

    summary.to_csv(output_dir + "/03_original_feature_importance_summary.csv")
    return summary


# ============================================================================
# SECTION 8: FEATURE-COUNT SWEEP
# ============================================================================

def feature_count_sweep(summary, X_train_filtered, X_val_filtered,
                        y_train, y_val, feature_names, output_dir="."):
    """
    Try top-3, top-5, top-6, ..., top-13 features.
    For each, train a fresh RF and measure validation F1.
    Finds the sweet spot between too few (underfit) and too many (overfit).

    Returns: (results_dataframe, ranked_feature_list)
    """
    h("SECTION 8: FEATURE-COUNT SWEEP")

    ranked_original = summary["Original_Feature"].tolist()
    results = []

    for k in FEATURE_COUNTS_TO_TEST:
        if k > len(ranked_original):
            k = len(ranked_original)

        top_k_features = ranked_original[:k]

        # Boolean mask: which encoded columns belong to these top-K?
        mask = []
        for encoded_name in feature_names:
            original_name = get_original_feature(encoded_name)
            mask.append(original_name in top_k_features)
        mask = np.array(mask)

        # Train and evaluate
        rf = build_eval_rf()
        rf.fit(X_train_filtered[:, mask], y_train)

        y_pred_val = rf.predict(X_val_filtered[:, mask])
        y_pred_train = rf.predict(X_train_filtered[:, mask])

        f1_val = f1_score(y_val, y_pred_val, average="macro")
        f1_train = f1_score(y_train, y_pred_train, average="macro")
        f1_normal = f1_score(y_val, y_pred_val, pos_label=0, average="binary")
        f1_attack = f1_score(y_val, y_pred_val, pos_label=1, average="binary")

        results.append({
            "Original_Features": k,
            "Encoded_Features": mask.sum(),
            "Train_F1": round(f1_train, 4),
            "Val_F1": round(f1_val, 4),
            "Val_F1_Normal": round(f1_normal, 4),
            "Val_F1_Attack": round(f1_attack, 4),
            "Overfit_Gap": round(f1_train - f1_val, 4),
            "Selected_Features": ", ".join(top_k_features),
        })

        print("  Top-" + str(k) + ": Val F1=" + str(round(f1_val, 4)),
              " Gap=" + str(round(f1_train - f1_val, 4)))

    sweep_df = pd.DataFrame(results)
    sweep_df.to_csv(output_dir + "/04_feature_count_sweep_results.csv", index=False)
    return sweep_df, ranked_original


# ============================================================================
# SECTION 9: OPTIMAL FEATURE COUNT (PARETO CRITERION)
# ============================================================================

def select_optimal_features(sweep_df, ranked_original, summary,
                            X_train_filtered, X_val_filtered,
                            y_train, y_val, feature_names, output_dir="."):
    """
    Pick the FEWEST features that achieve the BEST validation F1.
    Fewer features = faster inference = less energy = less CO2.

    Returns: (final_feature_list, optimal_count, best_f1)
    """
    h("SECTION 9: OPTIMAL FEATURE COUNT (PARETO)")

    best_row = sweep_df["Val_F1"].idxmax()
    best_f1 = float(sweep_df.loc[best_row, "Val_F1"])

    good_enough = sweep_df[sweep_df["Val_F1"] >= best_f1]
    optimal_n = int(good_enough["Original_Features"].min())

    print("Best Val F1:", best_f1)
    print("Pareto choice:", optimal_n, "features")

    final_features = ranked_original[:optimal_n]

    # Save the final feature list
    ranks = []
    names = []
    rf_scores = []
    for i in range(len(final_features)):
        feat = final_features[i]
        ranks.append(i + 1)
        names.append(feat)
        matching = summary[summary["Original_Feature"] == feat]
        rf_scores.append(matching["RF_Importance_Sum"].values[0])

    final_df = pd.DataFrame({
        "Rank": ranks,
        "Feature": names,
        "RF_Importance_Sum": rf_scores,
    })
    final_df.to_csv(output_dir + "/05_final_selected_features.csv", index=False)

    # Print selected features
    print("\nSelected", optimal_n, "features:")
    for i in range(len(final_features)):
        print("  " + str(i + 1) + ". " + final_features[i])

    # Final classification report
    mask = []
    for encoded_name in feature_names:
        original_name = get_original_feature(encoded_name)
        mask.append(original_name in final_features)
    mask = np.array(mask)

    rf_final = build_eval_rf()
    rf_final.fit(X_train_filtered[:, mask], y_train)
    y_pred = rf_final.predict(X_val_filtered[:, mask])

    print("\nValidation Report (" + str(optimal_n) + " features, unseen computers):")
    print(classification_report(y_val, y_pred, target_names=["Normal", "Attack"], digits=4))

    return final_features, optimal_n, best_f1


# ============================================================================
# CONVENIENCE: RUN THE FULL PIPELINE
# ============================================================================

def run_feature_selection_pipeline(X, y, groups, output_dir="."):
    """Run Sections 3-9 in sequence."""
    X_train, X_val, y_train, y_val, g_train, g_val = group_split(X, y, groups)
    X_train_f, X_val_f, feat_names, preprocess = encode_and_filter(X_train, X_val)
    mi_df = compute_mutual_information(X_train_f, y_train, feat_names, output_dir)
    imp_df = compute_rf_importance(X_train_f, y_train, feat_names, output_dir)
    summary = aggregate_to_original(imp_df, mi_df, output_dir)
    sweep_df, ranked = feature_count_sweep(
        summary, X_train_f, X_val_f, y_train, y_val, feat_names, output_dir)
    final_features, optimal_n, threshold = select_optimal_features(
        sweep_df, ranked, summary,
        X_train_f, X_val_f, y_train, y_val, feat_names, output_dir)

    return {
        "final_features": final_features,
        "OPTIMAL_N": optimal_n,
        "pareto_threshold": threshold,
        "summary": summary,
        "sweep_df": sweep_df,
        "mi_df": mi_df,
        "imp_df": imp_df,
    }
