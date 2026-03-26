import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, classification_report
from catboost import CatBoostClassifier

from src.feature_engineering import h


# CONFIG

RANDOM_STATE = 42
TEST_SIZE = 0.20
FEATURE_COUNTS_TO_TEST = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

KNOWN_ORIGINAL_FEATURES = [
    "DestinationPort_Bin",
    "SourcePort_Bin",
    "Image_Category",
    "RuleName",
    "FileVersion",
    "Description",
    "Product",
    "Company",
    "OriginalFileName",
    "CurrentDirectory",
    "TerminalSessionId",
    "IntegrityLevel",
    "ParentImage",
    "ParentUser",
    "Protocol",
    "Initiated",
    "SourceIsIpv6",
    "SourceIp",
    "SourceHostname",
    "SourcePortName",
    "DestinationIp",
    "DestinationHostname",
    "DestinationPortName",
    "ImageLoaded",
    "Signed",
    "Signature",
    "SourceImage",
    "TargetImage",
    "GrantedAccess",
    "SourceUser",
    "TargetUser",
    "EventType",
    "QueryName",
    "IsExecutable",
    "Archived",
    "NewThreadId",
    "StartAddress",
    "StartModule",
]


def build_selector_random_forest():
    # Random Forest used for feature-importance ranking.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )
    return model


def build_evaluation_random_forest():
    # Random Forest used for testing feature subsets.
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=1,
    )
    return model


def build_selector_catboost():
    # CatBoost used for feature-importance ranking.
    model = CatBoostClassifier(
        iterations=1000,
        random_seed=RANDOM_STATE,
        auto_class_weights="Balanced",
        verbose=0,
    )
    return model


# FEATURE NAME HELPERS

def get_original_feature(encoded_feature_name):
    # Map an encoded feature name back to its original feature name.
    if encoded_feature_name.startswith("cat__"):
        encoded_suffix = encoded_feature_name[5:]
        sorted_known_features = sorted(KNOWN_ORIGINAL_FEATURES, key=len, reverse=True)

        for original_feature_name in sorted_known_features:
            if encoded_suffix.startswith(original_feature_name + "_") or encoded_suffix == original_feature_name:
                return original_feature_name

        if "_" in encoded_suffix:
            parts = encoded_suffix.rsplit("_", 1)
            return parts[0]

        return encoded_suffix

    if encoded_feature_name.startswith("num__"):
        return encoded_feature_name[5:]

    return encoded_feature_name


def map_encoded_features_to_original_features(feature_names):
    # Convert a list of encoded feature names into original feature names.
    original_feature_names = []

    for encoded_feature_name in feature_names:
        original_feature_name = get_original_feature(encoded_feature_name)
        original_feature_names.append(original_feature_name)

    return original_feature_names


def create_feature_mask(encoded_feature_names, selected_original_features):
    # Build a boolean mask that selects encoded columns belonging to the requested original features.
    encoded_column_mask = []

    for encoded_feature_name in encoded_feature_names:
        original_feature_name = get_original_feature(encoded_feature_name)
        encoded_column_mask.append(original_feature_name in selected_original_features)

    return np.array(encoded_column_mask)


# ============================================================================
# SECTION 3: GROUP-BASED SPLIT
# ============================================================================

def group_split(X, y, groups):
    # Split the data so validation contains unseen computers.
    # This prevents rows from the same computer appearing in both training and validation.
    h("SECTION 3: GROUP-BASED TRAIN/VAL SPLIT")

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    split_iterator = splitter.split(X, y, groups=groups)
    train_indices, validation_indices = next(split_iterator)

    X_train = X.iloc[train_indices].copy()
    X_validation = X.iloc[validation_indices].copy()

    y_train = y.iloc[train_indices].copy()
    y_validation = y.iloc[validation_indices].copy()

    groups_train = groups.iloc[train_indices].copy()
    groups_validation = groups.iloc[validation_indices].copy()

    print("Train samples:", len(X_train))
    print("Val samples:  ", len(X_validation))

    return X_train, X_validation, y_train, y_validation, groups_train, groups_validation


# ============================================================================
# SECTION 4: ENCODE + FILTER
# ============================================================================

def encode_and_filter(X_train, X_validation):
    # One-hot encode categorical features and remove zero-variance columns.
    # The encoder is fit on training data only.
    h("SECTION 4: ONE-HOT ENCODING + VARIANCE FILTER")

    categorical_columns = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()

    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    preprocessing_pipeline = ColumnTransformer(
        transformers=[
            ("cat", one_hot_encoder, categorical_columns),
            ("num", "passthrough", numeric_columns),
        ],
        remainder="drop",
    )

    X_train_encoded = preprocessing_pipeline.fit_transform(X_train)
    X_validation_encoded = preprocessing_pipeline.transform(X_validation)

    encoded_feature_names = preprocessing_pipeline.get_feature_names_out()

    variance_filter = VarianceThreshold(threshold=0.0)

    X_train_filtered = variance_filter.fit_transform(X_train_encoded)
    X_validation_filtered = variance_filter.transform(X_validation_encoded)

    kept_feature_names = encoded_feature_names[variance_filter.get_support()]

    print("Encoded features:", X_train_encoded.shape[1])
    print("After variance filter:", X_train_filtered.shape[1])

    return X_train_filtered, X_validation_filtered, kept_feature_names, preprocessing_pipeline


# ============================================================================
# SECTION 5: MUTUAL INFORMATION
# ============================================================================

def compute_mutual_information(X_train_filtered, y_train, encoded_feature_names, output_dir="."):
    # Rank encoded features using Mutual Information.
    h("SECTION 5: MUTUAL INFORMATION")

    mutual_information_scores = mutual_info_classif(
        X_train_filtered,
        y_train,
        discrete_features="auto",
        random_state=RANDOM_STATE,
    )

    results_dataframe = pd.DataFrame({
        "Feature": encoded_feature_names,
        "MI_Score": mutual_information_scores,
    })

    results_dataframe = results_dataframe.sort_values("MI_Score", ascending=False).reset_index(drop=True)
    results_dataframe.to_csv(output_dir + "/01_mutual_information_scores.csv", index=False)

    return results_dataframe


# ============================================================================
# SECTION 6: RANDOM FOREST IMPORTANCE
# ============================================================================

def compute_random_forest_importance(X_train_filtered, y_train, encoded_feature_names, output_dir="."):
    # Rank encoded features using Random Forest Gini importance.
    h("SECTION 6: RF IMPORTANCE RANKING")

    random_forest_model = build_selector_random_forest()
    random_forest_model.fit(X_train_filtered, y_train)

    importance_scores = random_forest_model.feature_importances_
    ranked_indices = np.argsort(importance_scores)[::-1]

    results_dataframe = pd.DataFrame({
        "Feature": encoded_feature_names[ranked_indices],
        "RF_Importance": importance_scores[ranked_indices],
    })

    results_dataframe = results_dataframe.reset_index(drop=True)
    results_dataframe.to_csv(output_dir + "/02_rf_importance_ranking.csv", index=False)

    return results_dataframe


# ============================================================================
# SECTION 6b: CATBOOST IMPORTANCE
# ============================================================================

def convert_sparse_to_dense_if_needed(matrix):
    # Convert a sparse matrix to a dense array when required.
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def compute_catboost_importance(X_train_filtered, y_train, encoded_feature_names, output_dir="."):
    # Rank encoded features using CatBoost feature importance.
    h("SECTION 6b: CATBOOST IMPORTANCE RANKING")

    catboost_model = build_selector_catboost()
    X_train_dense = convert_sparse_to_dense_if_needed(X_train_filtered)

    catboost_model.fit(X_train_dense, y_train)

    importance_scores = catboost_model.get_feature_importance()
    ranked_indices = np.argsort(importance_scores)[::-1]

    results_dataframe = pd.DataFrame({
        "Feature": encoded_feature_names[ranked_indices],
        "CB_Importance": importance_scores[ranked_indices],
    })

    results_dataframe = results_dataframe.reset_index(drop=True)
    results_dataframe.to_csv(output_dir + "/02b_catboost_importance_ranking.csv", index=False)

    return results_dataframe


# ============================================================================
# SECTION 7: AGGREGATE TO ORIGINAL FEATURES
# ============================================================================

def add_original_feature_column(dataframe, feature_column_name="Feature"):
    # Add a column containing the original feature name for each encoded feature.
    updated_dataframe = dataframe.copy()
    updated_dataframe["Original_Feature"] = map_encoded_features_to_original_features(updated_dataframe[feature_column_name])
    return updated_dataframe


def aggregate_rf_scores_to_original_features(random_forest_dataframe):
    # Aggregate Random Forest encoded-level scores back to original features.
    grouped = random_forest_dataframe.groupby("Original_Feature")["RF_Importance"]
    aggregated = grouped.agg(["sum", "max", "count"])

    aggregated = aggregated.rename(columns={
        "sum": "RF_Importance_Sum",
        "max": "RF_Importance_Max",
        "count": "Encoded_Columns",
    })

    aggregated = aggregated.sort_values("RF_Importance_Sum", ascending=False).reset_index()
    return aggregated


def aggregate_mi_scores_to_original_features(mutual_information_dataframe):
    # Aggregate Mutual Information encoded-level scores back to original features.
    grouped = mutual_information_dataframe.groupby("Original_Feature")["MI_Score"]
    aggregated = grouped.agg(["sum", "max"])

    aggregated = aggregated.rename(columns={
        "sum": "MI_Score_Sum",
        "max": "MI_Score_Max",
    })

    aggregated = aggregated.sort_values("MI_Score_Sum", ascending=False).reset_index()
    return aggregated


def aggregate_catboost_scores_to_original_features(catboost_dataframe):
    # Aggregate CatBoost encoded-level scores back to original features.
    grouped = catboost_dataframe.groupby("Original_Feature")["CB_Importance"]
    aggregated = grouped.agg(["sum", "max"])

    aggregated = aggregated.rename(columns={
        "sum": "CB_Importance_Sum",
        "max": "CB_Importance_Max",
    })

    aggregated = aggregated.reset_index()
    return aggregated


def aggregate_to_original(
    random_forest_importance_dataframe,
    mutual_information_dataframe,
    catboost_importance_dataframe=None,
    output_dir=".",
):
    # Combine encoded-level ranking results into one row per original feature.
    h("SECTION 7: AGGREGATE TO ORIGINAL FEATURES")

    rf_with_originals = add_original_feature_column(random_forest_importance_dataframe)
    mi_with_originals = add_original_feature_column(mutual_information_dataframe)

    random_forest_summary = aggregate_rf_scores_to_original_features(rf_with_originals)
    mutual_information_summary = aggregate_mi_scores_to_original_features(mi_with_originals)

    summary_dataframe = random_forest_summary.merge(mutual_information_summary, on="Original_Feature", how="outer")

    if catboost_importance_dataframe is not None:
        cb_with_originals = add_original_feature_column(catboost_importance_dataframe)
        catboost_summary = aggregate_catboost_scores_to_original_features(cb_with_originals)
        summary_dataframe = summary_dataframe.merge(catboost_summary, on="Original_Feature", how="outer")

    summary_dataframe = summary_dataframe.fillna(0)
    summary_dataframe = summary_dataframe.sort_values("RF_Importance_Sum", ascending=False).reset_index(drop=True)

    summary_dataframe.to_csv(output_dir + "/03_original_feature_importance_summary.csv", index=False)
    return summary_dataframe


# SECTION 8: FEATURE COUNT SWEEP

def evaluate_feature_subset(
    selected_original_features,
    X_train_filtered,
    X_validation_filtered,
    y_train,
    y_validation,
    encoded_feature_names,
):
    # Train a Random Forest using only the chosen original features.
    encoded_column_mask = create_feature_mask(encoded_feature_names, selected_original_features)

    model = build_evaluation_random_forest()
    model.fit(X_train_filtered[:, encoded_column_mask], y_train)

    training_predictions = model.predict(X_train_filtered[:, encoded_column_mask])
    validation_predictions = model.predict(X_validation_filtered[:, encoded_column_mask])

    training_f1 = f1_score(y_train, training_predictions, average="macro")
    validation_f1 = f1_score(y_validation, validation_predictions, average="macro")
    validation_f1_normal = f1_score(y_validation, validation_predictions, pos_label=0, average="binary")
    validation_f1_attack = f1_score(y_validation, validation_predictions, pos_label=1, average="binary")

    results = {
        "Encoded_Features": int(encoded_column_mask.sum()),
        "Train_F1": round(training_f1, 4),
        "Val_F1": round(validation_f1, 4),
        "Val_F1_Normal": round(validation_f1_normal, 4),
        "Val_F1_Attack": round(validation_f1_attack, 4),
        "Overfit_Gap": round(training_f1 - validation_f1, 4),
    }

    return results


def feature_count_sweep(
    summary_dataframe,
    X_train_filtered,
    X_validation_filtered,
    y_train,
    y_validation,
    encoded_feature_names,
    output_dir=".",
):
    # Test several feature counts and record validation performance.
    h("SECTION 8: FEATURE-COUNT SWEEP")

    ranked_original_features = summary_dataframe["Original_Feature"].tolist()
    sweep_results = []

    for number_of_features in FEATURE_COUNTS_TO_TEST:
        adjusted_feature_count = min(number_of_features, len(ranked_original_features))
        selected_original_features = ranked_original_features[:adjusted_feature_count]

        metrics = evaluate_feature_subset(
            selected_original_features,
            X_train_filtered,
            X_validation_filtered,
            y_train,
            y_validation,
            encoded_feature_names,
        )

        sweep_results.append({
            "Original_Features": adjusted_feature_count,
            "Encoded_Features": metrics["Encoded_Features"],
            "Train_F1": metrics["Train_F1"],
            "Val_F1": metrics["Val_F1"],
            "Val_F1_Normal": metrics["Val_F1_Normal"],
            "Val_F1_Attack": metrics["Val_F1_Attack"],
            "Overfit_Gap": metrics["Overfit_Gap"],
            "Selected_Features": ", ".join(selected_original_features),
        })

        print("  Top-" + str(adjusted_feature_count) + ": Val F1=" + str(metrics["Val_F1"]) + " Gap=" + str(metrics["Overfit_Gap"]))

    sweep_dataframe = pd.DataFrame(sweep_results)
    sweep_dataframe.to_csv(output_dir + "/04_feature_count_sweep_results.csv", index=False)

    return sweep_dataframe, ranked_original_features


# SECTION 9: SELECT FINAL FEATURE SET

def get_best_validation_f1(sweep_dataframe):
    # Return the highest validation F1 in the sweep results.
    best_row_index = sweep_dataframe["Val_F1"].idxmax()
    best_validation_f1 = float(sweep_dataframe.loc[best_row_index, "Val_F1"])
    return best_validation_f1


def get_smallest_feature_count_that_matches_best_f1(sweep_dataframe, best_validation_f1):
    # Return the smallest feature count whose validation F1 matches the best score.
    good_enough_rows = sweep_dataframe[sweep_dataframe["Val_F1"] >= best_validation_f1]
    optimal_feature_count = int(good_enough_rows["Original_Features"].min())
    return optimal_feature_count


def save_final_feature_list(final_features, summary_dataframe, output_dir="."):
    # Save the chosen final feature list to CSV.
    ranks = []
    feature_names = []
    random_forest_scores = []

    for feature_rank, feature_name in enumerate(final_features, start=1):
        matching_rows = summary_dataframe[summary_dataframe["Original_Feature"] == feature_name]
        ranks.append(feature_rank)
        feature_names.append(feature_name)
        random_forest_scores.append(matching_rows["RF_Importance_Sum"].values[0])

    final_features_dataframe = pd.DataFrame({
        "Rank": ranks,
        "Feature": feature_names,
        "RF_Importance_Sum": random_forest_scores,
    })

    final_features_dataframe.to_csv(output_dir + "/05_final_selected_features.csv", index=False)


def print_selected_features(final_features):
    # Print the final selected features.
    print("\nSelected", len(final_features), "features:")
    for feature_rank, feature_name in enumerate(final_features, start=1):
        print("  " + str(feature_rank) + ". " + feature_name)


def print_final_validation_report(
    final_features,
    X_train_filtered,
    X_validation_filtered,
    y_train,
    y_validation,
    encoded_feature_names,
):
    # Train the final model and print a validation classification report.
    encoded_column_mask = create_feature_mask(encoded_feature_names, final_features)

    final_model = build_evaluation_random_forest()
    final_model.fit(X_train_filtered[:, encoded_column_mask], y_train)

    validation_predictions = final_model.predict(X_validation_filtered[:, encoded_column_mask])

    print("\nValidation Report (" + str(len(final_features)) + " features, unseen computers):")
    print(classification_report(y_validation, validation_predictions, target_names=["Normal", "Attack"], digits=4))


def select_optimal_features(
    sweep_dataframe,
    ranked_original_features,
    summary_dataframe,
    X_train_filtered,
    X_validation_filtered,
    y_train,
    y_validation,
    encoded_feature_names,
    output_dir=".",
):
    # Choose the smallest feature set that still achieves the best validation F1.
    h("SECTION 9: OPTIMAL FEATURE COUNT (PARETO)")

    best_validation_f1 = get_best_validation_f1(sweep_dataframe)
    optimal_feature_count = get_smallest_feature_count_that_matches_best_f1(sweep_dataframe, best_validation_f1)

    print("Best Val F1:", best_validation_f1)
    print("Pareto choice:", optimal_feature_count, "features")

    final_features = ranked_original_features[:optimal_feature_count]

    save_final_feature_list(final_features, summary_dataframe, output_dir)
    print_selected_features(final_features)

    print_final_validation_report(
        final_features,
        X_train_filtered,
        X_validation_filtered,
        y_train,
        y_validation,
        encoded_feature_names,
    )

    return final_features, optimal_feature_count, best_validation_f1


# RUN THE FULL PIPELINE

def run_feature_selection_pipeline(X, y, groups, output_dir="."):
    # Run the full feature-selection pipeline from start to finish.
    X_train, X_validation, y_train, y_validation, groups_train, groups_validation = group_split(X, y, groups)

    X_train_filtered, X_validation_filtered, encoded_feature_names, preprocessing_pipeline = encode_and_filter(
        X_train,
        X_validation,
    )

    mutual_information_dataframe = compute_mutual_information(
        X_train_filtered,
        y_train,
        encoded_feature_names,
        output_dir,
    )

    random_forest_importance_dataframe = compute_random_forest_importance(
        X_train_filtered,
        y_train,
        encoded_feature_names,
        output_dir,
    )

    catboost_importance_dataframe = compute_catboost_importance(
        X_train_filtered,
        y_train,
        encoded_feature_names,
        output_dir,
    )

    summary_dataframe = aggregate_to_original(
        random_forest_importance_dataframe,
        mutual_information_dataframe,
        catboost_importance_dataframe,
        output_dir,
    )

    sweep_dataframe, ranked_original_features = feature_count_sweep(
        summary_dataframe,
        X_train_filtered,
        X_validation_filtered,
        y_train,
        y_validation,
        encoded_feature_names,
        output_dir,
    )

    final_features, optimal_feature_count, pareto_threshold = select_optimal_features(
        sweep_dataframe,
        ranked_original_features,
        summary_dataframe,
        X_train_filtered,
        X_validation_filtered,
        y_train,
        y_validation,
        encoded_feature_names,
        output_dir,
    )

    return {
        "final_features": final_features,
        "OPTIMAL_N": optimal_feature_count,
        "pareto_threshold": pareto_threshold,
        "summary": summary_dataframe,
        "sweep_df": sweep_dataframe,
        "mi_df": mutual_information_dataframe,
        "imp_df": random_forest_importance_dataframe,
        "cb_df": catboost_importance_dataframe,
    }