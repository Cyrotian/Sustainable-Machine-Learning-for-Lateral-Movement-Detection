import pandas as pd
from sklearn.model_selection import train_test_split
from models.random_forest import RandomForestModel
from models.isolation_forest import IsolationForestModel
from models.lstm import LSTMModel


def random_forest():
    supervised_learning_dataset = pd.read_csv('final_datasets/supervised_learning_dataset.csv', low_memory=False)
    #testing_dataset = pd.read_csv('final_datasets/isolation_forest_test.csv', low_memory=False)
    testing_dataset = pd.read_csv('final_datasets/lmd2023_test_dataset.csv', low_memory=False)

    target_column = 'isMalicious'

    X_train = supervised_learning_dataset.drop(columns=[target_column])
    y_train = supervised_learning_dataset[target_column]
    X_test  = testing_dataset.drop(columns=[target_column])
    y_test  = testing_dataset[target_column]

    random_forest_model = RandomForestModel()
    random_forest_model.train(X_train, y_train)

    random_forest_model.plot_decision_tree(X_train)
    print("\nSanity check:", random_forest_model.evaluate(X_test, y_test))

    print("\n── Random Forest: 10 Inference Runs ──")
    for i in range(10):
        result = random_forest_model.inference_with_metrics(X_test, y_test)
        print(f"  Run {i+1}/10 | F1: {result['f1']:.4f} | "
              f"Time: {result['inference_time']:.3f}s | "
              f"Emissions: {result['emissions']:.8f} kg CO₂")

    summary = random_forest_model.get_inference_summary()
    print("\n── Random Forest Inference Summary (mean ± std) ──")
    for key, val in summary.items():
        print(f"  {key}: {val:.6f}")


def isolation_forest():

    CATEGORICAL_DTYPES = {"Image_Category": str, "RuleName": str, "SourceImage": str}


    unsupervised_learning_dataset = pd.read_csv('final_datasets/isolation_forest_train.csv', dtype=CATEGORICAL_DTYPES, low_memory=False)
    #testing_dataset = pd.read_csv('final_datasets/lmd2023_test_dataset.csv', low_memory=False)


    testing_dataset = pd.read_csv('final_datasets/isolation_forest_test.csv', dtype=CATEGORICAL_DTYPES, low_memory=False)

    target_column = 'isMalicious'

    X_train = unsupervised_learning_dataset.drop(columns=[target_column])
    X_test  = testing_dataset.drop(columns=[target_column])
    y_test  = testing_dataset[target_column]

    isolation_forest_model = IsolationForestModel(0.07)
    isolation_forest_model.train(X_train)

    print("\nSanity check:", isolation_forest_model.evaluate(X_test, y_test))

    print("\n── Isolation Forest: 10 Inference Runs ──")
    
    for i in range(10):
        result = isolation_forest_model.inference_with_metrics(X_test, y_test)
        print(f"  Run {i+1}/10 | F1: {result['f1']:.4f} | "
            f"Time: {result['inference_time']:.3f}s | "
            f"Emissions: {result['emissions']:.8f} kg CO₂")

    summary = isolation_forest_model.get_inference_summary()
    print("\n── Isolation Forest Inference Summary (mean ± std) ──")
    for key, val in summary.items():
        print(f"  {key}: {val:.6f}")


def lstm():
    supervised_learning_dataset = pd.read_csv('final_datasets/supervised_learning_dataset.csv', low_memory=False)
    #testing_dataset = pd.read_csv('final_datasets/isolation_forest_test.csv', low_memory=False)
    testing_dataset = pd.read_csv('final_datasets/lmd2023_test_dataset.csv', low_memory=False)

    target_column = 'isMalicious'

    X = supervised_learning_dataset.drop(columns=[target_column])
    y = supervised_learning_dataset[target_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test = testing_dataset.drop(columns=[target_column])
    y_test = testing_dataset[target_column]

    from codecarbon import EmissionsTracker
    carbon_tracker = EmissionsTracker(
        output_dir='emissions',
        project_name='lstm_training',
        log_level='error'
    )

    config = {'epochs': 10, 'batch_size': 512}
    lstm_model = LSTMModel(config)
    lstm_model.train(X_train, y_train, X_val, y_val, carbon_tracker)

    print("\nSanity check:", lstm_model.evaluate(X_test, y_test))

    print("\n── LSTM: 10 Inference Runs ──")
    for i in range(10):
        result = lstm_model.inference_with_metrics(X_test, y_test)
        print(f"  Run {i+1}/10 | F1: {result['f1']:.4f} | "
              f"Time: {result['inference_time']:.3f}s | "
              f"Emissions: {result['emissions']:.8f} kg CO₂")

    summary = lstm_model.get_inference_summary()
    print("\n── LSTM Inference Summary (mean ± std) ──")
    for key, val in summary.items():
        print(f"  {key}: {val:.6f}")


if __name__ == "__main__":
    random_forest()
    #isolation_forest()
    #lstm()
