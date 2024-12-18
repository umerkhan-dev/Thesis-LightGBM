import sklearn.metrics
from sklearn.model_selection import RepeatedKFold
import lightgbm as lgb
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
import pandas as pd
import numpy as np

def train_func(config):
    
    train_data_file = '/Users/umarkhan/Downloads/breast+cancer+wisconsin+original/breast-cancer-wisconsin.data'
    
    columns = [
        "Sample code number",
        "Clump Thickness",
        "Uniformity of Cell Size",
        "Uniformity of Cell Shape",
        "Marginal Adhesion",
        "Single Epithelial Cell Size",
        "Bare Nuclei",
        "Bland Chromatin",
        "Normal Nucleoli",
        "Mitoses",
        "Class"
    ]

    df = pd.read_csv(train_data_file, names = columns)

    df.replace('?', np.nan, inplace=True)
    df['Bare Nuclei'] = pd.to_numeric(df['Bare Nuclei'], errors='coerce')  # Convert to numeric
    # df.dropna(inplace=True)
    df['Class'] = df['Class'].map({2: 0, 4: 1})

    X = df.drop(['Class'], axis=1)
    y = df["Class"]
    
    # Repeated Stratified K-Fold Cross-Validation setup
    n_splits = 5
    n_repeats = 3
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    accuracies = []
    iteration_results = []

    for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Creating LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        spw = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        # Update config with additional parameters
        config["num_leaves"] = int(config["num_leaves"])
        config["num_iterations"] = int(config["num_iterations"])
        config["scale_pos_weight"] = spw
        # Train the classifier
        model = lgb.train(
            config,
            train_data,
            valid_sets=[test_data]
        )

        # Evaluate the model
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        accuracy = sklearn.metrics.accuracy_score(y_test, (y_pred > 0.5).astype(int))
        Recall = sklearn.metrics.recall_score(y_test, (y_pred > 0.5).astype(int))
        Precision = sklearn.metrics.precision_score(y_test, (y_pred > 0.5).astype(int))
        accuracies.append(accuracy)
        f1_score = sklearn.metrics.f1_score(y_test, (y_pred > 0.5).astype(int), average="weighted")
        roc_auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
        iteration_results.append((fold_index, f1_score,Recall,Precision, accuracy,roc_auc))

    mean_accuracy = np.mean(accuracies)

    return {"mean_accuracy": mean_accuracy, "iteration_results": iteration_results}

if __name__ == "__main__":
    config = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "scale_pos_weight":3.3,
        "learning_rate": tune.uniform(0.05, 0.15),
        "num_leaves": tune.randint(31, 80),
        "num_iterations": tune.randint(50, 150),
    }
    
    hyperband_scheduler = HyperBandScheduler(
        time_attr="training_iteration",
        max_t=100,
        reduction_factor=3,
        stop_last_trials=True
    )
    
    analysis = tune.run(
        train_func,
        resources_per_trial={"cpu": 8},
        num_samples=27,
        scheduler=hyperband_scheduler,
        config=config,
        metric="mean_accuracy",
        mode="max"
    )

    best_trial = analysis.get_best_trial("mean_accuracy", mode="max")
    best_config = best_trial.config
    best_accuracy = best_trial.last_result["mean_accuracy"]

    print("Best configuration found:")
    print(best_config)
    print("Best mean accuracy:", best_accuracy)

    # Extract the iteration results for all trials
    all_iteration_results = []
    for trial in analysis.trials:
        all_iteration_results.extend(trial.last_result.get("iteration_results", []))

    # Create a DataFrame with all iteration results
    results_df = pd.DataFrame(all_iteration_results, columns=["Fold","F1","Recall","Precision","Accuracy","AUC-ROC"])

    
    # Save the results DataFrame to a CSV file
    results_df.to_csv("/Users/umarkhan/hyperband/results/breast_cancer_results.csv", index=False)
