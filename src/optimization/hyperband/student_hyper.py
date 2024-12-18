import sklearn.metrics
from sklearn.model_selection import RepeatedKFold
import lightgbm as lgb
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
def train_func(config):
    print("========student dropout=========")

    # Load the dataset
    dfile_path = '/Users/umarkhan/hyperband/dataset/student_data.csv'

    df = pd.read_csv(dfile_path,delimiter=';')

    le =LabelEncoder()

    X = df.drop('Target',axis = 1)
    y = le.fit_transform(df['Target'])

    # Repeated Stratified K-Fold Cross-Validation setup
    n_splits = 5
    n_repeats = 3
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    accuracies = []
    iteration_results = []
    f1_scores = []
    for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Creating LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # Update config with additional parameters
        config["num_leaves"] = int(config["num_leaves"])
        config["num_iterations"] = int(config["num_iterations"])

        # Train the classifier
        model = lgb.train(
            config,
            train_data,
            valid_sets=[test_data]
        )

        # Evaluate the model
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        predictions = np.argmax(y_pred, axis=1)
        accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
        f1_score = sklearn.metrics.f1_score(y_test, predictions, average="weighted")
        recall = sklearn.metrics.recall_score(y_test,predictions, average="weighted")
        precision = sklearn.metrics.precision_score(y_test,predictions, average="weighted")
        
        # Omitting ROC AUC calculation for multi-class
        accuracies.append(accuracy)
        f1_scores.append(f1_score)
        iteration_results.append((fold_index, f1_score, recall , precision,accuracy))

    mean_accuracy = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)
    return {"mean_accuracy": mean_accuracy,"f1_score": mean_f1, "iteration_results": iteration_results}

if __name__ == "__main__":
    config = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        'num_class': 3,  # Ensure this matches your number of classes
        "boosting_type": "gbdt",
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
        resources_per_trial={"cpu": 2},
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
    results_df = pd.DataFrame(all_iteration_results, columns=["Fold","F1","Recall","Precision","Accuracy"])
    
    # Save the results DataFrame to a CSV file
    results_df.to_csv("/Users/umarkhan/hyperband/results/student_results.csv", index=False)



