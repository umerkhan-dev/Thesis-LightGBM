import sklearn.metrics
from sklearn.model_selection import RepeatedKFold
import lightgbm as lgb
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
def train_func(config):
    # URL and local path (Modify the local path as per your setup)
    data_train_f2 = '/Users/umarkhan/Bayesian/datasets/credit_risk/german.data'

    column_ = [
        'attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6', 'attr7', 'attr8', 'attr9', 'attr10',
        'attr11', 'attr12', 'attr13', 'attr14', 'attr15', 'attr16', 'attr17', 'attr18', 'attr19', 'attr20', 'target'
    ]

    categorical_col = [
        'attr1', 'attr3', 'attr4', 'attr6', 'attr7', 'attr9', 'attr10', 'attr12', 'attr14', 'attr15', 'attr17', 'attr19', 'attr20'
    ]

    df = pd.read_csv(data_train_f2, delim_whitespace=' ', header=None, names=column_)

    le = LabelEncoder()
    for col in categorical_col:
        df[col] = le.fit_transform(df[col])

    X = df.drop('target', axis=1)
    y = le.fit_transform(df['target'])
    
    # Repeated Stratified K-Fold Cross-Validation setup
    n_splits = 5
    n_repeats = 3
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    accuracies = []
    iteration_results = []
    f1_scores =[]
    for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Creating LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train,categorical_feature=categorical_col)
        test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=categorical_col)
        # spw= len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        # # Update config with additional parameters
        # config["num_leaves"] = int(config["num_leaves"])
        # config["num_iterations"] = int(config["num_iterations"])
        
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

    mean_accuracy= np.mean(accuracies)

    return {"mean_accuracy": mean_accuracy, "iteration_results": iteration_results}

if __name__ == "__main__":
    config = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "scale_pos_weight":2.3335,
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
        resources_per_trial={"cpu": 4},
        num_samples=27,
        scheduler=hyperband_scheduler,
        config=config,
        metric="mean_accuracy",
        mode="max"
    )

    best_trial = analysis.get_best_trial("f1_weighted", mode="max")
    best_config = best_trial.config
    best_accuracy = best_trial.last_result["f1_weighted"]

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
    results_df.to_csv("/Users/umarkhan/hyperband/results/credit_results_v2.csv", index=False)
