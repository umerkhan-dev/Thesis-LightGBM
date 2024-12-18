import sklearn.metrics
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
from matplotlib import pyplot as plt
# data_loc = ('/Users/umarkhan/Downloads/apple_quality.csv')


# def plot_class_distribution(data):
#     # Define colors
#         colors = ['#ADD8E6', '#FA8072', '#90EE90', '#FFA07A', '#87CEEB', '#FF69B4', '#FFD700', '#00FF00', '#FF00FF', '#00FFFF'] 
#         # Plot class distribution
#         data['Quality'].value_counts().plot(kind='bar', color=colors)
#         plt.title('Class Distribution for Apple Quality')
#         plt.xlabel('Class')
#         plt.ylabel('Count')
#         plt.xticks(rotation=360, ha='right')
#         # plt.subplots_adjust(bottom=0.3)  # Adjust the value as needed
#         plt.savefig('/Users/umarkhan/Bayesian/datasets/target_class dist fig/Apple_target_dist.png')
#         plt.show()

# plot_class_distribution(df)
def apple_train_func(config):
    print("========Apple Quality=========")

    # Load the dataset
    data_loc = ('/Users/umarkhan/Downloads/apple_quality.csv')
    df = pd.read_csv(data_loc)
   
    df = df.drop('A_id',axis=1)
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].apply(lambda x: x.astype('category'))
    
    le = LabelEncoder()
    X = df.drop('Quality',axis=1)
    y = le.fit_transform(df['Quality'])

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
        iteration_results.append((fold_index, f1_score, recall , precision,accuracy))


    mean_accuracy = np.mean(accuracies)

    return {"mean_accuracy": mean_accuracy, "iteration_results": iteration_results}

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
        apple_train_func,
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
    best_trial_iteration_results = best_trial.last_result.get("iteration_results", [])

# Calculate mean values for the metrics of the best iteration
    metrics = ["F1", "Recall", "Precision", "Accuracy"]
    mean_values = {metric: np.mean([result[metrics.index(metric) + 1] for result in best_trial_iteration_results]) for metric in metrics}

    

    # Path for the file with mean values
    mean_values_file_path = "apple_best_iteration_mean_values2.csv"

    # Write the mean values to a CSV file
    pd.DataFrame([mean_values]).to_csv(mean_values_file_path, index=False)
    # Extract the iteration results for all trials
    all_iteration_results = []
    for trial in analysis.trials:
        all_iteration_results.extend(trial.last_result.get("iteration_results", []))

    # Create a DataFrame with all iteration results
    results_df = pd.DataFrame(all_iteration_results, columns=["Fold","F1","Recall","Precision","Accuracy"])
    
    # Save the results DataFrame to a CSV file
    results_df.to_csv("/Users/umarkhan/hyperband/results/apple_hyper_results2.csv", index=False)
