import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, confusion_matrix, make_scorer, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import optuna
import matplotlib.pyplot as plt
import os
import joblib
import seaborn as sns
import numpy as np
print("============- Heart cleavland -================")

# Load the heart disease dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
tst_pth = "/Users/umarkhan/Thesis/Datasets/heart+disease/processed.hungarian.data"

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
data = pd.read_csv(url, names=column_names)
test = pd.read_csv(tst_pth, names=column_names)
data["target"] = data["target"].apply(lambda x: 1 if x > 0 else 0)
test["target"] = test["target"].apply(lambda x: 1 if x > 0 else 0)
# Preprocessing: handling missing values
data = data.replace("?", np.nan)
test = test.replace("?",np.nan)
df= pd.concat([data, test], ignore_index=True)
def plot_class_distribution(data):
    # Define colors
    colors = ['#ADD8E6', '#FA8072', '#90EE90', '#FFA07A', '#87CEEB', '#FF69B4', '#FFD700', '#00FF00', '#FF00FF', '#00FFFF'] 
    # Plot class distribution
    data['target'].value_counts().plot(kind='bar', color=colors)
    plt.title('Class Distribution for Breast Heart Disease')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=360, ha='right')
    plt.savefig('/Users/umarkhan/Bayesian/datasets/target_class dist fig/heart_target_dist.png')
    plt.show()

# plot_class_distribution(df)

# Assuming df_test (Hungarian) and df_data (Cleveland) are your DataFrames

# Convert 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'slope' to float in test (Hungarian) DataFrame
columns_to_convert_test = ['trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'slope']
test[columns_to_convert_test] = test[columns_to_convert_test].astype(float)

# Convert 'ca' and 'thal' to float in training (Cleveland) DataFrame
columns_to_convert_data = ['ca', 'thal']
data[columns_to_convert_data] = data[columns_to_convert_data].astype(float)


# print("Test (Hungarian):\n", test.dtypes)
# print("\nTraining (Cleveland):\n", data.dtypes)



data["ca"] = pd.to_numeric(data["ca"], errors='coerce')
data["thal"] = pd.to_numeric(data["thal"], errors='coerce')

test["ca"] = pd.to_numeric(test["ca"], errors='coerce')
test["thal"] = pd.to_numeric(test["thal"], errors='coerce')

# Convert target to integer (if it's not binary, you should adjust this part)
data["target"] = data["target"].apply(lambda x: 1 if x > 0 else 0)
test["target"] = test["target"].apply(lambda x: 1 if x > 0 else 0)

# Split the data into features (X) and target (y)
X = data.drop("target", axis=1)
y = data["target"]

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X
y_train = y

X_test = test.drop("target",axis=1)
y_test = test["target"]
# Create a RepeatedKFold object
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=69)

scoring = {
    'f1_weighted': make_scorer(f1_score, average='weighted'),
    'recall_weighted': make_scorer(recall_score, average='weighted'),
    'precision_weighted': make_scorer(precision_score, average='weighted'),
    'accuracy': make_scorer(accuracy_score),
    'roc_auc_ovr': make_scorer(roc_auc_score)  # Custom scoring function
}

def objective(trial):
    param = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "binary_logloss",
        "num_leaves": trial.suggest_int("num_leaves", 31, 80),
        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.15),
        "n_jobs": -1,
        "verbose":-1
    }
    lgbm = lgb.LGBMClassifier(**param)
    scores = cross_validate(lgbm, X_train, y_train, scoring=scoring, cv=rkf, n_jobs=-1)
    trial.set_user_attr("fold_scores", scores)
    return np.mean(scores['test_f1_weighted'])

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=27)

# Save per-parameter and per-fold results
all_trials = []
for trial in study.trials:
    trial_results = {
        "number": trial.number,
        "params": trial.params,
        "mean_f1_weighted": trial.value,
        "fold_scores": trial.user_attrs.get("fold_scores", {})
    }
    all_trials.append(trial_results)

all_trials_df = pd.DataFrame(all_trials)
all_trials_df.to_csv("Heart_optuna_all_trials.csv", index=False)

best_params = study.best_params
best_params_str = str(study.best_params)
best_params_str = best_params_str.replace(', ', ',\n')
best_model = lgb.LGBMClassifier(**best_params, objective='binary', boosting_type='gbdt', metric='binary_logloss')
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'Heart_bst_optuna.joblib')

print("Best Model Parameters:", best_params)
scores = cross_validate(best_model, X_train, y_train, cv=rkf, scoring=scoring, n_jobs=-1)


# Extracting and saving the scores
fold_results = pd.DataFrame(scores)
output_file_path = os.path.join(os.getcwd(), 'Heart_optuna_pfr.csv')
fold_results.to_csv(output_file_path, index=False)

# Calculating and saving metrics
mean_scores = fold_results.mean()
variance_scores = fold_results.var()
std_dev_scores = fold_results.std()
min_scores = fold_results.min()
max_scores = fold_results.max()

output_file_name = 'Heart_optuna_metrics_output.md'
output_file_path = os.path.join(os.getcwd(), output_file_name)

with open(output_file_path, 'w') as file:
    file.write("Metric | Mean | Variance | Std Dev | Min | Max\n")
    file.write("-------|------|----------|---------|-----|-----\n")
    for metric in mean_scores.index:
        file.write(f"{metric} | {mean_scores[metric]:.4f} | {variance_scores[metric]:.4f} | {std_dev_scores[metric]:.4f} | {min_scores[metric]:.4f} | {max_scores[metric]:.4f}\n")
        file.write("\n\nBest Parameters:\n" + best_params_str)
# Make predictions on the test data
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save AUC-ROC plot
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('Heart_optuna_auc_roc_plot.png')

# Save Confusion Matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('Heart_optuna_confusion_matrix_plot.png')
