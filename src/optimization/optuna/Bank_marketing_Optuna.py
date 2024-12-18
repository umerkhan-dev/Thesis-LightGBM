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


data_train_f = '/Users/umarkhan/Thesis/Datasets/bank/bank-full.csv'
data_test_f = '/Users/umarkhan/Thesis/Datasets/bank/bank.csv'

df = pd.read_csv(data_train_f, delimiter=';')
df_test = pd.read_csv(data_test_f, delimiter=';')

labe = LabelEncoder()
X_train = df.drop('y', axis=1)
X_test = df_test.drop('y', axis=1)
y_train = labe.fit_transform(df['y'])
y_test = labe.transform(df_test['y'])

categorical_columns = X_train.select_dtypes(include=['object']).columns

le = LabelEncoder()
for col in categorical_columns:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

spw= len(y_train[y_train == 0]) / len(y_train[y_train == 1])
# Scoring
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
all_trials_df.to_csv("bank_Marketing_optuna_all_trials.csv", index=False)

best_params = study.best_params
best_params_str = str(study.best_params)
best_params_str = best_params_str.replace(', ', ',\n')
best_model = lgb.LGBMClassifier(**best_params, objective='binary', boosting_type='gbdt', metric='binary_logloss')
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'Bank_bst_optuna.joblib')


print("Best Model Parameters:", best_params)



scores = cross_validate(best_model, X_train, y_train, cv=rkf, scoring=scoring, n_jobs=-1)


# Extract individual fold scores
f1_scores = scores['test_f1_weighted']
recall_scores = scores['test_recall_weighted']
precision_scores = scores['test_precision_weighted']
accuracy_scores = scores['test_accuracy']
roc_auc_scores = scores['test_roc_auc_ovr']

# Create a DataFrame with individual fold scores
fold_results = pd.DataFrame({
    'F1': f1_scores,
    'Recall': recall_scores,
    'Precision': precision_scores,
    'Accuracy': accuracy_scores,
    'AUC-ROC': roc_auc_scores
})


# Extracting and saving the scores
fold_results = pd.DataFrame(scores)
output_file_path = os.path.join(os.getcwd(), 'Bank_optuna_pfr.csv')
fold_results.to_csv(output_file_path, index=False)

# Calculating and saving metrics
mean_scores = fold_results.mean()
variance_scores = fold_results.var()
std_dev_scores = fold_results.std()
min_scores = fold_results.min()
max_scores = fold_results.max()

output_file_name = 'Bank_optuna_metrics_output.md'
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
plt.savefig('Bank_optuna_auc_roc_plot.png')

# Save Confusion Matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('Bank_optuna_confusion_matrix_plot.png')
