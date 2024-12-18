import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_validate,train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, confusion_matrix, make_scorer, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import optuna
import matplotlib.pyplot as plt
import os
import joblib
import seaborn as sns
import numpy as np
data_train_f2 = '/Users/umarkhan/Thesis/Datasets/credit_risk/german.data'

column_ = [
    'attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6', 'attr7', 'attr8', 'attr9', 'attr10',
    'attr11', 'attr12', 'attr13', 'attr14', 'attr15', 'attr16', 'attr17', 'attr18', 'attr19', 'attr20', 'target'
]

categorical_col = [
    'attr1', 'attr3', 'attr4', 'attr6', 'attr7', 'attr9', 'attr10', 'attr12', 'attr14', 'attr15', 'attr17', 'attr19', 'attr20'
]

df = pd.read_csv(data_train_f2, delim_whitespace=' ', header=None, names=column_)
print("==============- Optuna with Credit Risk -================")
le = LabelEncoder()
for col in categorical_col:
    df[col] = le.fit_transform(df[col])

X = df.drop('target', axis=1)
y = le.fit_transform(df['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=69)

# Example cost matrix for binary classification
# cost_fp = 5
# cost_fn = 1
# sample_weight = np.where(y_train == 1, cost_fn, cost_fp)
categorical_feature_indices = [X.columns.get_loc(col) for col in categorical_col]

rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
spw= len(y_train[y_train == 0]) / len(y_train[y_train == 1])

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
        "scale_pos_weight": spw
    }
    
    lgbm= lgb.LGBMClassifier(**param)
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
all_trials_df.to_csv("Credit_risk__optuna_all_trials.csv", index=False)
#Get Best model Parameters
best_params = study.best_params
best_params_str = str(best_params)
print("Best Model Parameters:", best_params)
best_model = lgb.LGBMClassifier(**best_params, objective='binary', boosting_type='gbdt', metric='binary_logloss', n_jobs=-1, scale_pos_weight=spw)
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'Credit_risk_bst_optuna.joblib')

# Write the best parameters to a file
best_params_file = os.path.join(os.getcwd(), "credit_risk_optuna_best_params.txt")
with open(best_params_file, "w") as file:
    file.write("Best Model Parameters:\n")
    for param, value in best_params.items():
        file.write(f"{param}: {value}\n")

print(f"Best Model Parameters saved to: {best_params_file}")

print("Best Model Parameters:", best_params)

scores = cross_validate(best_model, X_train, y_train, cv=rkf, scoring=scoring, n_jobs=-1)

# Extracting and saving the scores
fold_results = pd.DataFrame(scores)
output_file_path = os.path.join(os.getcwd(), 'Credit_risk_optuna_pfr.csv')
fold_results.to_csv(output_file_path, index=False)

# Calculating and saving metrics
mean_scores = fold_results.mean()
variance_scores = fold_results.var()
std_dev_scores = fold_results.std()
min_scores = fold_results.min()
max_scores = fold_results.max()

output_file_name = 'Credit_risk_optuna_metrics_output.md'
output_file_path = os.path.join(os.getcwd(), output_file_name)

with open(output_file_path, 'w') as file:
    file.write("Metric | Mean | Variance | Std Dev | Min | Max\n")
    file.write("-------|------|----------|---------|-----|-----\n")
    for metric in mean_scores.index:
        file.write(f"{metric} | {mean_scores[metric]:.4f} | {variance_scores[metric]:.4f} | {std_dev_scores[metric]:.4f} | {min_scores[metric]:.4f} | {max_scores[metric]:.4f}\n")
        file.write("\n\nBest Parameters\n",best_params_str)
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
plt.savefig('Credit_risk_optuna_auc_roc_plot.png')

# Save Confusion Matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('Credit_risk_optuna_confusion_matrix_plot.png')
