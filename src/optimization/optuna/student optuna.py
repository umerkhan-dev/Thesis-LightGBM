import lightgbm as lgb
import os
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, confusion_matrix, make_scorer, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import joblib
import optuna
import numpy as np


print("================- Student Droput -=================")
dfile_path = 'datasets/student_dropout/student_data.csv'

df = pd.read_csv(dfile_path,delimiter=';')

le =LabelEncoder()

X = df.drop('Target',axis = 1)
y = le.fit_transform(df['Target'])



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=69,stratify=y)

rkf = RepeatedKFold(n_splits=5,n_repeats=3, random_state=69)

def roc_auc_ovr_scorer(estimator, X, y):
    y_pred_proba = estimator.predict_proba(X)
    return roc_auc_score(y, y_pred_proba, multi_class='ovr')

scoring = {
    'f1_weighted': make_scorer(f1_score, average='weighted'),
    'recall_weighted': make_scorer(recall_score, average='weighted'),
    'precision_weighted': make_scorer(precision_score, average='weighted'),
    'accuracy': make_scorer(accuracy_score),
    'roc_auc_ovr': roc_auc_ovr_scorer  # Custom scoring function
}

def objective(trial):
    param = {
        'objective': 'multiclass',
        'num_class': len(df['Target'].unique()),
        'boosting_type': 'gbdt',
        'metric': 'multi_logloss',
        'num_leaves': trial.suggest_int('num_leaves', 31, 80),
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),
        'n_jobs': -1,
        'verbose': -1
    }
    lgb_classifier = lgb.LGBMClassifier(**param)
    scores = cross_validate(lgb_classifier, X_train, y_train, scoring=scoring, cv=rkf, n_jobs=-1)
    trial.set_user_attr("fold_scores", scores)
    return np.mean(scores['test_f1_weighted'])

# Run Optuna optimization
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
all_trials_df.to_csv("student_optuna_all_trials.csv", index=False)

# Get the best model and parameters
best_params = study.best_params
best_model = lgb.LGBMClassifier(**best_params, objective='multiclass', boosting_type='gbdt', metric='multi_logloss')
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'student_bst_optuna.joblib')

# Write the best parameters to a file
best_params_file = os.path.join(os.getcwd(), "student_optuna_best_params.txt")
with open(best_params_file, "w") as file:
    file.write("Best Model Parameters:\n")
    for param, value in best_params.items():
        file.write(f"{param}: {value}\n")

print(f"Best Model Parameters saved to: {best_params_file}")

print("Best Model Parameters:", best_params)

# Use cross_val_score to perform cross-validation with the best model and get individual fold scores
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

# Save individual fold scores to a CSV file
output_file_path = os.path.join(os.getcwd(), 'student_optuna_per_fold_results.csv')
fold_results.to_csv(output_file_path, index=False)

# Calculate mean, variance, and standard deviation
mean_f1 = f1_scores.mean()
variance_f1 = f1_scores.var()
std_dev_f1 = f1_scores.std()
min_f1 = f1_scores.min()
max_f1 = f1_scores.max()

mean_recall = recall_scores.mean()
variance_recall = recall_scores.var()
std_dev_recall = recall_scores.std()
min_rcall = recall_scores.min()
max_recall = recall_scores.max()

mean_precision = precision_scores.mean()
variance_precision = precision_scores.var()
std_dev_precision = precision_scores.std()
min_precision = precision_scores.min()
max_precision = precision_scores.max()

mean_accuracy = accuracy_scores.mean()
variance_accuracy = accuracy_scores.var()
std_dev_accuracy = accuracy_scores.std()
min_accuracy_score = accuracy_scores.min()
max_accuracy_score = accuracy_scores.max()

mean_roc_auc = roc_auc_scores.mean()
variance_roc_auc = roc_auc_scores.var()
std_dev_roc_auc = roc_auc_scores.std()
min_roc_auc_score = roc_auc_scores.min()
max_roc_auc_score = roc_auc_scores.max()

current_directory = os.getcwd()
# Define the file name
output_file_name = 'student_optuna_metrics_output.md'

# Combine the current directory with the file name
output_file_path = os.path.join(current_directory, output_file_name)

# Open the file in write mode
with open(output_file_path, 'w') as file:
    # Write the metrics to the file in Markdown table format
    file.write( " Metric    | Mean | Variance | Std Dev | Min | Max \n")
    file.write( "-----------|------|----------|---------|\n")
    file.write(f" F1        | {mean_f1:.4f} | {variance_f1:.4f} | {std_dev_f1:.4f} | {min_f1:.4f} | {max_f1:.4f} \n")
    file.write(f" Recall    | {mean_recall:.4f} | {variance_recall:.4f} | {std_dev_recall:.4f} | {min_rcall:.4f} | {max_recall:.4f} \n")
    file.write(f" Precision | {mean_precision:.4f} | {variance_precision:.4f} | {std_dev_precision:.4f} | {min_precision:.4f} | {max_precision:.4f} \n")
    file.write(f" Accuracy  | {mean_accuracy:.4f} | {variance_accuracy:.4f} | {std_dev_accuracy:.4f} | {min_accuracy_score:.4f} | {max_accuracy_score:.4f} \n")
    file.write(f" AUC-ROC   | {mean_roc_auc:.4f} | {variance_roc_auc:.4f} | {std_dev_roc_auc:.4f} | {min_roc_auc_score:.4f} | {max_roc_auc_score:.4f} \n")

# Print a message indicating that the data has been written to the file
print(f"Metrics have been written to: {output_file_path}")


# Make predictions on the test data
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
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
plt.savefig('student_random_auc_roc_plot.png')


# Save Confusion Matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('student_random_marketing_plot.png')

