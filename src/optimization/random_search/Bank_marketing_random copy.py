import lightgbm as lgb
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, make_scorer, f1_score, recall_score, precision_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from scipy.stats import randint, uniform
import joblib

# Load data
data_train_f = '/Users/umarkhan/Thesis/Datasets/bank/bank-full.csv'
data_test_f = '/Users/umarkhan/Thesis/Datasets/bank/bank.csv'

df = pd.read_csv(data_train_f, delimiter=';')
df_test = pd.read_csv(data_test_f, delimiter=';')

# Encode labels
labe = LabelEncoder()
X_train = df.drop('y', axis=1)
X_test = df_test.drop('y', axis=1)
y_train = labe.fit_transform(df['y'])
y_test = labe.transform(df_test['y'])

# Encode categorical features
categorical_columns = X_train.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_columns:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Repeated K-Fold Cross Validator
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

# Scale positive weight
spw = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Fixed parameters
fixed_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'binary_logloss',
    'n_jobs': -1,
    'scale_pos_weight': spw,
    'verbose': -1
}

# Parameter distributions
param_dist = {
    'num_leaves': randint(31, 80),
    'n_estimators': randint(50, 150),
    'learning_rate': uniform(0.05, 0.10)
}

# Scoring metrics
scoring = {
    'f1': make_scorer(f1_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'precision': make_scorer(precision_score, average='weighted'),
    'accuracy': make_scorer(accuracy_score),
    'roc_auc': make_scorer(roc_auc_score, average='weighted')
}

# LGBM Classifier
lgb_classifier = lgb.LGBMClassifier(**fixed_params)

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=lgb_classifier,
    param_distributions=param_dist,
    cv=rkf,
    scoring=scoring['f1'],  # Use f1 score for the RandomizedSearchCV
    n_iter=27,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
best_params = random_search.best_params_

joblib.dump(best_model, 'Bank_marketing_bst_rand.joblib')

print("Best Model Parameters:", best_params)

# Save the results of each hyperparameter set
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('BM_ppr.txt', sep='\t', index=True)

# Cross-validation with the best model
scores = cross_validate(
    best_model, X_train, y_train, cv=rkf, scoring=scoring, n_jobs=-1
)

# Extract individual fold scores
fold_results = pd.DataFrame({
    'F1': scores['test_f1'],
    'Recall': scores['test_recall'],
    'Precision': scores['test_precision'],
    'Accuracy': scores['test_accuracy'],
    'AUC-ROC': scores['test_roc_auc']
})

# Save individual fold scores
output_file_path = os.path.join(os.getcwd(), 'BM_random_per_fold_results.csv')
fold_results.to_csv(output_file_path, index=False)

# Calculate mean, variance, and standard deviation
metrics = ['F1', 'Recall', 'Precision', 'Accuracy', 'AUC-ROC']
stats = fold_results.describe().loc[['mean', 'std', 'min', 'max']].T
stats['variance'] = fold_results.var()

output_file_path = os.path.join(os.getcwd(), 'BM_random_metrics_output.md')
with open(output_file_path, 'w') as file:
    file.write(" Metric    | Mean | Variance | Std Dev | Min | Max \n")
    file.write("-----------|------|----------|---------|-----|-----\n")
    for metric in metrics:
        file.write(f" {metric:<9}| {stats.loc[metric, 'mean']:.4f} | {stats.loc[metric, 'variance']:.4f} | {stats.loc[metric, 'std']:.4f} | {stats.loc[metric, 'min']:.4f} | {stats.loc[metric, 'max']:.4f} \n")

print(f"Metrics have been written to: {output_file_path}")

# Make predictions on the test data
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_rep)
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
plt.savefig('bank_marketing_auc_roc_plot.png')

# Save Confusion Matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('cm_bank_marketing_plot.png')
