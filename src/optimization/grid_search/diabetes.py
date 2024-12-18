import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, recall_score, precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# File path for the diabetes dataset
fp = '/Users/umarkhan/Downloads/datasets/diabetes.csv'

# Load dataset
df = pd.read_csv(fp)

# Split data into features and target
X = df.drop(['Outcome'], axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# Calculate scale_pos_weight
spw = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Fixed parameters for LightGBM
fixed_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'binary_logloss',
    'n_jobs': -1,
    'scale_pos_weight': spw,
    'verbose': -1
}

# Create and train LightGBM model
lgb_classifier = lgb.LGBMClassifier(**fixed_params)
lgb_classifier.fit(X_train, y_train)

# Save the trained model
joblib.dump(lgb_classifier, 'diabetes_default_model.joblib')

# Make predictions on the test data
y_pred = lgb_classifier.predict(X_test)
y_pred_proba = lgb_classifier.predict_proba(X_test)[:, 1]

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')

# Display results
print(f"Test Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")

# Save AUC-ROC plot
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('diabetes_default_auc_roc_plot.png')

# Save Confusion Matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('diabetes_default_confusion_matrix.png')
