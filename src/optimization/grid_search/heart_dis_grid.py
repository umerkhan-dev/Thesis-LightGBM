import lightgbm as lgb
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_curve, f1_score, recall_score,
    precision_score, roc_auc_score, confusion_matrix
)
import numpy as np
import os
import joblib
import seaborn as sns

print("============- Heart Cleveland -================")

# Load the heart disease dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
tst_pth = "/Users/umarkhan/Thesis/Datasets/heart+disease/processed.hungarian.data"

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
data = pd.read_csv(url, names=column_names)
test = pd.read_csv(tst_pth, names=column_names)

# Preprocessing: handling missing values
data = data.replace("?", np.nan)
test = test.replace("?", np.nan)

# Convert columns to appropriate data types
columns_to_convert_test = ['trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'slope']
test[columns_to_convert_test] = test[columns_to_convert_test].astype(float)

columns_to_convert_data = ['ca', 'thal']
data[columns_to_convert_data] = data[columns_to_convert_data].astype(float)

data["ca"] = pd.to_numeric(data["ca"], errors='coerce')
data["thal"] = pd.to_numeric(data["thal"], errors='coerce')
test["ca"] = pd.to_numeric(test["ca"], errors='coerce')
test["thal"] = pd.to_numeric(test["thal"], errors='coerce')

# Convert target to binary integer values
data["target"] = data["target"].apply(lambda x: 1 if x > 0 else 0)
test["target"] = test["target"].apply(lambda x: 1 if x > 0 else 0)

# Split the data into features (X) and target (y)
X_train = data.drop("target", axis=1)
y_train = data["target"]
X_test = test.drop("target", axis=1)
y_test = test["target"]

# Create a LightGBM classifier with default settings
lgb_classifier = lgb.LGBMClassifier(verbose=-1)

# Fit the model to the training data
lgb_classifier.fit(X_train, y_train)

# Save the fitted model
joblib.dump(lgb_classifier, 'heart_default_model.joblib')

# Make predictions on the test data
y_pred = lgb_classifier.predict(X_test)
y_pred_proba = lgb_classifier.predict_proba(X_test)[:, 1]

# Evaluate the model's performance on the test data
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
plt.savefig('heart_default_auc_roc_plot.png')

# Save Confusion Matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('heart_default_confusion_matrix.png')
