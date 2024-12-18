import lightgbm as lgb
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score,roc_curve,make_scorer, f1_score,recall_score,precision_score,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import joblib

print("==============  Bank Marketing  =============")
start_time = time.time()

data_train_f = '~/bank/bank-full.csv'
data_test_f = '~/bank/bank.csv'

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

fixed_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'binary_logloss',
    'n_jobs': -1,
    'scale_pos_weight': spw,
    'verbose':-1
}

param_grid = {
    'num_leaves': [31, 50, 80],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.15, 0.05, 0.1]
}

scoring = {
    'f1': make_scorer(f1_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'precision': make_scorer(precision_score, average='weighted'),
    'accuracy': make_scorer(accuracy_score),
    'roc_auc': make_scorer(roc_auc_score, average='weighted')
}
lgb_classifier = lgb.LGBMClassifier(**fixed_params)
grid_search = GridSearchCV(estimator=lgb_classifier, param_grid=param_grid,
                           cv=rkf,
                           scoring='f1_weighted')

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

model_filename = 'bank_model.joblib'
joblib.dump(best_model, model_filename)
print(f"Best Model saved to: {model_filename}")


print("Best Model Parameters:", best_params)


# Access the results of each hyperparameter set
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv('BM_ppr.txt', sep='\t', index=True)

# Use cross_val_score to perform cross-validation with the best model and get individual fold scores
scores = cross_validate(
    best_model, X_train, y_train, cv=rkf, scoring=scoring, n_jobs=-1
)


# Extract individual fold scores
f1_scores = scores['test_f1']
recall_scores = scores['test_recall']
precision_scores = scores['test_precision']
accuracy_scores = scores['test_accuracy']
roc_auc_scores = scores['test_roc_auc']


fold_results = pd.DataFrame({
    'F1': f1_scores,
    'Recall': recall_scores,
    'Precision': precision_scores,
    'Accuracy': accuracy_scores,
    'AUC-ROC': roc_auc_scores
})

# Get the file path in the current directory
output_file_path = os.path.join(os.getcwd(), 'BM_Grid_per_fold_results.csv')

# Save the individual fold scores to a CSV file
fold_results.to_csv(output_file_path, index=False)


# Calculate mean, variance, and standard deviation
mean_f1 = f1_scores.mean()
variance_f1 = f1_scores.var()
std_dev_f1 = f1_scores.std()

mean_recall = recall_scores.mean()
variance_recall = recall_scores.var()
std_dev_recall = recall_scores.std()

mean_precision = precision_scores.mean()
variance_precision = precision_scores.var()
std_dev_precision = precision_scores.std()

mean_accuracy = accuracy_scores.mean()
variance_accuracy = accuracy_scores.var()
std_dev_accuracy = accuracy_scores.std()

mean_roc_auc = roc_auc_scores.mean()
variance_roc_auc = roc_auc_scores.var()
std_dev_roc_auc = roc_auc_scores.std()


current_directory = os.getcwd()
# Define the file name
output_file_name = 'BM_grid_metrics_output.md'

# Combine the current directory with the file name
output_file_path = os.path.join(current_directory, output_file_name)

# Open the file in write mode
with open(output_file_path, 'w') as file:
    # Write the metrics to the file in Markdown table format
    file.write( "| Metric    | Mean | Variance | Std Dev |\n")
    file.write( "|-----------|------|----------|---------|\n")
    file.write(f"| F1        | {mean_f1:.4f} | {variance_f1:.4f} | {std_dev_f1:.4f} |\n")
    file.write(f"| Recall    | {mean_recall:.4f} | {variance_recall:.4f} | {std_dev_recall:.4f} |\n")
    file.write(f"| Precision | {mean_precision:.4f} | {variance_precision:.4f} | {std_dev_precision:.4f} |\n")
    file.write(f"| Accuracy  | {mean_accuracy:.4f} | {variance_accuracy:.4f} | {std_dev_accuracy:.4f} |\n")
    file.write(f"| AUC-ROC   | {mean_roc_auc:.4f} | {variance_roc_auc:.4f} | {std_dev_roc_auc:.4f} |\n")

# Print a message indicating that the data has been written to the file
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
end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

print('===========================================')