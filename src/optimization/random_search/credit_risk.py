import os
import lightgbm as lgb
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedKFold,cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve,confusion_matrix,make_scorer,accuracy_score, f1_score,recall_score,precision_score,roc_auc_score
from sklearn.metrics import auc,roc_curve,confusion_matrix,make_scorer,accuracy_score, f1_score,recall_score,precision_score,roc_auc_score
from matplotlib import pyplot as plt
data_train_f2 = '/Users/umarkhan/Thesis/Datasets/credit_risk/german.data'

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=69)

# Example cost matrix for binary classification
# cost_fp = 5
# cost_fn = 1
# sample_weight = np.where(y_train == 1, cost_fn, cost_fp)
categorical_feature_indices = [X.columns.get_loc(col) for col in categorical_col]

rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)


spw= len(y_train[y_train == 0]) / len(y_train[y_train == 1])

fixed_params={
    'objective':'binary',
    'boosting_type':'gbdt',
    'metric':'binary_logloss',
    'n_jobs':-1,
    'scale_pos_weight':spw,
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

# Use GridSearchCV for hyperparameter tuning and cross-validation
grid_search = GridSearchCV(
    estimator=lgb_classifier,
    param_grid=param_grid,
    cv=rkf,
    scoring=scoring,
    refit='f1',  # Refit the model using the best parameters for F1 score
    n_jobs=-1
)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best Model Parameters:", best_params)

# Access the results of each hyperparameter set
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv('credit_risk_ppr.txt', sep='\t', index=True)
# Print the results or use them as needed
print(results[['params', 'mean_test_f1', 'std_test_f1', 'mean_test_recall', 'std_test_recall',
               'mean_test_precision', 'std_test_precision', 'mean_test_accuracy', 'std_test_accuracy',
               'mean_test_roc_auc', 'std_test_roc_auc']])


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
output_file_path = os.path.join(os.getcwd(), 'credir_risk_per_fold_results.csv')

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


# # Print or use the calculated statistics as needed
# print(f"Mean F1: {mean_f1:.4f}, Variance F1: {variance_f1:.4f}, Std Dev F1: {std_dev_f1:.4f}")
# print(f"Mean Recall: {mean_recall:.4f}, Variance Recall: {variance_recall:.4f}, Std Dev Recall: {std_dev_recall:.4f}")
# print(f"Mean Precision: {mean_precision:.4f}, Variance Precision: {variance_precision:.4f}, Std Dev Precision: {std_dev_precision:.4f}")
# print(f"Mean Accuracy: {mean_accuracy:.4f}, Variance Accuracy: {variance_accuracy:.4f}, Std Dev Accuracy: {std_dev_accuracy:.4f}")
# print(f"Mean AUC-ROC: {mean_roc_auc:.4f}, Variance AUC-ROC: {variance_roc_auc:.4f}, Std Dev AUC-ROC: {std_dev_roc_auc:.4f}")

current_directory = os.getcwd()
# Define the file name
output_file_name = 'credit_risk_grid_metrics_output.md'

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
plt.savefig('credit_risk_grid_auc_roc_plot.png')

# Save Confusion Matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('credit_risk_grid_marketing_plot.png')


'''
# # Create the model
# model = lgb.LGBMClassifier(**fixed_params,verbose=-1)
# model.fit(
#     X_train, y_train,
#     eval_metric='logloss',
#     feature_name='auto',
# )
# rkf = RepeatedKFold(n_splits=5,n_repeats=3,random_state=69)
# param_grid = {
#     'num_leaves': [31, 60, 90],
#     'n_estimators': [50, 100, 150],
#     'learning_rate': [0.15, 0.05, 0.1]
# }
# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     cv=rkf,  # Use RepeatedKFold cross-validation
#     scoring='f1_weighted',
#     n_jobs=-1  # Utilize all available CPU cores
#     # fit_params={'callbacks': [early_stopping]}  # Add early stopping callback
# )

# grid_search.fit(X_train,y_train)


# best_model = grid_search.best_estimator_
# best_params = grid_search.best_params_

# print("Best Model Parameters:", best_params)

# # After fitting, the model can be used to make predictions
# predictions = best_model.predict(X_test)
# y_pred_proba = best_model.predict_proba(X_test)[:, 1]



# # Calculate and print accuracy
# acc = accuracy_score(y_test, predictions)
# print("Test Accuracy:", acc)

# # Calculate and print confusion matrix
# print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# # Calculate and print AUC-ROC score
# roc_auc = roc_auc_score(y_test, y_pred_proba)
# print("AUC-ROC Score:", roc_auc)


# fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

# # Calculate AUC-ROC
# roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.savefig("Credit Risk ROC_AUC.png")
# plt.show()
'''