import lightgbm as lgb
import os
import seaborn as sns
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedKFold,cross_validate
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve,confusion_matrix,make_scorer,accuracy_score, f1_score,recall_score,precision_score,roc_auc_score
import time

print("==============  ADULT  =============")
start_time = time.time()

train_data_file = '~/Adult/adult.data'
test_data_file = '~/Adult/adult.test'

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
           'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
           'native_country', 'income']

df = pd.read_csv(train_data_file, header=None, names=columns, na_values=' ?')
test_df = pd.read_csv(test_data_file, header=None, names=columns, na_values=' ?')
df.drop('education', axis=1, inplace=True)  # Remove 'education' column
test_df.drop('education', axis=1, inplace=True)

cat_cols = ['workclass','marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income']

label_encoder = LabelEncoder()
for col in cat_cols:
   df[col] = label_encoder.fit_transform(df[col])
   test_df[col] = label_encoder.fit_transform(test_df[col])

X_train = df.drop('income', axis=1)
y_train = label_encoder.fit_transform(df['income'])

X_test = test_df.drop('income', axis=1)
y_test = label_encoder.fit_transform(test_df['income'])

rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)

spw= len(y_train[y_train == 0]) / len(y_train[y_train == 1])

fixed_params = {
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
results.to_csv('per_param_results.txt', sep='\t', index=True)
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
output_file_path = os.path.join(os.getcwd(), 'adult_random_per_fold_results.csv')

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
output_file_name = 'adult_grid_metrics_output.md'

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
plt.savefig('Adult_grid_auc_roc_plot.png')

# Save Confusion Matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('Adult_grid_marketing_plot.png')

end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

print('===========================================')