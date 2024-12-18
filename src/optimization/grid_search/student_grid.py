import lightgbm as lgb
import os
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV, RepeatedKFold,cross_validate
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve,confusion_matrix,make_scorer,accuracy_score, f1_score,recall_score,precision_score,roc_auc_score
from scipy.stats import randint, uniform
import joblib


print("================- Student Droput -=================")
dfile_path = '/Users/umarkhan/theback/datasets/student_dropout/student_data.csv'

df = pd.read_csv(dfile_path,delimiter=';')

le =LabelEncoder()

X = df.drop('Target',axis = 1)
y = le.fit_transform(df['Target'])



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=69,stratify=y)

rkf = RepeatedKFold(n_splits=5,n_repeats=3, random_state=69)

fixed_params = {
    'objective': 'multiclass',
    'num_class': len(df['Target'].unique()),  # Number of classes
    'boosting_type': 'gbdt',
    'metric': 'multi_logloss',
    'n_jobs': -1,
    'verbose': -1
}

param_grid = {
    'num_leaves': [31, 50, 80],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.15, 0.05, 0.1]
}

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


# Initialize LightGBM classifier
lgb_classifier = lgb.LGBMClassifier(**fixed_params)
# Using GridSearchCV for hyperparameter tuning and cross-validation
grid_search = GridSearchCV(
    estimator=lgb_classifier,
    param_grid=param_grid,
    cv=rkf,
    scoring=scoring,
    refit='f1_weighted',  # Refit the model using the best parameters for F1 score
    n_jobs=-1
)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_


joblib.dump(best_model, 'student_bst_grid.joblib')

print("Best Model Parameters:", best_params)

# Access the results of each hyperparameter set
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv('student_grid_ppr.txt')

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
output_file_path = os.path.join(os.getcwd(), 'student_grid_per_fold_results.csv')
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
output_file_name = 'student_grid_metrics_output.md'

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
plt.savefig('student_grid_auc_roc_plot.png')


# Save Confusion Matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('student_grid_marketing_plot.png')

