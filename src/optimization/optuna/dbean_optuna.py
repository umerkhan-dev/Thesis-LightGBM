import os
import lightgbm as lgb
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate,train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, confusion_matrix, make_scorer, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from matplotlib import pyplot as plt
import joblib
import optuna
from scipy.io.arff import loadarff
print("========- Dry Bean -=========")

# Load the dataset
data, meta = loadarff('/Users/umarkhan/Downloads/DryBeanDataset/Dry_Bean_Dataset.arff')

df = pd.DataFrame(data)
df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
# print(meta)
# print(df['Class'].unique())
# def plot_class_distribution(data):
    
#     # Define colors
#     colors = ['#ADD8E6', '#FA8072', '#90EE90', '#FFA07A', '#87CEEB', '#FF69B4', '#FFD700', '#00FF00', '#FF00FF', '#00FFFF'] 
#     # Plot class distribution
#     data['Class'].value_counts().plot(kind='bar', color=colors)
#     plt.title('Class Distribution for Dry Beans')
#     plt.xlabel('Class')
#     plt.ylabel('Count')
#     plt.xticks(rotation=45, ha='right')
#     plt.subplots_adjust(bottom=0.25)
#     plt.savefig('/Users/umarkhan/Bayesian/datasets/target_class dist fig/drybean_class_dist.png')
#     plt.show()

# plot_class_distribution(df)

le = LabelEncoder()
X = df.drop('Class',axis=1)
y = le.fit_transform(df['Class'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.2, random_state = 69)

# Repeated K-Fold Cross Validation
rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# Fixed parameters for LightGBM
fixed_params = {
    'objective': 'multiclass',
    'num_class': len(df['Class'].unique()),  # Number of classes
    'boosting_type': 'gbdt',
    'metric': 'multi_logloss',
    'n_jobs': -1,
    'verbose': -1
}

# Custom scoring function for ROC AUC in a multi-class setting
def roc_auc_ovr_scorer(estimator, X, y):
    y_pred_proba = estimator.predict_proba(X)
    return roc_auc_score(y, y_pred_proba, multi_class='ovr')
# Scoring dictionary for cross-validation
scoring = {
    'f1_weighted': make_scorer(f1_score, average='weighted'),
    'recall_weighted': make_scorer(recall_score, average='weighted'),
    'precision_weighted': make_scorer(precision_score, average='weighted'),
    'accuracy': make_scorer(accuracy_score),
    'roc_auc_ovr': roc_auc_ovr_scorer
}

def objective(trial):
    param = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 80),
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),
        **fixed_params
    }
    lgb_classifier = lgb.LGBMClassifier(**param)
    scores = cross_validate(lgb_classifier, X_train, y_train, scoring=scoring, cv=rkf, n_jobs=-1)
    roc_auc_ovr_scores = scores['test_roc_auc_ovr']
    return np.mean(roc_auc_ovr_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=27)

# Access the results of each hyperparameter set
study_results = pd.DataFrame([{
    "number": trial.number,
    "value": trial.value,
    "params": trial.params,
    "state": trial.state
} for trial in study.trials])
study_results.to_csv('dBean_optuna_ppr.csv', index=False)

# Get the best model and parameters
best_model = lgb.LGBMClassifier(**study.best_params, **fixed_params)
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'dBean_bst_optuna.joblib')
print("Best Model Parameters:", study.best_params)

# Use cross_validate to perform cross-validation with the best model and get individual fold scores
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
output_file_path = os.path.join(os.getcwd(), 'dBean_optuna_per_fold_results.csv')
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
output_file_name = 'dBean_risk_optuna_metrics_output.md'

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
plt.savefig('dBean_risk_random_auc_roc_plot.png')


# Save Confusion Matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('dBean_risk_random_marketing_plot.png')

