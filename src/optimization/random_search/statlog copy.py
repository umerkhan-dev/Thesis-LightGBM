import os
import lightgbm as lgb
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split,RandomizedSearchCV,RepeatedKFold,cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve,confusion_matrix,make_scorer,accuracy_score, f1_score,recall_score,precision_score,roc_auc_score
from matplotlib import pyplot as plt
from scipy.stats import randint, uniform
import joblib


print("========Statlog(Land Sat)=========")

# Load the dataset
train_data_file = 'datasets/statlog+landsat+satellite/sat.trn'
test_data_file = 'datasets/statlog+landsat+satellite/sat.tst'

columns = [
    'TopLeft1', 'TopLeft2', 'TopLeft3', 'TopLeft4',
    'TopMiddle1', 'TopMiddle2', 'TopMiddle3', 'TopMiddle4',
    'TopRight1', 'TopRight2', 'TopRight3', 'TopRight4',
    'MiddleLeft1', 'MiddleLeft2', 'MiddleLeft3', 'MiddleLeft4',
    'Center1', 'Center2', 'Center3', 'Center4',
    'MiddleRight1', 'MiddleRight2', 'MiddleRight3', 'MiddleRight4',
    'BottomLeft1', 'BottomLeft2', 'BottomLeft3', 'BottomLeft4',
    'BottomMiddle1', 'BottomMiddle2', 'BottomMiddle3', 'BottomMiddle4',
    'BottomRight1', 'BottomRight2', 'BottomRight3', 'BottomRight4',
    'Class'
]

df_trn = pd.read_csv(train_data_file, delim_whitespace=' ', names=columns)
df_tst = pd.read_csv(test_data_file, delim_whitespace=' ', names=columns)

X_train, y_train = df_trn.drop('Class', axis=1), df_trn['Class']
X_test, y_test = df_tst.drop('Class', axis=1), df_tst['Class']

# Repeated K-Fold Cross Validation
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

# Fixed parameters for LightGBM
fixed_params = {
    'objective': 'multiclass',
    'num_class': len(df_trn['Class'].unique()),  # Number of classes
    'boosting_type': 'gbdt',
    'metric': 'multi_logloss',
    'n_jobs': -1,
    'verbose': -1
}

param_dist = {
    'num_leaves': randint(31, 80),
    'n_estimators': randint(50, 150),
    'learning_rate': uniform(0.05, 0.15)
}
# Custom scoring function for ROC AUC in a multi-class setting
def roc_auc_ovr_scorer(estimator, X, y):
    y_pred_proba = estimator.predict_proba(X)
    roc_auc_per_class = []
    for i in range(len(estimator.classes_)):
        roc_auc_class = roc_auc_score(y == i, y_pred_proba[:, i])
        roc_auc_per_class.append(roc_auc_class)
    print("roc_auc_pc: ", roc_auc_per_class)
    return np.mean(roc_auc_per_class)



# Scoring dictionary for cross-validation
scoring = {
    'f1_weighted': make_scorer(f1_score, average='weighted'),
    'recall_weighted': make_scorer(recall_score, average='weighted'),
    'precision_weighted': make_scorer(precision_score, average='weighted'),
    'accuracy': make_scorer(accuracy_score),
    'roc_auc_ovr': roc_auc_ovr_scorer
}

# Initialize LightGBM classifier
lgb_classifier = lgb.LGBMClassifier(**fixed_params)

random_search = RandomizedSearchCV(
    estimator=lgb_classifier,
    param_distributions=param_dist,
    cv=rkf,
    scoring='f1_weighted',
    n_iter=27,  # Number of random combinations to try
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# Get the best model and parameters
best_model = random_search.best_estimator_
best_params = random_search.best_params_

joblib.dump(best_model, 'statlog_bst_rand.joblib')

print("Best Model Parameters:", best_params)

# Access the results of each hyperparameter set
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('statlog_random_ppr.txt', sep='\t', index=True)

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
output_file_path = os.path.join(os.getcwd(), 'statlog_random_per_fold_results.csv')
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
output_file_name = 'statlog_risk_random_metrics_output.md'

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
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.savefig('statlog_risk_random_auc_roc_plot.png')


# Save Confusion Matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('statlog_risk_random_marketing_plot.png')

