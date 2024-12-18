import lightgbm as lgb
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate,train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score,roc_curve,make_scorer, f1_score,recall_score,precision_score,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from scipy.io.arff import loadarff 
data, meta = loadarff('/Users/umarkhan/Downloads/PhishingData.arff')

df = pd.DataFrame(data)
df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
object_cols = df.select_dtypes(include=['object']).columns
df[object_cols] = df[object_cols].apply(lambda x: x.astype('category'))

le = LabelEncoder()
X = df.drop('Result',axis=1)
y = le.fit_transform(df['Result'])


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.2, random_state = 69)

rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)


fixed_params = {
    'objective': 'multiclass',
    'num_class': len(df['Result'].unique()),  # Number of classes
    'boosting_type': 'gbdt',
    'metric': 'multi_logloss',
    'n_jobs': -1,
    'verbose': -1
}


param_grid = {
    'num_leaves': [31, 60, 80],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.15]
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

lgb_classifier = lgb.LGBMClassifier(**fixed_params)

# Set up GridSearchCV for LightGBM
grid_search = GridSearchCV(estimator=lgb_classifier, param_grid=param_grid,
                           cv=rkf,
                           scoring='f1_weighted')

# Fit the model on the training data
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best Model Parameters:", best_params)

# Access the results of each hyperparameter set
results = pd.DataFrame(grid_search.cv_results_)
results.to_csv('phishing_grid_ppr.txt', sep='\t', index=True)
# # Print the results or use them as needed
# print(results[['params', 'mean_test_f1_weighted', 'std_test_f1_weighted', 'mean_test_recall_weighted', 'std_test_recall_weighted',
#                'mean_test_precision_weighted', 'std_test_precision_weighted', 'mean_test_accuracy', 'std_test_accuracy',
#                'mean_test_roc_auc_weighted', 'std_test_roc_auc_weighted']])




# Use cross_val_score to perform cross-validation with the best model and get individual fold scores
scores = cross_validate(
    best_model, X_train, y_train, cv=rkf, scoring=scoring, n_jobs=-1
)

# Extract individual fold scores
f1_scores = scores['test_f1_weighted']
recall_scores = scores['test_recall_weighted']
precision_scores = scores['test_precision_weighted']
accuracy_scores = scores['test_accuracy']
roc_auc_scores = scores['test_roc_auc_ovr'] 


fold_results = pd.DataFrame({
    'F1': f1_scores,
    'Recall': recall_scores,
    'Precision': precision_scores,
    'Accuracy': accuracy_scores,
    'AUC-ROC': roc_auc_scores
})

# Get the file path in the current directory
output_file_path = os.path.join(os.getcwd(), 'phishing_grid_per_fold_results.csv')

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
output_file_name = 'phishing_grid_metrics_output.md'

# Combine the current directory with the file name
output_file_path = os.path.join(current_directory, output_file_name)

# Open the file in write mode
with open(output_file_path, 'w') as file:
    # Write the metrics to the file in Markdown table format
    file.write( "| Metric    | Mean   | Variance | Std Dev |\n")
    file.write( "|-----------|--------|----------|---------|\n")
    file.write(f"| F1        | {mean_f1:.4f} | {variance_f1:.4f}   | {std_dev_f1:.4f}  |\n")
    file.write(f"| Recall    | {mean_recall:.4f} | {variance_recall:.4f}   | {std_dev_recall:.4f}  |\n")
    file.write(f"| Precision | {mean_precision:.4f} | {variance_precision:.4f}   | {std_dev_precision:.4f}  |\n")
    file.write(f"| Accuracy  | {mean_accuracy:.4f} | {variance_accuracy:.4f}   | {std_dev_accuracy:.4f}  |\n")
    file.write(f"| AUC-ROC   | {mean_roc_auc:.4f} | {variance_roc_auc:.4f}   | {std_dev_roc_auc:.4f}  |\n")




# Print a message indicating that the data has been written to the file
print(f"Metrics have been written to: {output_file_path}")


# Make predictions on the test data
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Use one-vs-rest strategy for ROC AUC calculation
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_rep)
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

plt.figure(figsize=(8, 8))
colors = ['darkorange', 'green', 'blue', 'purple', 'brown']  # Adjust the colors as needed

for i, color in zip(range(len(df['Result'].unique())), colors):
    fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)  # Calculate AUC for each class
    plt.plot(fpr, tpr, color=color, lw=2, label=f'Result {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (One-vs-Rest)')
plt.legend(loc='lower right')
plt.savefig('phishing_grid_auc_roc_plot.png')
plt.show()

# Save Confusion Matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('cm_phishing_grid_plot.png')
