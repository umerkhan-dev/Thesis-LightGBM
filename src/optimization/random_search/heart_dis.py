import lightgbm as lgb
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold,cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer,accuracy_score,roc_curve, f1_score,recall_score,precision_score,roc_auc_score,confusion_matrix
from scipy.stats import randint, uniform
import numpy as np
import os
import joblib
import seaborn as sns

print("============- Heart cleavland -================")

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
test = test.replace("?",np.nan)


# Assuming df_test (Hungarian) and df_data (Cleveland) are your DataFrames

# Convert 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'slope' to float in test (Hungarian) DataFrame
columns_to_convert_test = ['trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'slope']
test[columns_to_convert_test] = test[columns_to_convert_test].astype(float)

# Convert 'ca' and 'thal' to float in training (Cleveland) DataFrame
columns_to_convert_data = ['ca', 'thal']
data[columns_to_convert_data] = data[columns_to_convert_data].astype(float)


# print("Test (Hungarian):\n", test.dtypes)
# print("\nTraining (Cleveland):\n", data.dtypes)



data["ca"] = pd.to_numeric(data["ca"], errors='coerce')
data["thal"] = pd.to_numeric(data["thal"], errors='coerce')

test["ca"] = pd.to_numeric(test["ca"], errors='coerce')
test["thal"] = pd.to_numeric(test["thal"], errors='coerce')

# Convert target to integer (if it's not binary, you should adjust this part)
data["target"] = data["target"].apply(lambda x: 1 if x > 0 else 0)
test["target"] = test["target"].apply(lambda x: 1 if x > 0 else 0)

# Split the data into features (X) and target (y)
X = data.drop("target", axis=1)
y = data["target"]

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X
y_train = y

X_test = test.drop("target",axis=1)
y_test = test["target"]

fixed_params={'objective':'binary',
               'boosting_type':'gbdt',
                'metric':'binary_logloss',
                  'n_jobs':-1}



lgb_classifier = lgb.LGBMClassifier(**fixed_params, verbose=-1)

# Create a RepeatedKFold object
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=69)

# Define the parameter distributions for random search
param_dist = {
    'num_leaves': randint(31, 90),
    'n_estimators': randint(50, 150),
    'learning_rate': uniform(0.05, 0.15)
}

scoring = {
    'f1': make_scorer(f1_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'precision': make_scorer(precision_score, average='weighted'),
    'accuracy': make_scorer(accuracy_score),
    'roc_auc': make_scorer(roc_auc_score, average='weighted')
}

# Use RandomizedSearchCV for hyperparameter tuning and cross-validation
random_search = RandomizedSearchCV(
    estimator=lgb_classifier,
    param_distributions=param_dist,
    cv=rkf,
    scoring='f1_weighted',
    n_iter=27,  # Number of random combinations to try
    random_state=69,
    n_jobs=-1
)

# Fit the random search to the training data
random_search.fit(X_train, y_train)

# Get the best model from the random search
best_model = random_search.best_estimator_
best_params = random_search.best_params_

joblib.dump(best_model, 'heart_bst_rand.joblib')

print("Best Model Parameters:", best_params)

# Access the results of each hyperparameter set
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('Heart_random_ppr.txt', sep='\t', index=True)

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
output_file_path = os.path.join(os.getcwd(), 'heart_random_per_fold_results.csv')

# Save the individual fold scores to a CSV file
fold_results.to_csv(output_file_path, index=False)


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
output_file_name = 'heart_grid_metrics_output.md'

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
plt.savefig('credit_risk_random_auc_roc_plot.png')

# Save Confusion Matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('credit_risk_random_marketing_plot.png')





# # Define a parameter grid to search over
# param_grid = {
#     'num_leaves': [20, 31, 60],
#     'learning_rate': [0.05, 0.1, 0.15],
#     'n_estimators': [50, 100, 150]
# }

# # Create a LightGBM classifier
# lgb_classifier = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', random_state=42)

# # Create GridSearchCV object
# grid_search = GridSearchCV(estimator=lgb_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# # Perform the search
# grid_search.fit(X_train, y_train)

# # Output the best parameters
# print(f'Best parameters found by grid search are: {grid_search.best_params_}')

# # Use the best estimator to make predictions
# y_pred = grid_search.best_estimator_.predict(X_test)

# # Evaluate the predictions
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy of the best classifier after GridSearchCV: {accuracy * 100:.2f}%')
