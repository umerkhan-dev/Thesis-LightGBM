import os
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve
from scipy.stats import randint, uniform
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Directory containing CSV files
directory = '/Users/umarkhan/theback/'

# Name of the column to drop
column_to_drop = 'iter'

# Function to drop the specified column if it exists
def drop_column_if_exists(file_path, column_name):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Check if the column exists
        if column_name in df.columns:
            # Drop the column
            df.drop(columns=[column_name], inplace=True)
            
            # Save the modified DataFrame back to the file
            df.to_csv(file_path, index=False)
            print(f"Column '{column_name}' dropped successfully from '{file_path}'")
        else:
            print(f"Column '{column_name}' not found in '{file_path}'")
    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")

def save_fig(file_path, output_folder):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, delimiter='\t')
        # Rename 'num_iterations' to 'n_estimators' for consistency in plotting
        # df_renamed = df.rename(columns={'num_iterations': 'n_estimators'})

        # Extract the hyperparameters with the new column name
        hyperparameters = df[['param_learning_rate',	'param_n_estimators',	'param_num_leaves']]

        # Extract the filename without extension
        filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]

        # Plot histograms for hyperparameters
        plt.figure(figsize=(15, 5))
        for i, column in enumerate(hyperparameters.columns, 1):
            plt.subplot(1, 3, i)
            plt.hist(hyperparameters[column], bins=20, color='skyblue', edgecolor='black')
            plt.title(column.replace('param_', ''))
            plt.xlabel('Value')
            plt.ylabel('Frequency')

        # Save the figure
        output_fig_path = os.path.join(output_folder, f"{filename_without_extension}_figures.png")
        plt.savefig(output_fig_path)
        plt.close()

        print(f"Figure saved to: {output_fig_path}")

    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")

# Output folder for saving figures
output_figures_folder = '/Users/umarkhan/theback/results/output_figures/v2'
os.makedirs(output_figures_folder, exist_ok=True)

# Iterate through files in the directory
for filename in os.listdir(directory):
    if filename.endswith('ppr.txt'):
        file_path = os.path.join(directory, filename)
        save_fig(file_path, output_figures_folder)





'''def train_and_evaluate_xgb_model(X_train, y_train, X_test, y_test, dataset_name, search_type):
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

    spw = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    fixed_params = {
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'eval_metric': 'logloss',
        'n_jobs': -1,
        'scale_pos_weight': spw,
        'verbosity': 0
    }

    param_dist = {
        'max_depth': randint(3, 10),
        'n_estimators': randint(50, 150),
        'learning_rate': uniform(0.05, 0.1)
    }

    scoring = {
        'f1': make_scorer(f1_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'precision': make_scorer(precision_score, average='weighted'),
        'accuracy': make_scorer(accuracy_score),
        'roc_auc': make_scorer(roc_auc_score, average='weighted')
    }

    xgb_classifier = xgb.XGBClassifier(**fixed_params)

    # Use RandomizedSearchCV for hyperparameter tuning and cross-validation
    random_search = RandomizedSearchCV(
        estimator=xgb_classifier,
        param_distributions=param_dist,
        cv=rkf,
        scoring='f1_weighted',
        n_iter=27,  # Number of random combinations to try
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    # Get the best model from the random search
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    # Save the best model
    joblib.dump(best_model, f'{dataset_name}_{search_type}_best_xgb_model.joblib')

    print("Best Model Parameters:", best_params)

    # Access the results of each hyperparameter set
    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv(f'{dataset_name}_{search_type}_random_search_results.txt', sep='\t', index=True)

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

    # Save the individual fold scores to a CSV file
    fold_results.to_csv(f'{dataset_name}_{search_type}_fold_results.csv', index=False)

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

    # Print or use the calculated statistics as needed
    print(f"Mean F1: {mean_f1:.4f}, Variance F1: {variance_f1:.4f}, Std Dev F1: {std_dev_f1:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}, Variance Recall: {variance_recall:.4f}, Std Dev Recall: {std_dev_recall:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}, Variance Precision: {variance_precision:.4f}, Std Dev Precision: {std_dev_precision:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}, Variance Accuracy: {variance_accuracy:.4f}, Std Dev Accuracy: {std_dev_accuracy:.4f}")
    print(f"Mean AUC-ROC: {mean_roc_auc:.4f}, Variance AUC-ROC: {variance_roc_auc:.4f}, Std Dev AUC-ROC: {std_dev_roc_auc:.4f}")

    current_directory = os.getcwd()
    # Define the file name
    output_file_name = 'credit_risk_random_metrics_output.md'

    # Combine the current directory with the file name
    output_file_path = os.path.join(current_directory, output_file_name)

    # Open the file in write mode
    with open(output_file_path, 'w') as file:
        # Write the metrics to the file in Markdown table format
        file.write(" Metric    | Mean | Variance | Std Dev | Min | Max \n")
        file.write("-----------|------|----------|---------|\n")
        file.write(f" F1        | {mean_f1:.4f} | {variance_f1:.4f} | {std_dev_f1:.4f} | {min_f1:.4f} | {max_f1:.4f} \n")
        file.write(f" Recall    |
        {mean_recall:.4f} | {variance_recall:.4f} | {std_dev_recall:.4f} | {min_rcall:.4f} | {max_recall:.4f} \n")
        file.write(f" Precision | {mean_precision:.4f} | {variance_precision:.4f} | {std_dev_precision:.4f} | {min_precision:.4f} | {max_precision:.4f} \n")
        file.write(f" Accuracy  | {mean_accuracy:.4f} | {variance_accuracy:.4f} | {std_dev_accuracy:.4f} | {min_accuracy_score:.4f} | {max_accuracy_score:.4f} \n")
        file.write(f" AUC-ROC   | {mean_roc_auc:.4f} | {variance_roc_auc:.4f} | {std_dev_roc_auc:.4f} | {min_roc_auc_score:.4f} | {max_roc_auc_score:.4f} \n")

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
    plt.title(f'{dataset_name}_{search_type} Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'{dataset_name}_{search_type}_auc_roc_plot.png')

    # Save Confusion Matrix plot
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{dataset_name}_{search_type} Confusion Matrix')
    plt.savefig(f'{dataset_name}_{search_type}_confusion_matrix_plot.png')


'''