import lightgbm as lgb
import os
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV, RepeatedKFold,cross_validate
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve,confusion_matrix,make_scorer,accuracy_score, f1_score,recall_score,precision_score,roc_auc_score
from scipy.stats import randint, uniform
import joblib

print("=================- Random Search with Oil Spill -================")
# arff_to_csv('phpgEDZOc.arff', 'output.csv')
df = pd.read_csv('/Users/umarkhan/theback/datasets/oilspill/oilspill.csv') 
# unique_values = df['class'].unique()
# print(unique_values)
def plot_class_distribution(data):
    data['class'] = data['class'].astype(str)

    # Define a dictionary to map old class labels to new labels
    label_map = {"'1'": 'spill', "'-1'": 'no spill'}

    # Map the class labels using the dictionary
    data['class'] = data['class'].map(label_map)
    print(data.head())
    
    # Define colors
    colors = ['#ADD8E6', '#FA8072', '#90EE90', '#FFA07A', '#87CEEB', '#FF69B4', '#FFD700', '#00FF00', '#FF00FF', '#00FFFF'] 
    # Plot class distribution
    data['class'].value_counts().plot(kind='bar', color=colors)
    plt.title('Class Distribution for oil spill')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=360, ha='right')
    plt.savefig('/Users/umarkhan/Bayesian/datasets/target_class dist fig/oilspill_dist')
    plt.show()

plot_class_distribution(df)


  
# le = LabelEncoder()
    
# X = df.drop('class',axis=1)
# y = le.fit_transform(df['class'])    
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y) 

# rkf = RepeatedKFold(n_splits=5, n_repeats=3,random_state=69)

# spw= len(y_train[y_train == 0]) / len(y_train[y_train == 1])


# print(f"SpW values = {spw}")

# fixed_params = {
#     'objective':'',
#     'boosting_type':'gbdt',
#     'metric':'binary_logloss',
#     'n_jobs':-1,
#     # 'scale_pos_weight':spw,
#     'is_unbalance':True,
#     'verbose':-1
#     }


# param_dist = {
#     'num_leaves': randint(31, 80),
#     'n_estimators': randint(50, 150),
#     'learning_rate': uniform(0.05, 0.1)
# }

# scoring = {
#     'f1': make_scorer(f1_score, average='weighted'),
#     'recall': make_scorer(recall_score, average='weighted'),
#     'precision': make_scorer(precision_score, average='weighted'),
#     'accuracy': make_scorer(accuracy_score),
#     'roc_auc': make_scorer(roc_auc_score, average='weighted')
# }

# lgb_classifier = lgb.LGBMClassifier(**fixed_params)


# # Use randomSearchCV for hyperparameter tuning and cross-validation
# random_search = RandomizedSearchCV(
#     estimator=lgb_classifier,
#     param_distributions=param_dist,
#     cv=rkf,
#     scoring=scoring,
#     n_iter= 27,
#     random_state=69,  
#     n_jobs=-1,
#     refit='f1'
# )

# # Fit the random search to the training data
# random_search.fit(X_train, y_train)

# # Get the best model from the random search
# best_model = random_search.best_estimator_
# best_params = random_search.best_params_

# print("Best Model Parameters:", best_params)

# # Access the results of each hyperparameter set
# results = pd.DataFrame(random_search.cv_results_)
# results.to_csv('per_param_results.txt', sep='\t', index=True)

# # Use cross_val_score to perform cross-validation with the best model and get individual fold scores
# scores = cross_validate(
#     best_model, X_train, y_train, cv=rkf, scoring=scoring, n_jobs=-1
# )

# # Extract individual fold scores
# f1_scores = scores['test_f1']
# recall_scores = scores['test_recall']
# precision_scores = scores['test_precision']
# accuracy_scores = scores['test_accuracy']
# roc_auc_scores = scores['test_roc_auc']


# fold_results = pd.DataFrame({
#     'F1': f1_scores,
#     'Recall': recall_scores,
#     'Precision': precision_scores,
#     'Accuracy': accuracy_scores,
#     'AUC-ROC': roc_auc_scores
# })

# # Get the file path in the current directory
# output_file_path = os.path.join(os.getcwd(), 'oilspill_random_ppr.csv')

# # Save the individual fold scores to a CSV file
# fold_results.to_csv(output_file_path, index=False)


# # Calculate mean, variance, and standard deviation
# mean_f1 = f1_scores.mean()
# variance_f1 = f1_scores.var()
# std_dev_f1 = f1_scores.std()
# min_f1 = f1_scores.min()
# max_f1 = f1_scores.max()

# mean_recall = recall_scores.mean()
# variance_recall = recall_scores.var()
# std_dev_recall = recall_scores.std()
# min_rcall = recall_scores.min()
# max_recall = recall_scores.max()

# mean_precision = precision_scores.mean()
# variance_precision = precision_scores.var()
# std_dev_precision = precision_scores.std()
# min_precision = precision_scores.min()
# max_precision = precision_scores.max()

# mean_accuracy = accuracy_scores.mean()
# variance_accuracy = accuracy_scores.var()
# std_dev_accuracy = accuracy_scores.std()
# min_accuracy_score = accuracy_scores.min()
# max_accuracy_score = accuracy_scores.max()

# mean_roc_auc = roc_auc_scores.mean()
# variance_roc_auc = roc_auc_scores.var()
# std_dev_roc_auc = roc_auc_scores.std()
# min_roc_auc_score = roc_auc_scores.min()
# max_roc_auc_score = roc_auc_scores.max()


# # # Print or use the calculated statistics as needed
# # print(f"Mean F1: {mean_f1:.4f}, Variance F1: {variance_f1:.4f}, Std Dev F1: {std_dev_f1:.4f}")
# # print(f"Mean Recall: {mean_recall:.4f}, Variance Recall: {variance_recall:.4f}, Std Dev Recall: {std_dev_recall:.4f}")
# # print(f"Mean Precision: {mean_precision:.4f}, Variance Precision: {variance_precision:.4f}, Std Dev Precision: {std_dev_precision:.4f}")
# # print(f"Mean Accuracy: {mean_accuracy:.4f}, Variance Accuracy: {variance_accuracy:.4f}, Std Dev Accuracy: {std_dev_accuracy:.4f}")
# # print(f"Mean AUC-ROC: {mean_roc_auc:.4f}, Variance AUC-ROC: {variance_roc_auc:.4f}, Std Dev AUC-ROC: {std_dev_roc_auc:.4f}")

# current_directory = os.getcwd()
# # Define the file name
# output_file_name = 'oilspill_random_metrics_output.md'

# # Combine the current directory with the file name
# output_file_path = os.path.join(current_directory, output_file_name)

# # Open the file in write mode
# with open(output_file_path, 'w') as file:
#     # Write the metrics to the file in Markdown table format
#     file.write( " Metric    | Mean | Variance | Std Dev | Min | Max \n")
#     file.write( "-----------|------|----------|---------|\n")
#     file.write(f" F1        | {mean_f1:.4f} | {variance_f1:.4f} | {std_dev_f1:.4f} | {min_f1:.4f} | {max_f1:.4f} \n")
#     file.write(f" Recall    | {mean_recall:.4f} | {variance_recall:.4f} | {std_dev_recall:.4f} | {min_rcall:.4f} | {max_recall:.4f} \n")
#     file.write(f" Precision | {mean_precision:.4f} | {variance_precision:.4f} | {std_dev_precision:.4f} | {min_precision:.4f} | {max_precision:.4f} \n")
#     file.write(f" Accuracy  | {mean_accuracy:.4f} | {variance_accuracy:.4f} | {std_dev_accuracy:.4f} | {min_accuracy_score:.4f} | {max_accuracy_score:.4f} \n")
#     file.write(f" AUC-ROC   | {mean_roc_auc:.4f} | {variance_roc_auc:.4f} | {std_dev_roc_auc:.4f} | {min_roc_auc_score:.4f} | {max_roc_auc_score:.4f} \n")

# # Print a message indicating that the data has been written to the file
# print(f"Metrics have been written to: {output_file_path}")


# # Make predictions on the test data
# y_pred = best_model.predict(X_test)
# y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# # Evaluate the model's performance
# accuracy = accuracy_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_pred_proba)
# f1 = f1_score(y_test, y_pred, average='binary')

# print(f"Test Accuracy: {accuracy:.4f}")
# print(f"AUC-ROC: {roc_auc:.4f}")
# print(f"F1 Score: {f1:.4f}")

# # Save AUC-ROC plot
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.savefig('oilspill_random_auc_roc_plot.png')

# # Save Confusion Matrix plot
# conf_matrix = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.savefig('oilspill_random_cm_plot.png')