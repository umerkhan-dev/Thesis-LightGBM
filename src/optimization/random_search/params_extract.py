# import pandas as pd
# from tabulate import tabulate
# # Read data from file
# df = pd.read_csv("/Users/umarkhan/theback/results/apple_random_ppr.txt", delimiter='\t')


# # Extract hyperparameter values
# hyperparameters = df[['param_learning_rate', 'param_n_estimators', 'param_num_leaves','mean_test_score']]

# # Convert hyperparameters to a list of lists for tabulate
# hyperparameters_table = [["param_learning_rate", "param_n_estimators", "param_num_leaves",'mean_test_score']]
# for index, row in hyperparameters.iterrows():
#     hyperparameters_table.append([row['param_learning_rate'], row['param_n_estimators'], row['param_num_leaves'],row['mean_test_score']])

# # Print hyperparameters table
# print(tabulate(hyperparameters_table, headers="firstrow", tablefmt="grid"))

import pandas as pd
import matplotlib.pyplot as plt

# Assuming your data is already in a dataframe df
df = pd.read_csv("results/breas_cancer_per_param_results.txt", delimiter='\t')

# Select hyperparameters columns
hyperparameters = df[['param_learning_rate', 'param_n_estimators', 'param_num_leaves']]
# Select metric column
metric = df['mean_test_score']

# Plot histograms for each hyperparameter
plt.figure(figsize=(15, 5))
colors = ['lightgreen','salmon','skyblue']
for i, column in enumerate(hyperparameters.columns, 1):
    plt.subplot(1, 3, i)
    plt.hist(hyperparameters[column], bins=20, color=colors[i-1], edgecolor='black')
    plt.title(column.replace('param_', ''))
    plt.xlabel('Value')
    plt.ylabel('Frequency')

# Plot histogram for the metric
# plt.subplot(1, 4, 4)
# plt.hist(metric, bins=20, color='salmon', edgecolor='black')
# plt.title('mean_test_score')
# plt.xlabel('Value')
# plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
