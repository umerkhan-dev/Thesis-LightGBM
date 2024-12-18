import pandas as pd
from matplotlib import pyplot as plt
import os

directory= '/Users/umarkhan/hyperband/results/params'
saveAt= '/Users/umarkhan/hyperband/output_figures/WithoutAccFig/'
def savefig(fp):
    df = pd.read_csv(fp)
    print(df.head())
    hyperparameters = df[['learning_rate', 'num_iterations', 'num_leaves']]

    # Create a figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loop through each hyperparameter column
    for i, column in enumerate(hyperparameters.columns):
        ax = axes[i]
        ax.hist(hyperparameters[column], bins=20, color='skyblue', edgecolor='black')
        ax.set_title(column)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    # Add a separate title for the entire plot
    # fig.suptitle('Parameter Histograms', fontsize=16, y=0.175)
    plt.subplots_adjust(bottom=0.3)
    # plt.tight_layout()  # Adjust subplot layout
    plt.show()
def save_fig(file_path, output_folder):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        df_renamed = df.rename(columns={'num_iterations': 'n_estimators'})
        # Extract the hyperparameters with the new column name
        hyperparameters = df_renamed[['learning_rate','n_estimators','num_leaves']]
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

os.makedirs(saveAt, exist_ok=True)

# Iterate through files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        save_fig(file_path, saveAt)





'''csv_file_path = 'results/params/heart_hyper_param.csv'
# List to store cleaned lines
cleaned_lines = []

# Open the CSV file for reading
with open(csv_file_path, 'r') as file:
    # Read each line in the file
    for line in file:
        # Strip trailing whitespace (including the trailing comma)
        cleaned_line = line.rstrip(',\n')
        # Append cleaned line to the list
        cleaned_lines.append(cleaned_line)

# Open the same CSV file for writing (this will overwrite the original file)
with open(csv_file_path, 'w') as file:
    # Write the cleaned lines back to the file
    for cleaned_line in cleaned_lines:
        file.write(cleaned_line + '\n')

print(f"Original CSV file '{csv_file_path}' has been updated.")'''
