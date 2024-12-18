import os
import pandas as pd

def calculate_statistics(df):
    statistics = {
        'Mean': df.mean(),
        'Variance': df.var(),
        'Std Dev': df.std(),
        'Min': df.min(),
        'Max': df.max(),
        'Mode': df.mode().iloc[0],
        'Median': df.median()
    }
    return pd.DataFrame(statistics)

def format_caption(file_name):
    dataset_name = file_name.split('_')[0]
    new_caption = f"Metrics Summary for {dataset_name.capitalize()} with random search"
    return new_caption

def append_caption_to_latex(file_path, caption, label):
    with open(file_path, 'a') as f:
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table}\n")

def read_csv_files(folder_path):
    table_folder = "latex_tables3"
    os.makedirs(table_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            file_statistics = calculate_statistics(df)
            table_name = f"table_{file_name.split('.')[0]}.tex"
            table_path = os.path.join(table_folder, table_name)
            caption = format_caption(file_name)
            label = f"tab:metrics_summary_{file_name.split('.')[0]}"

            # Generate LaTeX table without caption and label, but start table environment
            with open(table_path, 'w') as f:
                f.write("\\begin{table}[!htbp]\n")
                f.write("\\centering\n")
                file_statistics.to_latex(f, index=False, longtable=False, escape=False)
            
            # Append caption and label at the end
            append_caption_to_latex(table_path, caption, label)

if __name__ == "__main__":
    folder_path = "/Users/umarkhan/theback/pfr2"
    read_csv_files(folder_path)
