from joblib import load

# Paths to the saved models, assuming HyperBand model path is not available yet
model_paths = {
    'Grid Search': '/Users/umarkhan/Desktop/grid_search/code/Models/heart_bst_grid.joblib',
    'Random Search': '/Users/umarkhan/theback/heart_bst_rand.joblib',
    'Optuna': '/Users/umarkhan/Bayesian/Credit_risk_bst_optuna.joblib'
    # HyperBand model is omitted since it's a placeholder
}

# Initialize a dictionary to hold the hyperparameters
hyperparameters = {
    'learning_rate': {},
    'n_estimators': {},
    'num_leaves': {}
}

# Placeholder values for HyperBand
hyperparameters['learning_rate']['HyperBand'] = 'TBD'
hyperparameters['n_estimators']['HyperBand'] = 'TBD'
hyperparameters['num_leaves']['HyperBand'] = 'TBD'

# Load each model and extract the hyperparameters
for method, path in model_paths.items():
    model = load(path)
    hyperparameters['learning_rate'][method] = model.get_params().get('learning_rate', 'N/A')
    hyperparameters['n_estimators'][method] = model.get_params().get('n_estimators', 'N/A')
    hyperparameters['num_leaves'][method] = model.get_params().get('num_leaves', 'N/A')

# Generate LaTeX table with a placeholder for HyperBand
table_latex = """
\\begin{table}[ht]
\\centering
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Parameter} & \\textbf{Grid Search} & \\textbf{Random Search} & \\textbf{Optuna} & \\textbf{HyperBand} \\\\
\\midrule
learning_rate & {lr_gs} & {lr_rs} & {lr_opt} & {lr_hb} \\\\
n_estimators & {ne_gs} & {ne_rs} & {ne_opt} & {ne_hb} \\\\
num_leaves & {nl_gs} & {nl_rs} & {nl_opt} & {nl_hb} \\\\
\\bottomrule
\\end{tabular}
\\caption{{Best Model Hyperparameters for Adult}}
\\label{{tab:best_params_adult}}
\\end{table}
""".format(
    lr_gs=hyperparameters['learning_rate']['Grid Search'],
    lr_rs=hyperparameters['learning_rate']['Random Search'],
    lr_opt=hyperparameters['learning_rate']['Optuna'],
    lr_hb=hyperparameters['learning_rate']['HyperBand'],  # Placeholder
    ne_gs=hyperparameters['n_estimators']['Grid Search'],
    ne_rs=hyperparameters['n_estimators']['Random Search'],
    ne_opt=hyperparameters['n_estimators']['Optuna'],
    ne_hb=hyperparameters['n_estimators']['HyperBand'],  # Placeholder
    nl_gs=hyperparameters['num_leaves']['Grid Search'],
    nl_rs=hyperparameters['num_leaves']['Random Search'],
    nl_opt=hyperparameters['num_leaves']['Optuna'],
    nl_hb=hyperparameters['num_leaves']['HyperBand']  # Placeholder
)

print(table_latex)
