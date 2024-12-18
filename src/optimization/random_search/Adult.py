import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
train_data_file = '/Users/umarkhan/Thesis/Datasets/adult/adult.data'
test_data_file = '/Users/umarkhan/Thesis/Datasets/adult/adult.test'

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
           'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
           'native_country', 'income']

df = pd.read_csv(train_data_file, header=None, names=columns, na_values=' ?')
test_df = pd.read_csv(test_data_file, header=None, names=columns, na_values=' ?')
df.drop('education', axis=1, inplace=True)  # Remove 'education' column
test_df.drop('education', axis=1, inplace=True)

cat_cols = ['workclass','marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income']

label_encoder = LabelEncoder()
for col in cat_cols:
   df[col] = label_encoder.fit_transform(df[col])
   test_df[col] = label_encoder.fit_transform(test_df[col])

X_train = df.drop('income', axis=1)
y_train = label_encoder.fit_transform(df['income'])

X_test = test_df.drop('income', axis=1)
y_test = label_encoder.fit_transform(test_df['income'])

rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)

spw= len(y_train[y_train == 0]) / len(y_train[y_train == 1])

fixed_params = {'objective':'binary',
               'boosting_type':'gbdt',
                'metric':'binary_logloss',
                  'n_jobs':-1,
                  'scale_pos_weight':spw}



# Define the parameter grid for hyperparameter tuning (without 'feature_fraction' and 'colsample_bytree')
param_grid = {
    'num_leaves': [31, 60, 90],
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.15, 0.05, 0.1]
}

# Define the LightGBM classifier
lgb_classifier = lgb.LGBMClassifier(**fixed_params,verbose=-1) #verbose to turn off messages
# Create a GridSearchCV object with early stopping callback
grid_search = GridSearchCV(
    estimator=lgb_classifier,
    param_grid=param_grid,
    # cv=rkf,  # Use RepeatedKFold cross-validation
    scoring='f1_weighted',
    n_jobs=-1  # Utilize all available CPU cores
    # fit_params={'callbacks': [early_stopping]}  # Add early stopping callback
)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best Model Parameters:", best_params)

# Make predictions on the test data
y_pred = best_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_rep)
