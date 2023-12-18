import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

# Load train data
# this modified training data 'train4.csv' has been pre-processed externally, original csv was 'train.csv'
df_train = pd.read_csv('train4.csv')

# Separate target variable 'SalePrice'
y_train = df_train['SalePrice']
X_train = df_train.drop('SalePrice', axis=1)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Leave out bc of the non-linearity
# # PCA dimensionality reduction
# pca = PCA(n_components=10)
# X_train_reduced = pca.fit_transform(X_train_scaled)

# XGBoost model
xgboost = XGBRegressor(random_state=0)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=0)
cross_val_results = cross_val_score(xgboost, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cross_val_results)

print(f'Cross-Validation RMSE Scores: {rmse_scores}')
print(f'Mean RMSE: {np.mean(rmse_scores)}')
mean_rmse = np.mean(rmse_scores)

# Now, you can use the trained model to make predictions on the test data
# Load the test data
df_test = pd.read_csv('test4.csv')

# Ensure same columns
print(set(X_train.columns) == set(df_test.columns))

# Normalize
X_test_scaled = scaler.transform(df_test)

# Leave out bc of non-linearity
# PCA dimensionality reduction
# X_test_reduced = pca.transform(X_test_scaled)

# Fit the model on the entire training data
xgboost.fit(X_train_scaled, y_train)

# Predict on the test data using the trained XGBoost model
y_test_pred = xgboost.predict(X_test_scaled)

# Create a DataFrame with predictions and test set indices or IDs
predictions_df = pd.DataFrame({'Id': df_test['Id'].astype('int32'), 'SalePrice': y_test_pred})

# Save predictions to a CSV file
predictions_df.to_csv('CV_kai_predictions.csv', index=False)

# Plotting the Cross-Validation Results with Line Plot
plt.figure(figsize=(10, 6))

# Line Plot
plt.plot(range(1, 6), rmse_scores, marker='o', label='Fold RMSE')

# Highlight mean RMSE with a red line and annotate
plt.axhline(y=mean_rmse, color='red', linestyle='-', label='Mean RMSE')
plt.text(3.5, mean_rmse * 1.01, f'Mean RMSE: {mean_rmse:.4f}', color='red')

plt.title('Cross-Validation RMSE Scores for Each Fold')
plt.xlabel('Fold')
plt.ylabel('RMSE Score')
plt.xticks(range(1, 6))  # Set x-axis ticks to whole numbers
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Define the hyperparameter grid to search
# param_grid = {
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 4, 5],
#     'min_child_weight': [1, 2, 3],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0],
#     'gamma': [0, 0.1, 0.2],
#     'reg_alpha': [0, 0.1, 0.2],
#     'reg_lambda': [0, 0.1, 0.2]
# }
# GridSearchCV
# grid_search = GridSearchCV(estimator=xgboost, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kf, verbose=1)
# grid_search.fit(X_train_scaled, y_train)
  
# # Best hyperparameters
# best_params = grid_search.best_params_
# print(f'Best Hyperparameters: {best_params}')
