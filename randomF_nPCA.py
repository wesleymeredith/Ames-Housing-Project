import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import random
import math

random.seed(0)

# Load the dataset
data = pd.read_csv('train5.csv')
test_data = pd.read_csv('test5.csv')  # Load test dataset

# Separate features and target
X = data.drop(['Id', 'SalePrice'], axis=1)
y = data['SalePrice']

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Initialize lists to store the results of each fold
rmses = []
r2_scores = []

# Perform 5-fold cross-validation
for train_index, test_index in kf.split(X):
    # Split data into training and test sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Create a Random Forest Regressor model
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=0, max_features='sqrt')

    # Fit the Random Forest model
    random_forest_model.fit(X_train, y_train)

    # Predict on test set using the original data
    y_preds = random_forest_model.predict(X_test)

    # Calculate and append the metrics
    mse = mean_squared_error(y_test, y_preds)
    r2 = r2_score(y_test, y_preds)
    rmses.append(math.sqrt(mse))
    r2_scores.append(r2)

# Calculate the average RMSE and average R-squared across all folds
avg_rmse = np.mean(rmses)
avg_r2 = np.mean(r2_scores)

print(f'Average Root Mean Squared Error: {avg_rmse}')
print(f'Average R-squared Score: {avg_r2}')

# Predict on the test2.csv dataset using the original data
X_test_data = test_data.drop(['Id'], axis=1)
test_predictions = random_forest_model.predict(X_test_data)

output_df = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})

output_df.to_csv('output.csv', index=False)  # Save the predictions
