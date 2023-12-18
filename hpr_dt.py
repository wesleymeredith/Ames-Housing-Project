import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np
import random
random.seed(0)

# Load the dataset
data = pd.read_csv('train4.csv')

# Separate features and target
X = data.drop(['Id', 'SalePrice'], axis=1)
y = data['SalePrice']

# Create a decision tree regressor model
model = DecisionTreeRegressor(min_samples_split=38, max_depth=12, random_state=0)

# Define a 5-fold cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Custom scorer for RMSE
def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse = make_scorer(rmse_scorer, greater_is_better=False)

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=kf, scoring=rmse)

# Calculate the average RMSE across all folds
average_rmse = np.abs(np.mean(cv_scores))

print(f'Average RMSE across 5 folds: {average_rmse}')
