import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Load libraries
import numpy as np

# Load train data
df_train = pd.read_csv('train.csv')

# Data inspection  
print(df_train.isnull().sum())
print(df_train.describe())

# Impute missing values
imputer = SimpleImputer(strategy='median')
df_train = pd.DataFrame(imputer.fit_transform(df_train), columns=df_train.columns) 

# Remove outliers
df_train = df_train[np.abs(df_train.SalePrice - df_train.SalePrice.mean()) <= (3*df_train.SalePrice.std())]

# Feature engineering  
y_train = df_train['SalePrice']
X_train = df_train.drop('SalePrice', axis=1)

# Normalize 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# PCA dimensionality reduction
pca = PCA(n_components=10)
X_train_reduced = pca.fit_transform(X_train_scaled)

# Load test data
df_test = pd.read_csv('test.csv') 

# Impute missing values
df_test = pd.DataFrame(imputer.transform(df_test), columns=df_test.columns)

# Normalize  
X_test_scaled = scaler.transform(X_test)

# PCA 
X_test_reduced = pca.transform(X_test_scaled)

# Train XGBoost model on train data 
xgboost = XGBRegressor()
xgboost.fit(X_train_reduced, y_train)

# Predict on test data  
y_pred = xgboost.predict(X_test_reduced)

# Evaluate 
mae = mean_absolute_error(y_test, y_pred)
print("MAE: ", mae)