import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load train data
df_train = pd.read_csv('train.csv')

# Data inspection  
print(df_train.isnull().sum())
print(df_train.describe())

# Impute missing values
imputer = SimpleImputer(strategy='median')
df_train = pd.DataFrame(imputer.fit_transform(df_train), columns=df_train.columns) 

# Select only numerical columns
numeric_columns = df_train.select_dtypes(include=['float64', 'int64']).columns
numeric_df = df_train[numeric_columns]

# Remove outliers
numeric_df = numeric_df[np.abs(numeric_df.SalePrice - numeric_df.SalePrice.mean()) <= (3*numeric_df.SalePrice.std())]
y_train = numeric_df['SalePrice']

# Feature engineering  
X_train = numeric_df.drop('SalePrice', axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normalize 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA dimensionality reduction
pca = PCA(n_components=10)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)

# Load test data
df_test = pd.read_csv('test.csv') 

# Impute missing values for test data
df_test = pd.DataFrame(imputer.transform(df_test), columns=df_test.columns)

# Select only numerical columns for test data
numeric_columns_test = df_test.select_dtypes(include=['float64', 'int64']).columns
numeric_df_test = df_test[numeric_columns_test]

# Normalize test data
X_test_scaled = scaler.transform(numeric_df_test)

# PCA on test data
X_test_reduced = pca.transform(X_test_scaled)

# Train XGBoost model on train data 
xgboost = XGBRegressor()
xgboost.fit(X_train_reduced, y_train)

# Predict on test data  
y_pred = xgboost.predict(X_test_reduced)

# Evaluate 
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: ", rmse)
