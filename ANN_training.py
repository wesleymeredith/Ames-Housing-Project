#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers  # Importing the regularizers module
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt


# ## Import Data

# In[2]:


raw_data = pd.read_csv(f"data/train.csv")
raw_test_data = pd.read_csv(f"data/test.csv")


# In[3]:


raw_data


# In[4]:


raw_test_data


# # Preprocessing

# ## 1. Dealing with None data

# Since it's possible that the price is influenced by the None data, we won't simply drop the None data. Instead, we treat them as Nominal data by giving them a trivial value.

# In[72]:


threshold = len(raw_data) * 0.5  # 50% threshold
columns_to_drop = raw_data.columns[raw_data.isna().sum() > threshold]
columns_to_drop


# In[73]:
encoder = 'one-hot' # Choose from ['one-hot', 'label']
preprocessing = ['corr', 'pca'] # Choose from ['corr', 'pca']


# In[74]:


if encoder == 'one-hot':
    non_numeric_columns = raw_data.select_dtypes(exclude=['number']).columns.tolist()
    df = pd.get_dummies(raw_data,prefix=non_numeric_columns,columns=non_numeric_columns)
elif encoder == 'label':
    df = raw_data
    label_encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':  # Check if the column is categorical
            df[col].fillna('missing', inplace=True)  # Fill NaN with 'missing'
            df[col] = label_encoder.fit_transform(df[col])
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)


# In[75]:


cols_with_na = df_train.columns[df_train.isna().sum() > 0]
for col in cols_with_na: 
    print(f"There are {df_train[col].isna().sum()} NAN values in column {col}")


# So there are 217 data points without LotFrontage, 6 without MasVnrArea, and 64 without GarageYrBlt.
# 
# According to `data_description.txt`, GarageYrBlt = NAN means the house does not have a garage, so we replace them with 0. And we choose to drop the LotFrontage column because there are too many empty values, and we replace the empty MasVnrArea values with 0 too.

# In[76]:


df_train = df_train.drop(columns=['Id', 'LotFrontage'])
df_train = df_train.fillna(0)

df_test = df_test.drop(columns=['Id', 'LotFrontage'])
df_test = df_test.fillna(0)


# In[77]:


column_to_move = 'SalePrice'
columns = list(df_train.columns)
columns.insert(0, columns.pop(columns.index(column_to_move)))
df_train = df_train[columns]

df_test = df_test[columns]


# In[78]:


df_train.describe()


# In[79]:


df_test.describe()


# In[80]:


scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(df_train)
scaled_test_data = scaler.fit_transform(df_test)


# Now we use correlation coefficients to select some features.

# In[81]:


if 'corr' in preprocessing:
    scaled_df = pd.DataFrame(data=scaled_train_data, columns=df_train.columns)
    correlation_matrix = scaled_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.show()

    correlations = correlation_matrix['SalePrice'].abs().sort_values(ascending=False)
    threshold = 0.3
    relevant_features = correlations[correlations >= threshold].index.tolist()
    print(f"The features with the correlevant value no less than {threshold} are {relevant_features}.")
    df_train = df_train[relevant_features]
    df_test = df_test[relevant_features]


# Now select the attributes where the absolute value of their correlation coeficients with the SalesPrice is larger than .3.

# In[82]:


if 'pca' in preprocessing:

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_train)

    pca = PCA().fit(scaled_data)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_variance)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Explained Variance')
    plt.grid(True)
    plt.show()
for i in range(len(cumulative_variance)):
    if cumulative_variance[i] > 0.95:
        print(f"{i} components are selected.")
        break
cumulative_variance = cumulative_variance[:i]


# In[83]:


X_train.shape


# In[84]:


X_train = df_train.drop('SalePrice', axis=1)  # Independent variables
y_train = df_train['SalePrice']  # Target variable
X_test = df_test.drop('SalePrice', axis=1)  # Independent variables
y_test = df_test['SalePrice']  # Target variable

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Apply PCA
if 'pca' in preprocessing:
    num_components = len(cumulative_variance)  # Choose the number of principal components
    pca = PCA(n_components=num_components)
    pca_train_result = pca.fit_transform(X_train_scaled)
    pca_test_result = pca.transform(X_test_scaled)

    # Create a DataFrame with PCA results
    pca_train_df = pd.DataFrame(data=pca_train_result, columns=[f"PC{i+1}" for i in range(num_components)])
    pca_test_df = pd.DataFrame(data=pca_test_result, columns=[f"PC{i+1}" for i in range(num_components)])


    # Concatenate PCA components with the target variable
    pca_train_combined = pd.concat([pca_train_df, y_train.reset_index(drop=True)], axis=1)
    pca_test_combined = pd.concat([pca_test_df, y_test.reset_index(drop=True)], axis=1)
    
    X_train = pca_train_combined.iloc[:, :-1]  # Features (principal components)
    y_train = pca_train_combined.iloc[:, -1]   # Target variable ('SalePrice')
    X_test = pca_test_combined.iloc[:, :-1]  # Features (principal components)
    y_test = pca_test_combined.iloc[:, -1]   # Target variable ('SalePrice')


# # Model Building

# In[ ]:





# In[90]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def root_mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Cast y_true to float32
    y_pred = tf.cast(y_pred, tf.float32)
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# Build the ANN model
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     Dense(32, activation='relu'),
#     Dense(1)  # Output layer with one neuron for regression
# ])
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#     Dropout(0.1),  # Adding dropout for regularization
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
#     Dropout(0.1),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dense(1)  # Output layer for regression
])


# Compile the model
model.compile(optimizer='adam', loss=root_mean_squared_error)

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=300, batch_size=32, validation_data=(X_test_scaled, y_test))

# Evaluate the model on test data
loss = model.evaluate(X_test_scaled, y_test)
print(f"Mean Squared Error on test data: {loss}")

# Visualize training history - loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('sample_plot.png')
plt.show()


# In[34]:


predictions = model.predict(X_test_scaled)