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
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt


# ## Import Data


X_train = pd.read_csv('data/train4.csv')
X_train = X_train.drop(columns = ['Id'])
y_train = X_train['SalePrice']
X_train = X_train.drop(columns=['SalePrice'])
X_test = pd.read_csv('data/test4.csv')
X_test = X_test.drop(columns = ['Id'])


# In[ ]:
# Model building
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


def root_mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Cast y_true to float32
    y_pred = tf.cast(y_pred, tf.float32)
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# Build the ANN model
model = Sequential([
    Dense(82, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dense(1)  # Output layer for regression
])


# Compile the model
model.compile(optimizer='adam', loss=root_mean_squared_error)

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=400, batch_size=32, validation_data=(X_val_scaled, y_val))

# Evaluate the model on validation data
loss = model.evaluate(X_val_scaled, y_val)
print(f"Mean Squared Error on validation data: {loss}")

# Visualize training history - loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('data/sample_plot2.png')
plt.show()

# In[28]:


predictions = model.predict(X_test_scaled)


# In[29]:


test_ids = pd.Series(range(1461, 1461 + len(predictions)))
result_df = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions.flatten()})  # Assuming predictions is a numpy array
result_df.to_csv('data/predicted_sale_prices.csv', index=False)

