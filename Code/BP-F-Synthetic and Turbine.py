import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the data
#data = np.loadtxt('A1-synthetic.txt')
#X = data[:, :9]
#y = data[:, 9]

data = np.loadtxt('A1-turbine.txt')
X = data[:, :4]  # Select the first 4 columns as features
y = data[:, 4]   # Select the 5th column as the target variable


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a neural network model
model = tf.keras.Sequential([
    #tf.keras.layers.Dense(12, activation='relu', input_shape=(9,)),
    tf.keras.layers.Dense(12, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Plot the training errors
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Errors')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Make predictions on the test set
predictions = model.predict(X_test_scaled).flatten()

# Scatter plot for predictions vs targets in the test set
plt.scatter(y_test, predictions, alpha=0.5)
plt.title('Predictions vs Actual')
plt.xlabel('Actual Target Values')
plt.ylabel('Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r')  
plt.legend(['Perfect Predictions', 'Model Predictions'])
plt.show()
