import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# load data
inputs = np.loadtxt('A1-synthetic.txt')[:, 0:9]
targets = np.loadtxt('A1-synthetic.txt')[:, 9]

#inputs = np.loadtxt('A1-turbine.txt', skiprows=1, usecols=(0, 1, 2, 3))
#targets = np.loadtxt('A1-turbine.txt', skiprows=1, usecols=4)

#data = np.loadtxt('winequality-white.csv', delimiter=';', skiprows=1)
#inputs = data[:, :-1]
#targets = data[:, -1]

# split into train and test sets
split_index = int(0.8 * len(inputs))  # More efficient to calculate once
inputs_train, inputs_test = inputs[:split_index], inputs[split_index:]
targets_train, targets_test = targets[:split_index], targets[split_index:]

# Create and train linear regression model
mlr = LinearRegression()
mlr.fit(inputs_train, targets_train.reshape(-1, 1))  

# Test the trained model
predictions = mlr.predict(inputs_test)

# Evaluate the model
mse = mean_squared_error(targets_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Plot the predictions against actual targets
plt.scatter(targets_test, predictions, alpha=0.5)
plt.title('Predictions vs Actual')
plt.xlabel('Actual Target Values')
plt.ylabel('Predicted Values')
plt.plot([targets_test.min(), targets_test.max()], [targets_test.min(), targets_test.max()], 'r')  
plt.legend(['Perfect Predictions', 'Model Predictions'])
plt.show()
