import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with random values
        np.random.seed(42)
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.biases_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.biases_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, inputs):
        hidden_input = np.dot(inputs, self.weights_input_hidden) + self.biases_hidden
        hidden_activation = self.sigmoid(hidden_input)
        output_input = np.dot(hidden_activation, self.weights_hidden_output) + self.biases_output
        output_activation = self.sigmoid(output_input)
        return hidden_activation, output_activation

    def calculate_loss(self, predictions, targets):
        return np.mean(0.5 * (targets - predictions) ** 2)

    def backward_pass(self, inputs, targets, learning_rate):
        hidden_activation, output_activation = self.forward_pass(inputs)
        output_error = targets - output_activation
        output_delta = output_error * self.sigmoid_derivative(output_activation)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_activation)

        # Update weights and biases
        self.weights_hidden_output += np.dot(hidden_activation.T, output_delta) * learning_rate
        self.biases_output += np.sum(output_delta, axis=0) * learning_rate
        self.weights_input_hidden += np.dot(inputs.T, hidden_delta) * learning_rate
        self.biases_hidden += np.sum(hidden_delta, axis=0) * learning_rate

    def train(self, inputs, targets, learning_rate, epochs):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            _, output_activation = self.forward_pass(inputs)

            # Calculate loss
            loss = self.calculate_loss(output_activation, targets)
            losses.append(loss)

            # Backward pass
            self.backward_pass(inputs, targets, learning_rate)

            if epoch % 10 == 0:  # Print loss every 10 epochs
                print(f"Epoch {epoch}, Loss: {loss}")

        return losses, output_activation
    
    def predict(self, X):
        _, output_activation = self.forward_pass(X)
        return output_activation


# Load the data
inputs = np.loadtxt('A1-turbine.txt', skiprows=1, usecols=(0, 1, 2, 3))
targets = np.loadtxt('A1-turbine.txt', skiprows=1, usecols=4)

# Split into train and test sets
train_size = int(0.8 * len(inputs))
inputs_train, inputs_test = inputs[:train_size], inputs[train_size:]
targets_train, targets_test = targets[:train_size], targets[train_size:]

# Scale the inputs and targets
scaler_inputs = MinMaxScaler()
scaler_targets = MinMaxScaler()

# Fit and transform training data; transform test data
inputs_train_scaled = scaler_inputs.fit_transform(inputs_train)
targets_train_scaled = scaler_targets.fit_transform(targets_train.reshape(-1, 1))
inputs_test_scaled = scaler_inputs.transform(inputs_test)
targets_test_scaled = scaler_targets.transform(targets_test.reshape(-1, 1))

# Initialize and train the neural network with scaled inputs
neural_network = NeuralNetwork(input_size=4, hidden_size=12, output_size=1)
losses, _ = neural_network.train(inputs_train_scaled, targets_train_scaled, learning_rate=0.01, epochs=100)

# Predict using scaled test inputs and inverse transform predictions
predictions_scaled = neural_network.predict(inputs_test_scaled)
predictions = scaler_targets.inverse_transform(predictions_scaled).flatten()

# Plot the training losses
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot the predictions against actual targets
plt.scatter(targets_test, predictions, alpha=0.5)
plt.title('Predictions vs Actual')
plt.xlabel('Actual Target Values')
plt.ylabel('Predicted Values')
plt.plot([targets_test.min(), targets_test.max()], [targets_test.min(), targets_test.max()], 'r')  
plt.legend(['Perfect Predictions', 'Model Predictions'])
plt.show()
