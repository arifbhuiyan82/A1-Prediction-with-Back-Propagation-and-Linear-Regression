import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

# Define the Custom Backpropagation Neural Network class
class CustomBPNeuralNetwork:
    def __init__(self, layers, learning_rate=0.001, momentum=0.01, num_epochs=500, fact='tanh', validation_percentage=0.2):
        self.layers = layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_epochs = num_epochs
        self.fact = fact
        self.validation_percentage = validation_percentage
        self.weights = []
        self.biases = []
        self.training_errors = []
        self.validation_errors = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layers[i + 1])))

    def activation(self, x):
        if self.fact == 'tanh':
            return np.tanh(x)
        elif self.fact == 'relu':
            return np.maximum(0, x)
        elif self.fact == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError(f"Unsupported activation function: {self.fact}")

    def activation_derivative(self, x):
        if self.fact == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.fact == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.fact == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        else:
            raise ValueError(f"Unsupported activation function: {self.fact}")

    def fit(self, X_train, y_train):
        split_index = int(len(X_train) * (1 - self.validation_percentage))
        X_train_split, X_val_split = X_train[:split_index], X_train[split_index:]
        y_train_split, y_val_split = y_train[:split_index], y_train[split_index:]

        prev_weight_updates = [np.zeros_like(w) for w in self.weights]
        prev_bias_updates = [np.zeros_like(b) for b in self.biases]

        for epoch in range(self.num_epochs):
            outputs, activations = self.forward(X_train_split)
            training_error = np.mean((outputs - y_train_split.reshape(-1, 1)) ** 2)
            self.training_errors.append(training_error)
            self.backward(X_train_split, y_train_split.reshape(-1, 1), activations, prev_weight_updates, prev_bias_updates)

            val_outputs, _ = self.forward(X_val_split)
            validation_error = np.mean((val_outputs - y_val_split.reshape(-1, 1)) ** 2)
            self.validation_errors.append(validation_error)

    def forward(self, X):
        activations = []
        input_to_layer = X

        for w, b in zip(self.weights, self.biases):
            z = np.dot(input_to_layer, w) + b
            activations.append(z)
            input_to_layer = self.activation(z)

        return input_to_layer, activations

    def backward(self, X, y, activations, prev_weight_updates, prev_bias_updates):
        deltas = []
        output = activations[-1]

        deltas.append((output - y) * self.activation_derivative(activations[-1]))

        for i in range(len(self.layers) - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.activation_derivative(activations[i - 1])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            weight_update = np.dot(activations[i - 1].T, deltas[i]) if i > 0 else np.dot(X.T, deltas[i])
            bias_update = np.sum(deltas[i], axis=0, keepdims=True)

            self.weights[i] -= self.learning_rate * weight_update + self.momentum * prev_weight_updates[i]
            self.biases[i] -= self.learning_rate * bias_update + self.momentum * prev_bias_updates[i]

            prev_weight_updates[i] = weight_update
            prev_bias_updates[i] = bias_update

    def predict(self, X):
        outputs, _ = self.forward(X)
        return outputs

# Load dataset
dataset_path = r"C:\Users\Arif Bhuiyan\Desktop\A1 Final"
dataset_name = "processed_cybersecurity_attacks.csv"
data = pd.read_csv(f"{dataset_path}\\{dataset_name}")

# Prepare data
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameters for tuning
hyperparameters = [
    {"layers": [128, 64], "epochs": 50, "learning_rate": 0.01, "momentum": 0.9, "activation_func": 'relu'},
    {"layers": [64, 32], "epochs": 100, "learning_rate": 0.001, "momentum": 0.8, "activation_func": 'tanh'},
    {"layers": [256, 128, 64], "epochs": 150, "learning_rate": 0.0005, "momentum": 0.9, "activation_func": 'relu'},
    {"layers": [128, 64, 32], "epochs": 75, "learning_rate": 0.005, "momentum": 0.85, "activation_func": 'tanh'},
    {"layers": [96, 48], "epochs": 120, "learning_rate": 0.01, "momentum": 0.85, "activation_func": 'sigmoid'},
    {"layers": [128, 64, 32, 16], "epochs": 200, "learning_rate": 0.001, "momentum": 0.9, "activation_func": 'relu'},
    {"layers": [256, 128, 64, 32], "epochs": 100, "learning_rate": 0.0001, "momentum": 0.95, "activation_func": 'relu'},
    {"layers": [64], "epochs": 50, "learning_rate": 0.01, "momentum": 0.9, "activation_func": 'tanh'},
    {"layers": [32, 16], "epochs": 300, "learning_rate": 0.005, "momentum": 0.8, "activation_func": 'sigmoid'},
    {"layers": [256, 128], "epochs": 125, "learning_rate": 0.002, "momentum": 0.88, "activation_func": 'relu'}
]

# Initialize a list to store the results
results = []

# Evaluate all hyperparameter combinations
for params in hyperparameters:
    # Build and train the model with the current hyperparameters
    nn = CustomBPNeuralNetwork(
        layers=[X_train.shape[1]] + params["layers"] + [1],
        learning_rate=params["learning_rate"],
        momentum=params["momentum"],
        num_epochs=params["epochs"],
        fact=params["activation_func"],
        validation_percentage=0.2
    )
    nn.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = nn.predict(X_test).flatten()
    
    # Calculate evaluation metrics
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    mae = np.mean(np.abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred) ** 2)
    
    # Store the results in a dictionary
    results.append({
        "Number of Layers": len(params["layers"]),
        "Layer Structure": str(params["layers"]),  # Correctly represent the layer structure
        "Epochs": params["epochs"],
        "Learning Rate": params["learning_rate"],
        "Momentum": params["momentum"],
        "Activation Function": params["activation_func"],
        "MAPE": mape,
        "MAE": mae,
        "MSE": mse
    })
    
    # Plot training and validation loss
    plt.plot(nn.training_errors, label="Training Loss")
    plt.plot(nn.validation_errors, label="Validation Loss")
    plt.title(f"Training and Validation Loss ({params['layers']})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Scatter plot of predictions vs actual values
    plt.scatter(y_test, y_pred, alpha=0.6, label="Predicted")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=0.5, label="Ideal Line")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predicted vs Actual ({params['layers']})")
    plt.legend()
    plt.show()

# Convert results to a DataFrame for analysis
results_df = pd.DataFrame(results)
print(results_df)

# Save results to a CSV file for reporting
results_df.to_csv("hyperparameter_results_BP.csv", index=False)
