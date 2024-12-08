import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Input, Dense
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the dataset
data_path = r"C:\Users\Arif Bhuiyan\Desktop\A1 Final\processed_cybersecurity_attacks.csv"
data = pd.read_csv(data_path)

# Separate features and target
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to build and train a neural network model
def build_nn_model(layers, epochs, learning_rate, momentum, activation_func):
    # Define the model
    model = Sequential()
    model.add(Input(shape=(X_train_scaled.shape[1],)))  # Input layer
    for units in layers:
        model.add(Dense(units, activation=activation_func))  # Hidden layers
    model.add(Dense(1))  # Output layer

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum), 
                  loss='mse', 
                  metrics=[tf.keras.metrics.MeanAbsoluteError(), 
                           tf.keras.metrics.MeanSquaredError(), 
                           tf.keras.metrics.MeanAbsolutePercentageError()])
    
    # Train the model
    history = model.fit(X_train_scaled, y_train, 
                        validation_data=(X_test_scaled, y_test), 
                        epochs=epochs, 
                        batch_size=32, 
                        verbose=0)  # Turn off verbose output for cleaner execution
    return model, history

# Define hyperparameters for tuning
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
    model, history = build_nn_model(
        layers=params["layers"],
        epochs=params["epochs"],
        learning_rate=params["learning_rate"],
        momentum=params["momentum"],
        activation_func=params["activation_func"]
    )
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Extract evaluation metrics from the training history
    mape = history.history['val_mean_absolute_percentage_error'][-1]
    mae = history.history['val_mean_absolute_error'][-1]
    mse = history.history['val_mean_squared_error'][-1]
    
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
    plt.plot(history.history['loss'], label="Training Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
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
results_df.to_csv("hyperparameter_results_MLR-F.csv", index=False)

