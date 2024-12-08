import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data_path = r'C:\Users\Arif Bhuiyan\Desktop\A1 Final\processed_cybersecurity_attacks.csv'
df = pd.read_csv(data_path)

# Set 'Anomaly Scores' as the target column
X = df.drop(columns=['Anomaly Scores'])
y = df['Anomaly Scores']

# Data Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Function to build and train the model
def build_and_train_model(layers_config, learning_rate, epochs, activation_fn):
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))  # Input layer
    for units in layers_config:  # Hidden layers
        model.add(layers.Dense(units, activation=activation_fn))
    model.add(layers.Dense(1, activation='linear'))  # Output layer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_split=0.2)
    y_pred = model.predict(X_test, verbose=0)
    if np.isnan(y_pred).any():
        raise ValueError("Model predictions resulted in NaN. Check your data and training process.")
    if np.isnan(y_test).any():
        raise ValueError("Target values contain NaN.")
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return mape, mae, mse, history, y_pred

# Hyperparameter combinations to test
hyperparameter_combinations = [
    {"layers": [128, 64], "epochs": 50, "learning_rate": 0.01, "activation_fn": 'relu'},
    {"layers": [64, 32], "epochs": 100, "learning_rate": 0.001, "activation_fn": 'tanh'},
    {"layers": [256, 128, 64], "epochs": 150, "learning_rate": 0.0005, "activation_fn": 'relu'},
    {"layers": [128, 64, 32], "epochs": 75, "learning_rate": 0.005, "activation_fn": 'tanh'},
    {"layers": [96, 48], "epochs": 120, "learning_rate": 0.01, "activation_fn": 'sigmoid'},
    {"layers": [128, 64, 32, 16], "epochs": 200, "learning_rate": 0.001, "activation_fn": 'relu'},
    {"layers": [256, 128, 64, 32], "epochs": 100, "learning_rate": 0.0001, "activation_fn": 'relu'},
    {"layers": [64], "epochs": 50, "learning_rate": 0.01, "activation_fn": 'tanh'},
    {"layers": [32, 16], "epochs": 300, "learning_rate": 0.005, "activation_fn": 'sigmoid'},
    {"layers": [256, 128], "epochs": 125, "learning_rate": 0.002, "activation_fn": 'relu'}
]

# Store results in a list
results = []

# Loop through all combinations and train models
for config in hyperparameter_combinations:
    print(f"Running configuration: {config}")
    mape, mae, mse, history, y_pred = build_and_train_model(
        config['layers'], config['learning_rate'], config['epochs'], config['activation_fn']
    )
    
    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label="Training Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.title(f"Training and Validation Loss for {config['layers']} Layers")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Scatter plot of predictions vs actual values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, label="Predicted")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=2, label="Ideal Line")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predicted vs Actual for {config['layers']} Layers")
    plt.legend()
    plt.show()

    # Store results in the list
    results.append({
        'Number of Layers': len(config['layers']),
        'Layer Structure': config['layers'],
        'Epochs': config['epochs'],
        'Learning Rate': config['learning_rate'],
        'Activation Function': config['activation_fn'],
        'MAPE': mape,
        'MAE': mae,
        'MSE': mse
    })

# Convert results into a DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_path = r'C:\Users\Arif Bhuiyan\Desktop\A1 Final\hyperparameter_results_BP-F.csv'
results_df.to_csv(results_path, index=False)


