# Neuronal and Evolutionary Computation (MIA-MEISISI) Practical Exercise

Students Name: MD ARIF ULLAH BHUIYAN & JULIO CÉSAR SIGUEÑAS PACHECO

GitHub Link: [A1 Prediction with Back-Propagation and Linear Regression](https://github.com/arifbhuiyan82/A1-Prediction-with-Back-Propagation-and-Linear-Regression)

 Objective

Perform data classification with the following algorithms:

1. Neural Network with Back-Propagation (BP), custom Implementation.
2. Neural Network with Back-Propagation (BP-F), using free software.
3. Multiple Linear Regression (MLR-F), using free software.

 Algorithms Overview

# Neural Network with Back-Propagation (BP)

The Neural Network with Back-Propagation, often referred to as Back-Propagation (BP), is a supervised learning algorithm used for training Multi-Layer Perceptrons (MLPs). It involves both forward and backward passes, where inputs are passed through the network to generate an output, and errors are propagated back through the network to update the weights. It is widely used for pattern recognition, classification, and regression tasks.

# Neural Network with Back-Propagation with Momentum (BP-F)

This variant of the standard BP algorithm includes a momentum term in the weight update rule. The momentum term accelerates the learning process by considering the previous change in weight along with the current gradient. It helps prevent the network from getting stuck in local minima and can result in faster convergence.

# Multiple Linear Regression with Feature Selection (MLR-F)

Multiple Linear Regression (MLR) models the relationship between a dependent variable and multiple independent variables by fitting a linear equation to the data. The 'F' typically indicates the involvement of feature selection, which simplifies the model by selecting relevant features and removing irrelevant ones.

Datasets

This dataset contains detailed records of cybersecurity incidents and can be accessed from Kaggle. It includes over 40,000 records with 25 different metrics, providing a comprehensive view of various types of cybersecurity threats.

Data Preprocessing

Data normalization is performed using the MinMaxScaler to bring all data values to a consistent scale within the range [0, 1]. This process ensures stable training and avoids dominance of features based on magnitude.

 Implementation of BP-F

The implementation of BP-F includes:

- Initialization of random weights and biases.
- Forward pass for prediction.
- Loss calculation using mean squared error.
- Backward pass to update weights and biases.
- Training the neural network with multiple epochs.
- Making predictions using the trained model.

 Plots

The code includes two plots for visualization:

1. Training Losses Plot: This plot displays the training errors over epochs, helping visualize the learning progress.

2. Predictions vs. Actual Targets Plot: This scatter plot shows the predictions against actual target values in the test set, allowing you to assess the model's performance.

Feel free to explore the code and datasets in this repository to learn more about these algorithms and their applications.

