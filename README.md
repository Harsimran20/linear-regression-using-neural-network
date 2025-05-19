Linear Regression with Neural Network
This project demonstrates a very simple deep learning model that learns a basic linear function:
[ y = 2x + 1 ]
The model uses TensorFlow/Keras to approximate this function using a single-layer neural network. It serves as an ideal introduction to deep learning and regression problems.

📌 Project Overview

Problem Type: Regression
Model Type: Fully Connected Neural Network (1 layer)
Framework: TensorFlow / Keras
Goal: Learn the mapping from x to y = 2x + 1 using data-driven training


🧠 Model Architecture
textInput (x) → Dense(1) → Output (y)
This is the simplest neural network structure, consisting of:

1 input neuron
1 dense layer with 1 output neuron
No activation function (default linear)

🛠️ Dependencies
Install the required package:
bashpip install tensorflow
🚀 Running the Project

Clone the Repository

git clone repository
cd linear-regression-using-neural-network
Run the Script

import tensorflow as tf
import numpy as np

# Data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y = 2 * x + 1

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=100, verbose=0)

# Predict
print(model.predict(np.array([[15.0]])))  # Expected output: ~31
📈 Result
The model learns to predict outputs like:
textInput: 15.0 → Predicted Output: 30.95 (approx)
This closely matches the true function ( y = 2x + 1 ), showing that the network has successfully learned the linear relationship.
📚 Educational Use
This project is ideal for:

First-time users of TensorFlow
Understanding the training process of neural networks
Observing how neural networks can learn mathematical functions
