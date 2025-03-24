# Implementing a Simple Neural Network from Scratch in Python

## Introduction 🎯📊🧠

In this blog post, we'll walk through the process of building a simple neural network from scratch using Python. Understanding the fundamentals of neural networks by implementing them from scratch is a great way to grasp key concepts like weight adjustments, activation functions, and gradient descent before moving on to more complex frameworks. We'll break down the implementation into key steps and use the `diabetes.csv` dataset as an example to train our network.

## Step 1: Importing Required Libraries 🖥️📚⚙️

We need `numpy` for mathematical operations and `pandas` for handling our dataset.

```python
import numpy as np
import pandas as pd
```

## Step 2: Loading the Dataset 📂📊🔍

We load `diabetes.csv`, extract input features (`X`), and define the target variable (`y`). The features (`X`) represent various health indicators such as glucose levels, blood pressure, and BMI, while the target variable (`y`) indicates whether the patient has diabetes (1) or not (0).

```python
data = pd.read_csv("diabetes.csv")
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values.reshape(-1, 1)  # Last column as target
```

## Step 3: Initializing Weights Randomly 🎲🧪🔢

The neural network consists of a single-layer with randomly initialized weights.

```python
np.random.seed(1)
synaptic_weights = 2 * np.random.random((X.shape[1], 1)) - 1
```

## Step 4: Defining the Activation Function 🔄📈✅

We use the sigmoid function, which helps in normalizing the outputs between 0 and 1. Additionally, it plays a crucial role in gradient-based learning by providing smooth gradients, ensuring stable weight updates during training.

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

## Step 5: Training the Neural Network 🏋️‍♂️🔬📉

We use gradient descent to adjust weights based on the error. This process involves computing the derivative of the loss function to determine the direction and magnitude of weight updates, gradually minimizing the error over multiple iterations.

```python
def train(X, y, weights, iterations=10000, learning_rate=0.1):
    for iteration in range(iterations):
        input_layer = X
        outputs = sigmoid(np.dot(input_layer, weights))
        
        error = y - outputs
        adjustments = error * (outputs * (1 - outputs))
        
        weights += learning_rate * np.dot(input_layer.T, adjustments)
    
    return weights

synaptic_weights = train(X, y, synaptic_weights)
```

## Step 6: Making Predictions 🔮🤖📊

Now that our model is trained, we can make predictions on new data.

```python
def predict(X, weights):
    return sigmoid(np.dot(X, weights))

predictions = predict(X, synaptic_weights)
```

## Full Code Example 📝💻🚀

```python
import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv("diabetes.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Initialize weights
np.random.seed(1)
synaptic_weights = 2 * np.random.random((X.shape[1], 1)) - 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(X, y, weights, iterations=10000, learning_rate=0.1):
    for iteration in range(iterations):
        input_layer = X
        outputs = sigmoid(np.dot(input_layer, weights))
        
        error = y - outputs
        adjustments = error * (outputs * (1 - outputs))
        
        weights += learning_rate * np.dot(input_layer.T, adjustments)
    
    return weights

synaptic_weights = train(X, y, synaptic_weights)

def predict(X, weights):
    return sigmoid(np.dot(X, weights))

predictions = predict(X, synaptic_weights)
```

## Conclusion 🎉📢🔍

We've successfully implemented a simple neural network from scratch using Python. This model learns from data through weight adjustments and can make predictions based on learned patterns. While this is a basic implementation, more complex models with multiple layers and optimizations can yield better results.

