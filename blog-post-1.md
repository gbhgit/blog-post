# Implementing a Simple Neural Network in Python

## 🎯 Introduction 

In this blog post, we'll walk through the process of building a simple neural network from scratch using Python. We'll break down the implementation into key steps and use the `diabetes.csv` dataset as an example to train our network.

### Neural Network Explanation

#### 🤔 What are Weights ? 
Weights are numerical values that determine the importance of input features in a neural network. They are adjusted during training to minimize error and improve the model’s predictions. Each connection between neurons has an associated weight that influences the network's learning process.

#### 🤔 What are Activation Functions? 
Activation functions introduce non-linearity into the model, allowing it to learn complex patterns. The most common activation functions include:

- **Sigmoid**: Outputs values between 0 and 1, useful for probability-based predictions.
- **ReLU**: Outputs zero for negative values and linear for positive values, commonly used in deep networks.
- **Tanh**: Outputs values between -1 and 1, often used in hidden layers.

#### Simple Neural Network Equation
#### 🤔 How is defined a Simple Neural Network Equation ?
For a simple neural network, the equation that relates the input, weights, bias, and output is **y = f (Wx + b)**.

- **x** = Input input matrix (e.g., features of the data)
- **W** = Weight matrix (learnable parameters)
- **b** = Bias vector (learnable parameters)
- **f** = Activation function (e.g., ReLU, Sigmoid, or Softmax)
- **y** = Output matrix

![Screenshot from 2025-03-25 08-36-09](https://github.com/user-attachments/assets/f6b9b972-a1a1-4f62-9e39-8ba3d5628692)

## 🎉 Fun Time
Now i will show the steps to create a Simple Neural Network, using a real database. **I hope you have fun !**

### Step 1: Importing Required Libraries

We need `numpy` for mathematical operations and `pandas` for handling our dataset.

```python
import numpy as np
import pandas as pd
```

### Step 2: Loading the Dataset 📊

We load `diabetes.csv`, extract input features (`X`), and define the target variable (`y`). The features (`X`) represent various health indicators such as glucose levels, blood pressure, and BMI, while the target variable (`y`) indicates whether the patient has diabetes (1) or not (0). Download csv here [diabetes.csv](https://raw.githubusercontent.com/gbhgit/blog-post/refs/heads/main/csv-files/diabetes.csv)

```python
data = pd.read_csv("diabetes.csv")
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values.reshape(-1, 1)  # Last column as target
```

### Step 3: Initializing Weights Randomly 🎲

The neural network consists of a single-layer with randomly initialized weights.

```python
np.random.seed(1)
synaptic_weights = 2 * np.random.random((X.shape[1], 1)) - 1
```

### Step 4: Defining the Activation Function 📈

We use the sigmoid function, which helps in normalizing the outputs between 0 and 1. Additionally, it plays a crucial role in gradient-based learning by providing smooth gradients, ensuring stable weight updates during training.

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

### Step 5: Training the Neural Network 📉

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

### Step 6: Making Predictions 🤖

Now that our model is trained, we can make predictions on new data.

```python
def predict(X, weights):
    return sigmoid(np.dot(X, weights))

predictions = predict(X, synaptic_weights)
```

### Full Code Example 🚀

```python
import numpy as np
import pandas as pd

def train_test_split(X, y, test_size=0.2, seed=None):
    if seed:
        np.random.seed(seed)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    test_size = int(len(indices) * test_size)
    train_indices, test_indices = indices[test_size:], indices[:test_size]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(X, y, weights, iterations=10000, learning_rate=0.1):
    for _ in range(iterations):
        input_layer = X
        outputs = sigmoid(np.dot(input_layer, weights))

        error = y - outputs
        adjustments = error * (outputs * (1 - outputs))

        weights += learning_rate * np.dot(input_layer.T, adjustments)

    return weights

def predict(X, weights):
    return sigmoid(np.dot(X, weights))

# Load dataset
data = pd.read_csv("https://raw.githubusercontent.com/gbhgit/blog-post/refs/heads/main/csv-files/diabetes.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Normalize the data
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=1)

# Initialize weights
np.random.seed(1)
synaptic_weights = 2 * np.random.random((X_train.shape[1], 1)) - 1

# Train the model
synaptic_weights = train(X_train, y_train, synaptic_weights)

# Make predictions on test data
y_pred = predict(X_test, synaptic_weights)
y_pred_labels = (y_pred > 0.5).astype(int)  # Convert probabilities to 0 or 1

# Calculate TP, TN, FP, FN
TP = np.sum((y_test == 1) & (y_pred_labels == 1))
TN = np.sum((y_test == 0) & (y_pred_labels == 0))
FP = np.sum((y_test == 0) & (y_pred_labels == 1))
FN = np.sum((y_test == 1) & (y_pred_labels == 0))

# Compute metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Show results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
```
