# Lyrae

**Lyrae** is a lightweight, from-scratch machine learning and neural network library built purely in Python and NumPy. Designed for clarity, mathematical transparency, and execution in resource-constrained environments, it implements classical algorithms, custom optimization routines, and multi-layer neural architectures without relying on high-level deep learning frameworks.

---

## Features

### 1. Classical Machine Learning (`models/classical_models.py`)
* **Linear & Logistic Regression:** Gradient-descent-based linear models with configurable learning rates, epochs, and probability thresholding.
* **K-Nearest Neighbors (`KNN`):** Distance-based classification and regression using Euclidean distance and majority voting / mean aggregation.
* **Gaussian Naive Bayes (`GaussianNB` & `GaussianNB_predictor`):** Fast probabilistic classification with cached log-priors, variance smoothing ($\epsilon$), numerically stable log-sum-exp softmax probabilities, and discretized continuous target prediction.
* **Decision Trees & Random Forests (`DecisionTreeClassifier`, `RandomForest`):** CART-style binary splitting using Gini impurity, configurable depth/split constraints, feature subsampling, and ASCII tree visualization.

### 2. Neural Networks & Convolutions (`models/network.py`)
* **`Layer` & `NeuralNetwork`:** Modular dense layers with forward activation caching and recursive/iterative backpropagation, mini-batch gradient descent, gradient clipping, and automatic power-of-two hidden layer sizing.
* **`ConvolutionalLayer` & `CNN`:** 2D cross-correlation / convolution over image tensors with optional zero-padding, feeding directly into dense classification layers.

### 3. Activations, Initializers & Optimizers (`models/optimization.py`)
* **Activation Functions:** Vectorized `sigmoid`, inverse sigmoid (`asigmoid`), `relu`, and `softmax` with analytical derivatives (`deriv=True`).
* **Weight Initialization:** He / Xavier-style variance scaling (`he_initialization`) tailored to activation functions.
* **Optimizers:** Modular implementations of `Momentum`, `RMSprop`, and `Adam` for gradient-based parameter updates.

---

## Project Structure

```text
lyrae/
├── models/
│   ├── classical_models.py   # Linear/Logistic Regression, KNN, Naive Bayes, Trees & Forests
│   ├── network.py            # Dense NeuralNetwork, Layer, ConvolutionalLayer, and CNN
│   └── optimization.py       # Activations, derivatives, initializers, and optimizers
├── data/                     # Datasets (e.g., MNIST CSVs)
├── train.py                  # Training and evaluation entry point
└── README.md
```

---

## Requirements

* Python 3.10+
* `numpy`
* `polars` *(used in `train.py` for fast CSV loading)*
* `scikit-learn` *(optional, used in `train.py` for train/test splitting and scoring metrics)*

---

## Quick Start

### Training a Dense Neural Network on MNIST

```python
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from models import optimization as op
from models import network as ntwk

# Load and normalize MNIST data
mnist = pl.read_csv("data/mnist_train.csv", has_header=False).drop_nulls()
labels = mnist[:, 0].to_numpy().flatten()
images = mnist[:, 1:].to_numpy() / 255.0
y = np.eye(10)[labels]

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.15)

# Initialize and train network
nn = ntwk.NeuralNetwork(activation_func=op.sigmoid)
nn.hidden_layer_sizes = [264, 30]

nn.fit(
    X=X_train,
    y=y_train,
    epochs=50,
    learning_rate=0.01,
    batch_size=128,
)

predictions = nn.predict(X_test)
```

### Using Classical Models

```python
from models.classical_models import GaussianNB, RandomForest

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train_labels)
preds = gnb.predict(X_test)

# Random Forest
rf = RandomForest(n_trees=10, max_depth=8, min_sample_split=5)
rf.fit(X_train, y_train_labels)
rf_preds = rf.predict(X_test)
```

---

## Roadmap

* [ ] Complete end-to-end backpropagation through `ConvolutionalLayer` in `CNN`.
* [ ] Integrate `Momentum`, `RMSprop`, and `Adam` directly into `NeuralNetwork` weight updates.
* [ ] Refine `DecisionTreeClassifier` split evaluation to use weighted left/right Gini impurity.
* [ ] Add cross-entropy loss and multi-class accuracy reporting out of the box.