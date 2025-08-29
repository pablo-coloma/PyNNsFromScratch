# PyNNsFromScratch
![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

In this project, a **Multi-Layer Perceptron (MLP)** is implemented from scratch in `Python` using only *NumPy*, without high-level frameworks like *TensorFlow* or *PyTorch*. The implementation has been tested on some classical datasets: Breast Cancer Wisconsin (BCW), California Housing (CH), Energy Efficiency (EE) and MNIST.

The development started with a **Shallow Neural Network (SNN)** class, which implements a network with a single hidden layer. Later, this was generalized into the **MLP** class, which supports an arbitrary number of hidden layers.

## 📂 Project Structure

```
PyNNsFromScratch
├── examples/         # Training scripts and demos
├── figures/          # Graphs and results from example scripts
├── src/mynns/        # Source code
│   ├── functions/    # Activations, outputs and losses
│   ├── snn.py        # Shallow NN Class
│   └── mlp.py        # Multi-layer Perceptron Class
└── requirements.txt  # Dependencies
```

## MLP Class

### 🧩 Features

**Supports different tasks**
- 📈 Regression
- 🏷️ Multiclass classification
- ✅ Multilabel classification

**Activation functions** 
- `sigmoid`
- `ReLU`
- `tanh`

**Output functions** (depending on the task)
- Regression → `identity`
- Multiclass classification → `softmax`
- Multilabel classification → `sigmoid`

**Losses** (depending on the task)
- Regression → `Mean Square Error (MSE)`
- Multiclass classification → `Cross Entropy (CE)`
- Multilabel classification → `Binary Cross Entropy (BCE)`

**Weights initialization**
- Xavier/Glorot uniform initialization.
- He/Kaiming normal initialization.

**Training**
- Backpropagation with `SGD` or `Adam`.
- `L2` regularization for the weights.

### 📚 Public methods

| Method         | Description                          |
|----------------|--------------------------------------|
| `fit`          | Train the model                      |
| `predict`      | Make predictions                     |
| `save_weights` | Export trained weights               |
| `plot_network` | Visualize the network architecture   |

## ⚙️ Installation (Python 3.13)

Clone this repository:

```
git clone https://github.com/pablo-coloma/PyNNsFromScratch.git
cd PyNNsFromScratch
```

Create a virtual environment:

```
python3.13 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

Install dependencies:

```
pip install -r requirements.txt
pip install -e .
```

## 🎯 Results on classic datasets

To reproduce the results the data must be downloaded from the following links and added to `examples/data/` directory:

- **MNIST:** 
https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

- **Breast Cancer Wisconsin:**
No download required, as it is included in `sklearn.datasets`.

- **California Housing:** 
https://storage.googleapis.com/tensorflow/tf-keras-datasets/california_housing.npz

- **Energy Efficiency:** 
https://archive.ics.uci.edu/ml/datasets/energy+efficiency

and execute the following files:

```
python examples/mnist_classification.py
python examples/bcw_classification.py
python examples/ch_regression.py
python examples/ee_regression.py
```

The first execution may take a few seconds while libraries are loaded.

---
### MNIST Classification

#### 🗂️ Dataset

The **MNIST** dataset is a benchmark in computer vision, consisting of 70,000 grayscale images of handwritten digits (0–9).
Each image is 28×28 pixels, and the dataset is divided into 60,000 training samples and 10,000 test samples. The objective is to predict which number is drawn in the image.

#### 📊 Results

Two models were trained on MNIST for comparison:

<div align="center">

| Model         | Accuracy on test  |
|:-------------:|:-----------------:|
| **My MLP**    | **97.7%**         |
| Sklearn MLP   | 97.9%             |

</div>

Both models achieve very similar accuracy, showing that the from-scratch implementation performs competitively against a standard library.

Below are examples of correct and incorrect predictions for each digit:

<p align="center">
    <img src="figures/MNIST%20Examples%20MyNet.png" alt="MNIST Example MyNet" width="800"/>
    <br>
    <em>Predictions with my MLP implementation</em>
</p>
<p align="center">
    <img src="figures/MNIST%20Examples%20SklearnNet.png" alt="MNIST Example SklearnNet" width="800"/>
    <br>
    <em>Predictions with sklearn MLP</em>
</p>

The architecture of my trained MLP can also be visualized:

<p align="center">
    <img src="figures/MNIST%20MyNetwork.png" alt="MNIST Network" width="600"/>
    <br>
    <em>Visualization of my trained MLP</em>
</p>

---
### Breast Cancer Wisconsin Classification

#### 🗂️ Dataset

The **Breast Cancer Wisconsin (Diagnostic)** dataset contains 569 samples with 30 numeric features extracted from digitized images of fine needle aspirates of breast masses.  
The target is binary: benign (0) vs malignant (1). It’s a classic small-to-medium dataset for testing binary classifiers.

#### 📊 Results

After training the model, it achieved an **accuracy of 97.2%** on the test.
The confusion matrix below summarizes the classification performance:

<p align="center">
    <img src="figures/BCW%20Confusion%20Matrix.png" alt="BCW Confusion Matrix" width="400"/>
    <br>
    <em>Confusion Matrix</em>
</p>

---
### California Housing Regression

#### 🗂️ Dataset

The **California Housing** dataset contains information from the 1990 U.S. Census, with 20,640 samples and 8 numerical features (e.g., median income, average rooms, population, etc.).  
The target is the median house value in a district, making it a regression task.

#### 📊 Results

After training, the model was able to predict house values with a **Mean Absolute Error of $36,000**.
The plots below illustrate the performance:

<p align="center">
    <img src="figures/CH%20Predictions%20and%20Residuals.png" alt="California Housing Predictions" width="600"/>
    <br>
    <em>Predictions and residuals</em>
</p>

*Interpretation:*  
- The left plot shows *predicted vs. true values*.
- The right plot shows *residuals (errors)*. Ideally, they should be centered around zero.  

---
### Energy Efficiency Regression

#### 🗂️ Dataset
The **Energy Efficiency** dataset contains 768 samples with 8 features (e.g., relative compactness, surface area, wall area, glazing area).  
The task is to predict heating and cooling loads of buildings, so it is framed as a regression problem with two target variables.

#### 📊 Results

After training, the model achieved accurate predictions for both targets, with a **Mean Absolute Error of 0.62 and 1.04** for heating and cooling loads respectively.  
The plots below illustrate the results:

<p align="center">
    <img src="figures/EE%20Predictions%20and%20Residuals.png" alt="Energy Efficiency Predictions" width="500"/>
    <br>
    <em>Predictions and residuals</em>
</p>

*Interpretation:*  
- The upper plots show *predicted vs. true values*.  
- The bottom plots show *residuals*, where points close to zero indicate better predictions.

## 📜 License

MIT License © 2025 Pablo Coloma Montolío