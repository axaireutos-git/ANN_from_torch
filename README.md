# Neural Network for Dynamic System Prediction

This repository contains a Python project that uses a neural network to model and predict the behavior of a dynamic system. The system is modeled using a set of input-output data, and the neural network is trained to approximate the system's response. The project makes use of PyTorch for building and training the neural network, and the results are visualized using 3D plots.

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction
The goal of this project is to train a neural network to predict the response of a dynamic system given certain input conditions. The neural network is trained on a set of measured data points and then used to make predictions, which are visualized and compared with the actual system response.

## Data
The data for this project is stored in a CSV file named `data_NN.csv` and is read using pandas. It contains three features, `d1`, `d2`, and `d3`, which represent the state of the dynamic system.

### Data Preprocessing
- The input features `X_data` are constructed by concatenating an initial state `[-2, 0, -1]` with the rest of the data.
- The target values `y_data` are loaded directly from the CSV file.

## Model Architecture
The neural network consists of three layers, each followed by a `Tanhshrink` activation function. The architecture is defined as follows:
- Input Layer: Takes input of size 3 (number of features).
- Hidden Layers: Two hidden layers, each with 3 neurons and `Tanhshrink` activation.
- Output Layer: Produces an output of size 3 (number of target features).

The model is trained using the `L1Loss` function, and the `Adamax` optimizer is used to update the weights.

## Dependencies
- `numpy`: For numerical computations.
- `pandas`: For data loading and preprocessing.
- `torch`: For building and training the neural network.
- `matplotlib`: For plotting the results in 2D and 3D.

## Usage
1. Run the main script to train the neural network and visualize the results:
   ```bash
   python main.py
   ```

2. The training loop runs for 100,000 epochs, and the loss is recorded to monitor the performance.

3. Modify the hyperparameters, model architecture, or data as needed to experiment with different configurations.

## Results
- The training loss is plotted to show how the model improves over time.
- 3D plots are generated to compare the predicted system response with the actual data:
  - **Plot 1**: Compares the neural network's predictions with the true system response for the training data.
  - **Plot 2**: Shows how the model's prediction evolves over 200 iterations starting from an initial state.
