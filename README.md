# Deep Learning Experimentation with MNIST and notMNIST

## Overview
This project is designed to showcase deep learning techniques with a focus on Bayesian models, coreset visualization, and multi-headed architectures. By conducting experiments on MNIST and notMNIST datasets, it provides a practical understanding of how different models and approaches perform on image classification tasks.

## Project Structure
- `experiment_mnist.ipynb`: Jupyter notebook detailing experiments on the MNIST dataset.
- `experiment_notmnist.ipynb`: Jupyter notebook for experiments on the notMNIST dataset.
- `experiment_visualize_coresets.ipynb`: Notebook for visualization of coreset methods.
- `experiment_coresets.py`: Python script for coreset methods implementation.
- `BaysConv.py`: Implements the Bayesian Convolutional model.
- `BaysLinear.py`: Implements the Bayesian Linear model.
- `multihead_model_conv.py`: Python script for a multi-headed model using convolutional neural networks.
- `multihead_model.py`: Defines the multi-headed model architecture.
- `__init__.py`: Initializes the Python package for the project.

## Setup and Installation
Ensure you have Python 3.x installed along with the following packages:
- numpy
- matplotlib
- tensorflow (or tensorflow-gpu for GPU acceleration)
- keras

You can install these packages using pip:

```sh
pip install numpy matplotlib tensorflow keras

