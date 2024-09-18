# MNIST Handwritten Digit Classification using Deep Learning (Neural Network)
This project demonstrates how to build a Neural Network (NN) model to classify handwritten digits from the MNIST dataset using TensorFlow and Keras. The project includes loading the dataset, training the model, evaluating its performance, and building a predictive system for recognizing handwritten digits.

## Table of Contents
- Introduction
- Dataset
- Installation
- Usage
- Model Architecture
- Evaluation
- Dependencies

## Introduction
The MNIST dataset contains 70,000 grayscale images of handwritten digits, ranging from 0 to 9. Each image is a 28x28 pixel square. The goal is to train a neural network that can recognize and classify these handwritten digits accurately.

## Dataset
The MNIST dataset is directly loaded from the keras.datasets library, which includes:

- 60,000 training images
- 10,000 testing images
Each image is labeled with the corresponding digit it represents.

## Installation
Clone this repository:

```
git clone https://github.com/yourusername/mnist-handwritten-digit-classification.git
cd mnist-handwritten-digit-classification
```
Install the necessary dependencies:

```
pip install -r requirements.txt
```
Open the Jupyter Notebook on Google Colab or any other environment to run the code.

## Usage
1- Load the Jupyter Notebook:

- If using Google Colab, upload the notebook and run all cells to train the model.
- Alternatively, run it locally on Jupyter Notebook.
2- Train the Neural Network using the MNIST dataset. The model will output the accuracy on the training and test data.

3- The notebook includes a Predictive System that allows users to input an image for digit recognition:

The user can provide a path to a handwritten digit image, and the system will predict the corresponding digit.
4- Use the trained model to predict handwritten digits by running the predictive system and entering the image path.

## Model Architecture
The Neural Network model consists of:

Input Layer: Flatten the 28x28 image into a 1D array of 784 pixels.
Hidden Layers:
50 neurons with ReLU activation.
Another layer with 50 neurons and ReLU activation.
Output Layer: 10 neurons with sigmoid activation (for digits 0-9).
The model uses Adam optimizer and sparse_categorical_crossentropy as the loss function, and is trained for 10 epochs.

## Evaluation
The model achieves the following accuracies:

Training Accuracy: 98.9%
Test Accuracy: 96.6%
The model's performance is also evaluated using a Confusion Matrix and visualized with a heatmap to understand its classification accuracy across different digits.

Predictive System
The notebook also contains a system for recognizing handwritten digits from external images. Simply provide the path to the image, and the system will:

Convert the image to grayscale.
Resize it to 28x28 pixels.
Scale the pixel values.
Predict the digit using the trained neural network.
## Dependencies
To run this project, you will need the following libraries:

Python 3.x
NumPy
Matplotlib
Seaborn
OpenCV (cv2)
TensorFlow
Keras
Install the dependencies using:

```
pip install -r requirements.txt
```

