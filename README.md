# Image Classification with TensorFlow

This repository contains a simple image classification web application built using TensorFlow and Streamlit. The application allows users to upload an image and get a prediction of the item type based on a pre-trained neural network model. The model is trained on the Fashion MNIST dataset, which includes a variety of clothing items.

## Features

- Image upload and display
- Preprocessing of the uploaded image
- Prediction of the clothing item from the image
- Display of the predicted item name

## Code Overview

### Model Architecture

The model used in this project is a simple neural network with the following architecture:

- Flatten layer to convert the 28x28 input image into a 1D array.
- Dense layer with 128 neurons and ReLU activation.
- Output Dense layer with 10 neurons (one for each class) and softmax activation.

### Image Processing and Prediction

The uploaded image is processed as follows:

1. Convert the image to grayscale.
2. Resize the image to 28x28 pixels.
3. Normalize the pixel values to the range [0, 1].
4. Add batch and channel dimensions to the image array.
5. Use the model to predict the class of the image.

### Streamlit Application

The Streamlit app allows users to upload an image and see the prediction. The app includes:

- An image uploader widget.
- Display of the uploaded image.
- Display of the predicted label and item name.

## Files

- `Fashion_Streamlit.py`: The main script for running the Streamlit app.
- `my_model_weights.h5`: The pre-trained model weights (ensure this file is present before running the app).

## Acknowledgements

This project uses the Fashion MNIST dataset from [TensorFlow datasets](https://www.tensorflow.org/datasets/catalog/fashion_mnist).

## Contact

For more information, visit my [LinkedIn profile](https://www.linkedin.com/in/dharmendra-behara-230388239/).
