import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit as st

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Load the model weights
try:
    model.load_weights('my_model_weights.h5')
except Exception as e:
    st.error(f"Error loading the model weights: {e}")

# Function to process the image and make a prediction
def predict(image):
    # Convert image to grayscale
    image = image.convert('L')
    # Resize image to 28x28
    image = image.resize((28, 28))
    # Convert image to array
    image_array = np.array(image)
    # Normalize the image
    image_array = image_array / 255.0
    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    # Add a channel dimension
    image_array = np.expand_dims(image_array, axis=-1)
    # Make prediction
    predictions = model.predict(image_array)
    return np.argmax(predictions, axis=1)[0]

# Streamlit app
st.markdown('#### Image Classification with TensorFlow')

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    label = predict(image)
    st.write(f'Prediction label number : {label}')
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    st.write(f'Prediction item name: {class_names[label]}')
