import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


model = tf.keras.models.load_model('final.h5')

# Define the canvas size and stroke width
canvas_width = 300
canvas_height = 300
stroke_width = 10

# Define the drawing mode
drawing_mode = st.sidebar.selectbox("Drawing mode", ["freedraw", "transform"])

# Define a function to preprocess the image for the model


def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Define a function to get a prediction from the model


def predict_digit(image):
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    digit = np.argmax(prediction)
    return digit

# Create the Streamlit app


def main():
    st.title("Sketchboard")
    # Create a canvas for drawing
    canvas = st_canvas(
        fill_color="white",
        stroke_width=stroke_width,
        stroke_color="black",
        background_color="white",
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    # Get the image from the canvas
    if canvas.image_data is not None:
        image = Image.fromarray(
            canvas.image_data.astype('uint8'), 'RGBA').convert('L')
        st.image(image, width=150, caption="Drawn digit")
        digit = predict_digit(image)
        st.write(f"Predicted digit: {digit}")


if __name__ == "__main__":
    main()
