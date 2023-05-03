import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

model = load_model('final.h5')

canvas_width = 300
canvas_height = 300
stroke_width = 10

drawing_mode = st.sidebar.selectbox("Drawing mode", ["freedraw", "transform"])


def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_digit(image):
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    digit = np.argmax(prediction)
    return digit


def main():
    st.title("Sketchboard")
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
