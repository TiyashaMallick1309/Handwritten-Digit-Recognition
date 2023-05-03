<<<<<<< HEAD
import streamlit as st
import cv2
import numpy as np
import keras
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

model = load_model('final.h5')


def preprocess_image(img):
    img = cv2.resize(255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (28, 28))
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255
    return img


def app():
    st.title("Digit Recognition")
    canvas = st_canvas(
        fill_color="white",
        stroke_width=20,
        stroke_color="black",
        background_color="white",
        width=200,
        height=200,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("Predict"):
        drawn_image = canvas.image_data
        preprocessed_image = preprocess_image(drawn_image)
        prediction = model.predict(preprocessed_image)
        plt.imshow(preprocessed_image.reshape(28, 28), cmap='gray')
        plt.show()
        print(prediction)
        digit_index = np.argmax(prediction)
        st.write("Predicted Digit:", digit_index)


if __name__ == "__main__":
    app()
=======
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Load the Keras model
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
>>>>>>> e04176bd1e9366d02a7265eedf411d89e8c4d58c
