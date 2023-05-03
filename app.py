import streamlit as st
import cv2
import numpy as np
import keras
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

model = load_model('.\best_model.h5')


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
