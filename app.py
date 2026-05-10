import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image


import os

print(os.getcwd())

# Load trained model
model = load_model("digit_model.keras")

st.title("Handwritten Digit Recognition")

st.write("Draw a digit below:")

# Create drawing canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("Predict"):

    if canvas_result.image_data is not None:

        # Get image from canvas
        img = canvas_result.image_data

        # Convert to uint8
        img = (img[:, :, 0]).astype(np.uint8)

        # Resize to 28x28
        img = cv2.resize(img, (28, 28))

        # Normalize
        img = img / 255.0

        # Reshape for CNN
        img = img.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img)

        digit = np.argmax(prediction)

        confidence = np.max(prediction)

        st.subheader(f"Predicted Digit: {digit}")

        st.write(f"Confidence: {confidence:.4f}")

