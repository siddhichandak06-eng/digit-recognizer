import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("digit_model.keras")

st.title("✍️ Handwritten Digit Recognizer")

st.write("Upload an image of a digit (0–9)")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Uploaded Image", width=200)

    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Prediction: {digit}")
    st.info(f"Confidence: {confidence*100:.2f}%")