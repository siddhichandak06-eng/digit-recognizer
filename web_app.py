import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Recreate model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Load weights
model.load_weights("model.weights.h5")

st.title("✍️ Handwritten Digit Recognizer")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Show original image
    original_img = Image.open(uploaded_file).convert("L")
    st.image(original_img, caption="Uploaded Image", width=200)

    # Convert to numpy
    img = np.array(original_img)

    # Invert colors (important)
    img = 255 - img

    # Resize to 28x28
    img = Image.fromarray(img).resize((28, 28))

    # Normalize
    img = np.array(img) / 255.0

    # Reshape
    img = img.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Prediction: {digit}")
    st.write(f"Confidence: {confidence*100:.2f}%")