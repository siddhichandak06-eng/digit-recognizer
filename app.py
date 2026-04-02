import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("digit_model.h5")

# Create window
root = tk.Tk()
root.title("Handwritten Digit Recognizer")
root.geometry("320x400")

# Canvas
canvas = tk.Canvas(root, width=280, height=280, bg='black')
canvas.pack(pady=10)

# PIL image to draw on
image = Image.new("L", (280, 280), "black")
draw = ImageDraw.Draw(image)

# Drawing function
def draw_lines(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='white')
    draw.ellipse([x-r, y-r, x+r, y+r], fill='white')

canvas.bind("<B1-Motion>", draw_lines)

# Preprocess image
def preprocess(img):
    img = img.resize((28, 28))
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

# Predict function
def predict_digit():
    img = preprocess(image)
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    label.config(text=f"Digit: {digit} ({confidence*100:.2f}%)")

# Clear canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0,0,280,280], fill="black")
    label.config(text="Draw a digit")

# Buttons
btn_predict = tk.Button(root, text="Predict", command=predict_digit)
btn_predict.pack(pady=5)

btn_clear = tk.Button(root, text="Clear", command=clear_canvas)
btn_clear.pack(pady=5)

# Label for result
label = tk.Label(root, text="Draw a digit", font=("Arial", 16))
label.pack(pady=10)

# Run app
root.mainloop()