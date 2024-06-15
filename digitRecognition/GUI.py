import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from Prediction import predict

class DigitRecognizerApp:
    def __init__(self, root, Theta1, Theta2, Theta3):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas_width = 200
        self.canvas_height = 200
        self.white = 255  # Single integer value for grayscale

        # Title
        self.title_label = ttk.Label(root, text="Draw a Digit and Click Predict", font=('Helvetica', 16))
        self.title_label.pack(pady=10)

        # Canvas for drawing
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='white', bd=2, relief='solid')
        self.canvas.pack(pady=10)

        # Frame for buttons
        self.button_frame = ttk.Frame(root)
        self.button_frame.pack(pady=10)

        # Predict button
        self.predict_button = ttk.Button(self.button_frame, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=5)

        # Clear button
        self.clear_button = ttk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # Result label
        self.result_label = ttk.Label(root, text="", font=('Helvetica', 14))
        self.result_label.pack(pady=10)

        # Initialize the image and drawing context
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), self.white)
        self.draw = ImageDraw.Draw(self.image)

        # Bind the paint function to mouse movements
        self.canvas.bind("<B1-Motion>", self.paint)

        # Store the neural network parameters
        self.Theta1 = Theta1
        self.Theta2 = Theta2
        self.Theta3 = Theta3

    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill="white")
        self.result_label.config(text="")

    def preprocess_image(self):
        # Resize the image to 28x28 pixels
        image = self.image.resize((28, 28))
        image = ImageOps.invert(image)
        image = image.convert('L')
        image_data = np.array(image).reshape(1, 28 * 28)
        image_data = image_data / 255.0  # Normalize to [0, 1]
        return image_data

    def predict_digit(self):
        image_data = self.preprocess_image()
        prediction = predict(self.Theta1, self.Theta2, self.Theta3, image_data)
        self.result_label.config(text=f"Predicted Digit: {prediction[0]}")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

if __name__ == "__main__":
    Theta1 = np.loadtxt('model/digits/Theta1.txt')
    Theta2 = np.loadtxt('model/digits/Theta2.txt')
    Theta3 = np.loadtxt('model/digits/Theta3.txt')

    root = tk.Tk()
    
	# Set window size
    window_width = 350
    window_height = 400
    root.geometry(f"{window_width}x{window_height}")
    
    app = DigitRecognizerApp(root, Theta1, Theta2, Theta3)
    root.mainloop()
