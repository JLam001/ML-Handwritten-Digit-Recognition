# James Lam
# Panther ID: 6194394
# Machine Learning (CAP 4612)
# I hearby certify that this program is my own work and is not the work of others.

# This program uses a convolutional neural network to recognize handwritten digits.

# Libraries: numpy, tensorflow, tkinter, PIL, cv2, os
import numpy as np
from tensorflow import keras
from tkinter import *
from PIL import Image, ImageDraw, ImageTk
import cv2 as cv
import os

# Check if model file exists
if os.path.exists('Handwritten_digit_mnist.h5'):
    # Load the saved model
    model = keras.models.load_model('Handwritten_digit_mnist.h5')
    print("Model: successfully loaded")
else:
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Define convolutional neural network model
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dense(10, activation = 'softmax')
    ])

    # Compile the model
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_split=0.1)

    print("Model: successfully trained")
    model.save('Handwritten_digit_mnist.h5')
    print("Model saved as: Handwritten_digit_mnist.h5")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_accuracy)

# GUI setup
root = Tk()
root.title("Handwritten Digit Recognition")

canvas_width = 300
canvas_height = 300

image = Image.new("L", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(image)

def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill = "black", width = 10)
    draw.ellipse((x1, y1, x2, y2), fill = "black", width = 10)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, canvas_width, canvas_height), fill = "white")

def predict_digit():
    img = image.resize((28, 28)).convert('L')
    img_array = np.array(img)
    
    # Add OpenCV functionality
    img_array = cv.GaussianBlur(img_array, (3, 3), 0)
    _, img_array = cv.threshold(img_array, 120, 255, cv.THRESH_BINARY_INV)
    img_array = cv.resize(img_array, (28, 28))
    img_array = img_array.reshape(1, 28, 28, 1)
    img_array = img_array.astype('float32') / 255.0
    
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    label.config(text = f"Predicted Digit: {predicted_digit}")

canvas = Canvas(root, width = canvas_width, height = canvas_height, bg = "white")
canvas.pack(side = TOP)

button_frame = Frame(root)
button_frame.pack(side = BOTTOM)

predict_button = Button(button_frame, text = "Predict Digit", command = predict_digit)
predict_button.pack(side = LEFT, padx=10)

clear_button = Button(button_frame,text = "Clear Canvas", command = clear_canvas)
clear_button.pack(side = LEFT, padx=10)

label = Label(button_frame, text = "")
label.pack(side = LEFT, padx=10)

canvas.bind("<B1-Motion>", paint)

root.mainloop()