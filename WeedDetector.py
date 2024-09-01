import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("Weed_Detection_Model.keras")

# Global variables to track cropping
x0, y0 = None, None
bbox = None
selected_image = None


# Function to preprocess and predict the selected region
def predict_region(bbox):
    global selected_image
    x1, y1, x2, y2 = bbox
    classes = {0: "crop", 1: "weed"}

    region = selected_image.crop((x1, y1, x2, y2))
    processed_image = region.resize((244, 244))
    processed_image = np.array(processed_image) / 255.0
    prediction = model.predict(np.expand_dims(processed_image, axis=0))

    predicted_class_index = np.argmax(prediction)
    predicted_class_name = classes[predicted_class_index]
    confidence_score = prediction[0][predicted_class_index]

    print("Predicted Class:", predicted_class_name)
    print("Confidence Score:", confidence_score)

    # Display the image with the rectangle
    plt.imshow(region)
    plt.axis('off')
    plt.show()


# Function to handle the upload button click event
def upload_image():
    global selected_image
    file_path = filedialog.askopenfilename()
    if file_path:
        selected_image = Image.open(file_path)
        # Resize the image if it's too large
        max_size = (600, 600)
        selected_image.thumbnail(max_size)
        photo = ImageTk.PhotoImage(selected_image)
        label.config(image=photo)
        label.image = photo
        label.file_path = file_path

        # Bind the mouse events for cropping
        label.bind("<ButtonPress-1>", start_crop)
        label.bind("<B1-Motion>", update_crop)
        label.bind("<ButtonRelease-1>", end_crop)


# Function to handle the start of cropping
def start_crop(event):
    global x0, y0
    x0, y0 = event.x, event.y


# Function to handle the update during cropping
def update_crop(event):
    global bbox
    draw = ImageDraw.Draw(selected_image.copy())
    bbox = (x0, y0, event.x, event.y)
    if bbox:
        draw.rectangle([x0, y0, event.x, event.y], outline="red")
        photo = ImageTk.PhotoImage(selected_image)
        label.config(image=photo)
        label.image = photo


# Function to handle the end of cropping
def end_crop(event):
    if bbox:
        predict_region(bbox)


# Create the main window
root = tk.Tk()
root.title("Weed Detection")

# Create widgets
label = tk.Label(root)
label.pack()

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack()

# Start the GUI event loop
root.mainloop()
