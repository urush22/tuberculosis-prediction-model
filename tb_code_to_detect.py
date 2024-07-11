import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import cv2

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformations for input images
data_transforms = transforms.Compose([
    transforms.Resize(256),  # Resize to 256x256
    transforms.CenterCrop(224),  # Crop the center 224x224
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Load pretrained ResNet18 model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Replace the last fully connected layer
model.load_state_dict(torch.load('tuberculosis_prediction_model.pth', map_location=device))
model.eval()
model = model.to(device)


# Function to classify image
def classify_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image can't be loaded by the provided path")
        print(f'Path is {image_path}')
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_edited = data_transforms(Image.fromarray(image_rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_edited)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()

        if prediction == 1:
            text = "Tuberculosis detected"
            color = (0, 0, 255)  # Red color for detected
        elif prediction == 0:
            text = "Not suffering from tuberculosis"
            color = (0, 255, 0)  # Green color for not detected

        # Draw text on the image
        cv2.putText(image_rgb, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return Image.fromarray(image_rgb)


# Function to open an image using Tkinter file dialog and classify it
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        classified_image = classify_image(file_path)
        if classified_image is not None:
            # Convert classified image to Tkinter PhotoImage
            photo = ImageTk.PhotoImage(classified_image)
            label.config(image=photo)
            label.image = photo


# Create a Tkinter window
root = tk.Tk()
root.title("Tuberculosis Detection")

# Create a label widget to display the image
label = tk.Label(root)
label.pack(padx=10, pady=10)

# Create a button to open the file dialog
button = tk.Button(root, text="Open Image", command=open_image)
button.pack(pady=10)

# Run the Tkinter main loop
root.mainloop()
