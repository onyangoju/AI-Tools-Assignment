import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import os

# -----------------------------
# Define the CNN Model
# -----------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)  # 64 channels * 12 * 12 after conv+pool
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # Flatten except batch dim
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# -----------------------------
# Load model
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn_model.pt")

model = CNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Handwritten Digit Classifier (PyTorch)")
st.write("Draw a digit (0–9) in the box below and click **Predict**.")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img = ImageOps.grayscale(img)

    st.image(img, caption="Processed Input (28x28 grayscale)", width=150)

    if st.button("Predict"):
        transform = transforms.ToTensor()
        input_tensor = transform(img).unsqueeze(0)  # Shape: [1, 1, 28, 28]
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, 1).item()
            conf = torch.softmax(output, dim=1)[0, pred].item()
            st.success(f"Predicted Digit: **{pred}** with {conf*100:.2f}% confidence")
