#!/usr/bin/env python3
"""
Gradio app for the SPECT MPI ischemia classifier.

Loads the trained ResNet18 model (binary: Abnormal / Normal) and exposes
a simple upload -> prediction interface. Designed to run locally with
`python app.py` and to deploy directly on Hugging Face Spaces (Gradio SDK).

Expected repo layout (already matches your existing folder structure):

    SPECT_MPI_Project/
    ├── app.py                          <- this file
    ├── cnn_pipeline_spect_mpi.py       <- defines CustomCNN / build_model
    └── results_resnet18_epoch20/
        └── best_resnet18_spect_mpi.pth
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# -----------------------------
# Config
# -----------------------------

# Swap this to "results_resnet18/best_resnet18_spect_mpi.pth" if you'd
# rather ship the 10-epoch checkpoint instead of the 20-epoch one.
MODEL_PATH = "results_resnet18_epoch20/best_resnet18_spect_mpi.pth"

IMG_SIZE = 224

# Confirmed from class_mapping.csv: Abnormal=0, Normal=1
CLASS_NAMES = ["Abnormal", "Normal"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -----------------------------
# Device
# -----------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()


# -----------------------------
# Model
# -----------------------------

def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Rebuild the ResNet18 architecture used in training and load weights."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


model = load_model(MODEL_PATH, device)


# -----------------------------
# Preprocessing (matches val_test_transforms in cnn_pipeline_spect_mpi.py)
# -----------------------------

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# -----------------------------
# Inference
# -----------------------------

def predict(image: Image.Image):
    """
    Takes a PIL image (from Gradio's image input), returns a dict of
    {class_name: probability} that Gradio renders as a labeled bar chart.
    """
    if image is None:
        return None

    image = image.convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


# -----------------------------
# Interface
# -----------------------------

DESCRIPTION = """
Upload a myocardial perfusion SPECT (MPI) image to classify it as
**Normal** or **Abnormal** (indicating possible ischemia).

This is a research prototype trained on a public dataset of 192 patients
(BENG 280C, UC San Diego), achieving 87.5% test accuracy and 96% recall
on abnormal cases. It is not a diagnostic tool and has not been validated
for clinical use.
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="SPECT MPI image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="SPECT MPI Ischemia Classifier",
    description=DESCRIPTION,
    examples=None,  # add paths to a few sample images here if you want one-click demos
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
