# ============================================================
# Google Colab Grad-CAM for Trained ResNet18 SPECT MPI Model
# Upload .pth model + Upload SPECT MPI images
# ============================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from google.colab import files


IMG_SIZE = 224
CLASS_NAMES = ["Abnormal", "Normal"]   # change if your training order was ["Normal", "Abnormal"]
CAD_IDX = CLASS_NAMES.index("Abnormal")

RESULTS_DIR = Path("/content/cad_gradcam_outputs")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def build_model(num_classes=2):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


print("Upload your trained .pth model:")
uploaded_model = files.upload()
MODEL_PATH = list(uploaded_model.keys())[0]

model = build_model(num_classes=len(CLASS_NAMES)).to(device)

loaded = torch.load(MODEL_PATH, map_location=device)

if isinstance(loaded, dict):
    model.load_state_dict(loaded)
else:
    model = loaded.to(device)

model.eval()
print("Model loaded successfully.")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, img_tensor, target_class):
        x = img_tensor.unsqueeze(0).to(device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)

        pred_class = int(logits.argmax(dim=1).item())
        pred_prob = float(probs[0, pred_class].item())
        cad_prob = float(probs[0, target_class].item())

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=img_tensor.shape[1:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        cad_probability_map = np.clip(cam * cad_prob, 0, 1)

        return cam, cad_probability_map, pred_class, pred_prob, cad_prob

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img)
    original = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    return img_tensor, original


def save_gradcam_result(image_path, original, cam, cad_probability_map,
                        pred_class, pred_prob, cad_prob):

    cam_heatmap = plt.cm.jet(cam)[:, :, :3]
    prob_heatmap = plt.cm.jet(cad_probability_map)[:, :, :3]

    cam_overlay = np.clip(0.55 * original + 0.45 * cam_heatmap, 0, 1)
    prob_overlay = np.clip(0.55 * original + 0.45 * prob_heatmap, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(original)
    axes[0].set_title("Input SPECT MPI")
    axes[0].axis("off")

    axes[1].imshow(cam_heatmap)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(cam_overlay)
    axes[2].set_title("CAD Attention Map")
    axes[2].axis("off")

    axes[3].imshow(prob_overlay)
    axes[3].set_title(
        f"CAD Probability Map\n"
        f"CAD Prob: {cad_prob:.1%}\n"
        f"Pred: {CLASS_NAMES[pred_class]} ({pred_prob:.1%})"
    )
    axes[3].axis("off")

    plt.tight_layout()

    save_path = RESULTS_DIR / f"{Path(image_path).stem}_cad_gradcam.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()

    return save_path


print("Upload your SPECT MPI images:")
uploaded_images = files.upload()
image_paths = list(uploaded_images.keys())

print(f"Uploaded {len(image_paths)} images.")


gradcam = GradCAM(model, target_layer=model.layer4[-1])

for i, image_path in enumerate(image_paths):
    print(f"Processing {i+1}/{len(image_paths)}: {image_path}")

    img_tensor, original = load_image(image_path)

    cam, cad_probability_map, pred_class, pred_prob, cad_prob = gradcam.generate(
        img_tensor,
        target_class=CAD_IDX
    )

    save_path = save_gradcam_result(
        image_path=image_path,
        original=original,
        cam=cam,
        cad_probability_map=cad_probability_map,
        pred_class=pred_class,
        pred_prob=pred_prob,
        cad_prob=cad_prob
    )

    print("Saved:", save_path)

gradcam.remove_hooks()

!zip -r /content/cad_gradcam_outputs.zip /content/cad_gradcam_outputs
files.download("/content/cad_gradcam_outputs.zip")

print("Done.")

