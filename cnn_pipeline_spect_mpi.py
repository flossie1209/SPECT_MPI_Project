#!/usr/bin/env python3
"""
CNN Pipeline for SPECT MPI Image Classification

Project:
    CNN-based automated detection of myocardial ischemia from SPECT MPI polar maps.

Task:
    3-class classification:
        - normal
        - ischemia
        - infarction

Expected folder structure:

    project_folder/
    ├── cnn_pipeline_spect_mpi.py
    ├── data/
    │   ├── train/
    │   │   ├── normal/
    │   │   ├── ischemia/
    │   │   └── infarction/
    │   ├── val/
    │   │   ├── normal/
    │   │   ├── ischemia/
    │   │   └── infarction/
    │   └── test/
    │       ├── normal/
    │       ├── ischemia/
    │       └── infarction/
    └── results/

Run from terminal:
    python cnn_pipeline_spect_mpi.py

"""

import argparse
import os
from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)

from tqdm import tqdm


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # These can make results more deterministic but may slightly slow training.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Device selection
# -----------------------------

def get_device():
    """
    Select available device.

    Priority:
        1. CUDA GPU
        2. Apple Silicon MPS
        3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# -----------------------------
# Data loading
# -----------------------------

def build_transforms(img_size: int = 224):
    """
    Build image transformations.

    For transfer learning with ImageNet-pretrained ResNet18, we use ImageNet
    normalization statistics.

    Training uses mild augmentation.
    Validation/test do not use augmentation.
    """

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transforms, val_test_transforms


def build_dataloaders(data_dir: str, img_size: int, batch_size: int, num_workers: int):
    """
    Load train, validation, and test datasets using torchvision.datasets.ImageFolder.
    """

    data_dir = Path(data_dir)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    for required_dir in [train_dir, val_dir, test_dir]:
        if not required_dir.exists():
            raise FileNotFoundError(
                f"Missing folder: {required_dir}\n"
                "Expected data structure: data/train, data/val, data/test."
            )

    train_transforms, val_test_transforms = build_transforms(img_size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty. Check your data/train folders.")

    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Check your data/val folders.")

    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. Check your data/test folders.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# -----------------------------
# Model
# -----------------------------

def build_model(num_classes: int = 3, pretrained: bool = True):
    """
    Build a ResNet18 model for 3-class SPECT MPI classification.
    """

    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    # Replace final fully connected layer.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# -----------------------------
# Train and evaluate functions
# -----------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch."""

    model.train()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, desc="Evaluating"):
    """Evaluate model and return loss, accuracy, labels, predictions, and probabilities."""

    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)

            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)


# -----------------------------
# Plotting
# -----------------------------

def plot_training_curves(history: pd.DataFrame, results_dir: Path):
    """Save training/validation loss and accuracy curves."""

    # Loss curve
    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "training_validation_loss.png", dpi=300)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["train_acc"], label="Train accuracy")
    plt.plot(history["epoch"], history["val_acc"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "training_validation_accuracy.png", dpi=300)
    plt.close()


def plot_confusion_matrix(labels, preds, class_names, results_dir: Path):
    """Save confusion matrix plot."""

    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.title("Confusion Matrix: SPECT MPI CNN")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=300)
    plt.close()


# -----------------------------
# Save predictions
# -----------------------------

def save_predictions(
    test_dataset,
    labels,
    preds,
    probs,
    class_names,
    results_dir: Path,
):
    """
    Save test predictions to CSV.

    ImageFolder stores paths in test_dataset.samples:
        [(image_path, class_index), ...]
    """

    image_paths = [sample[0] for sample in test_dataset.samples]

    results = pd.DataFrame({
        "image_path": image_paths,
        "true_index": labels,
        "true_label": [class_names[i] for i in labels],
        "predicted_index": preds,
        "predicted_label": [class_names[i] for i in preds],
        "correct": labels == preds,
    })

    for i, class_name in enumerate(class_names):
        results[f"prob_{class_name}"] = probs[:, i]

    results.to_csv(results_dir / "test_predictions.csv", index=False)


def save_classification_report(labels, preds, class_names, results_dir: Path):
    """Save classification report as txt and csv."""

    report_text = classification_report(
        labels,
        preds,
        target_names=class_names,
        digits=4,
    )

    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(report_text)

    report_dict = classification_report(
        labels,
        preds,
        target_names=class_names,
        digits=4,
        output_dict=True,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(results_dir / "classification_report.csv")


def save_summary_metrics(labels, preds, results_dir: Path):
    """Save weighted and macro metrics."""

    acc = accuracy_score(labels, preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0,
    )

    summary = pd.DataFrame([{
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }])

    summary.to_csv(results_dir / "summary_metrics.csv", index=False)

    return summary


# -----------------------------
# Optional Grad-CAM
# -----------------------------

def unnormalize_image_tensor(img_tensor):
    """
    Convert normalized tensor back to RGB image in [0, 1].

    Input:
        img_tensor: C x H x W tensor
    Output:
        H x W x C numpy array
    """

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)

    return img


def generate_gradcam_examples(model, test_dataset, class_names, device, results_dir: Path, n_examples: int = 6):
    """
    Generate Grad-CAM examples.

    This function requires:
        pip install grad-cam opencv-python

    If packages are not installed, the function will skip Grad-CAM.
    """

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        warnings.warn(
            "Grad-CAM packages are not installed. Skipping Grad-CAM.\n"
            "To enable Grad-CAM, run: pip install grad-cam opencv-python"
        )
        return

    gradcam_dir = results_dir / "gradcam_examples"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # For ResNet18, the last convolutional block is layer4[-1].
    target_layers = [model.layer4[-1]]

    # Pick a few evenly spaced examples from the test set.
    n_examples = min(n_examples, len(test_dataset))
    indices = np.linspace(0, len(test_dataset) - 1, n_examples, dtype=int)

    cam = GradCAM(model=model, target_layers=target_layers)

    for idx in indices:
        img_tensor, true_label = test_dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_label = torch.argmax(outputs, dim=1).item()
            pred_prob = probs[0, pred_label].item()

        targets = [ClassifierOutputTarget(pred_label)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        rgb_img = unnormalize_image_tensor(img_tensor)
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        plt.figure(figsize=(6, 6))
        plt.imshow(cam_image)
        plt.axis("off")
        plt.title(
            f"True: {class_names[true_label]} | "
            f"Pred: {class_names[pred_label]} ({pred_prob:.2f})"
        )
        plt.tight_layout()
        plt.savefig(gradcam_dir / f"gradcam_test_index_{idx}.png", dpi=300)
        plt.close()

    print(f"Grad-CAM examples saved to: {gradcam_dir}")


# -----------------------------
# Main pipeline
# -----------------------------

def main(args):
    set_seed(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    print("\nLoading datasets...")
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    num_classes = len(class_names)

    print(f"Class names: {class_names}")
    print(f"Class to index: {class_to_idx}")
    print(f"Number of classes: {num_classes}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images:   {len(val_dataset)}")
    print(f"Test images:  {len(test_dataset)}")

    if num_classes != 3:
        warnings.warn(
            f"Expected 3 classes, but found {num_classes}. "
            "The script will still run, but check your folders."
        )

    # Save class mapping.
    pd.DataFrame(
        [{"class_name": k, "class_index": v} for k, v in class_to_idx.items()]
    ).to_csv(results_dir / "class_mapping.csv", index=False)

    print("\nBuilding model...")
    model = build_model(num_classes=num_classes, pretrained=not args.no_pretrained)
    model = model.to(device)

    from sklearn.utils.class_weight import compute_class_weight

    train_labels = [label for _, label in train_dataset.samples]

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_records = []
    best_val_acc = -np.inf
    best_model_path = results_dir / "best_resnet18_spect_mpi.pth"

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc, val_labels, val_preds, val_probs = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            desc="Validation",
        )

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")

        history_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to: {best_model_path}")

    history = pd.DataFrame(history_records)
    history.to_csv(results_dir / "training_history.csv", index=False)

    plot_training_curves(history, results_dir)

    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, test_labels, test_preds, test_probs = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        desc="Testing",
    )

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    print("\nClassification report:")
    print(classification_report(
        test_labels,
        test_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    ))

    save_predictions(
        test_dataset=test_dataset,
        labels=test_labels,
        preds=test_preds,
        probs=test_probs,
        class_names=class_names,
        results_dir=results_dir,
    )

    save_classification_report(
        labels=test_labels,
        preds=test_preds,
        class_names=class_names,
        results_dir=results_dir,
    )

    summary = save_summary_metrics(
        labels=test_labels,
        preds=test_preds,
        results_dir=results_dir,
    )

    plot_confusion_matrix(
        labels=test_labels,
        preds=test_preds,
        class_names=class_names,
        results_dir=results_dir,
    )

    print("\nSummary metrics:")
    print(summary)

    if args.gradcam:
        print("\nGenerating Grad-CAM examples...")
        generate_gradcam_examples(
            model=model,
            test_dataset=test_dataset,
            class_names=class_names,
            device=device,
            results_dir=results_dir,
            n_examples=args.gradcam_examples,
        )

    print("\nDone.")
    print(f"All outputs saved in: {results_dir.resolve()}")
    print("\nMain output files:")
    print(f"  - Best model: {best_model_path}")
    print(f"  - Training history: {results_dir / 'training_history.csv'}")
    print(f"  - Test predictions: {results_dir / 'test_predictions.csv'}")
    print(f"  - Classification report: {results_dir / 'classification_report.txt'}")
    print(f"  - Confusion matrix: {results_dir / 'confusion_matrix.png'}")
    print(f"  - Training curves: {results_dir / 'training_validation_loss.png'}")
    print(f"  - Training curves: {results_dir / 'training_validation_accuracy.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CNN pipeline for SPECT MPI 3-class image classification."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data folder containing train/val/test subfolders.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Folder where results will be saved.",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image resize size. Default: 224.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size. Default: 16.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs. Default: 10.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate. Default: 1e-4.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for Adam optimizer. Default: 1e-5.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers. Use 0 on many Mac setups.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use ResNet18 without ImageNet pretrained weights.",
    )

    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate Grad-CAM examples after test evaluation.",
    )

    parser.add_argument(
        "--gradcam-examples",
        type=int,
        default=6,
        help="Number of Grad-CAM test examples to save. Default: 6.",
    )

    args = parser.parse_args()
    main(args)
