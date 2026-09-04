---
title: SPECT MPI Ischemia Classifier
emoji: 🫀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---
# AI-Enabled Ischemia Detection from Myocardial Perfusion SPECT using Deep Learning

## Overview

This repository presents a deep learning framework for automated detection of myocardial ischemia from paired **stress** and **rest** myocardial perfusion SPECT (MPI) images. The project investigates the feasibility of using convolutional neural networks (CNNs) to classify ischemic and non-ischemic cases while providing visual interpretability through Grad-CAM.

Developed as part of a graduate project in **BENG 280C – Artificial Intelligence in Biomedical Imaging** at the University of California San Diego.

---

## Motivation

Coronary Artery Disease (CAD) remains one of the leading causes of mortality worldwide. Myocardial Perfusion Imaging (MPI) using SPECT is routinely used for diagnosing myocardial ischemia, but interpretation can be subjective and time-intensive.

This project explores whether deep learning models can:

* Improve diagnostic accuracy
* Learn discriminative imaging features automatically
* Provide explainable predictions using attention maps
* Support clinicians in computer-assisted diagnosis

---

## Dataset

The dataset consists of paired **stress** and **rest** myocardial perfusion SPECT studies.

### Image Processing

* Stress and rest images combined into a multi-channel input
* Image normalization
* Data augmentation
* Train/validation/test split
* Label encoding for ischemic vs. non-ischemic patients

---

## Methodology

### Model Architecture

* ResNet18 backbone
* Transfer learning with pretrained ImageNet weights
* Modified classification head
* Binary classification

### Training Pipeline

* PyTorch
* Binary Cross-Entropy Loss
* Adam Optimizer
* Learning Rate Scheduler
* Early Stopping

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

---

## Explainable AI

To improve model interpretability, **Gradient-weighted Class Activation Mapping (Grad-CAM)** is used to visualize image regions contributing most to the network's predictions.

These activation maps help verify that the network focuses on clinically meaningful myocardial regions rather than irrelevant background features.

---

## Repository Structure

```text
SPECT_MPI_Project/
│
├── data/                  # Dataset (not included)
├── notebooks/             # Jupyter notebooks
├── models/                # Saved model checkpoints
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── gradcam_examples/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Results

The proposed model achieved:

| Metric   | Value                          |
| -------- | ------------------------------ |
| Accuracy | **87.5%**                      |
| Task     | Binary Ischemia Classification |

The model successfully identified ischemic patients while maintaining robust generalization performance on unseen test data.

---

## Technologies

* Python
* PyTorch
* NumPy
* Pandas
* OpenCV
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

## Future Work

Potential extensions include:

* 3D CNN architectures
* Vision Transformers
* Multi-view SPECT analysis
* Integration of clinical metadata
* External validation on larger multi-center datasets
* Quantitative perfusion analysis
* Hybrid AI models combining imaging and electronic health records

---

http://localhost:8501/

## Citation

If you use this repository in your research, please cite:

```
Flossie Zhang, Sakshi Mohta.
AI-Enabled Ischemia Detection from Myocardial Perfusion SPECT using Deep Learning.
University of California San Diego.
2026.
```

---

## Author

## License

This repository is intended for research and educational purposes.

Please contact the author before using the code or dataset for commercial applications.

