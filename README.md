# 🧠 Brain Tumor MRI Classification using PyTorch

A PyTorch implementation of a custom **MiniVGG** Convolutional Neural Network architecture, trained and evaluated on the **Brain Tumor MRI Dataset** from Kaggle for multi-class image classification.

---

## 📌 Overview

This project demonstrates how to build, train, and evaluate a lightweight MiniVGG-inspired CNN from scratch using PyTorch. It covers the full deep learning pipeline — from data loading and preprocessing to model training, evaluation, and prediction visualization.

---

## 🗂️ Dataset

- **Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)
- **Classes (4):** Glioma, Meningioma, No Tumor, Pituitary
- **Training samples:** 5,712
- **Test samples:** 1,311
- **Image size:** 224×224 RGB

---

## 🏗️ Model Architecture — MiniVGG

```
CustomVGG(
  (block_1): Sequential(
    Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
    BatchNorm2d(32)
    ReLU()
    Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    BatchNorm2d(32)
    ReLU()
    MaxPool2d(kernel_size=2)
  )
  (block_2): Sequential(
    Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    BatchNorm2d(64)
    ReLU()
    Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    BatchNorm2d(64)
    ReLU()
    MaxPool2d(kernel_size=2)
  )
  (block_3): Sequential(
    Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    BatchNorm2d(64)
    ReLU()
    Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    BatchNorm2d(64)
    ReLU()
    MaxPool2d(kernel_size=2)
  )
  (classifier): Sequential(
    Flatten()
    Dropout(p=0.5)
    LazyLinear(out_features=4)
  )
)
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| LR Scheduler | StepLR (step=3, gamma=0.1) |
| Loss Function | CrossEntropyLoss |
| Accuracy Metric | MulticlassAccuracy (torchmetrics) |
| Device | GPU (CUDA) / CPU |

---

## 📦 Requirements

```bash
pip install torch torchvision torchmetrics kagglehub matplotlib pandas
```

Or if running on Google Colab:

```bash
!pip install torchmetrics
```

> All other dependencies (PyTorch, torchvision, matplotlib, etc.) come pre-installed in Colab.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/SadeepM/brain-tumor-mri-classifier.git
   ```

2. Open the notebook in [Google Colab](https://colab.research.google.com/) or Jupyter:
   ```
   brain_tumor_mri_classifier.ipynb
   ```

3. Run all cells from top to bottom.

> **Note:** A Kaggle API key is required to download the dataset via `kagglehub`.

---

## 📊 What's Covered

- ✅ Downloading and exploring the Brain Tumor MRI dataset via KaggleHub
- ✅ Visualizing sample MRI images
- ✅ Data augmentation (random flip, rotation, normalization)
- ✅ Creating PyTorch `DataLoader` with mini-batches
- ✅ Building the MiniVGG model with `nn.Module`
- ✅ Batch normalization and dropout regularization
- ✅ Device-agnostic code (GPU/CPU)
- ✅ Training and testing loop functions
- ✅ Learning rate scheduling
- ✅ Tracking loss and accuracy per epoch
- ✅ Evaluating the model on test data
- ✅ Visualizing predictions vs ground truth labels

---

## 🛠️ Built With

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/)
- [Torchmetrics](https://torchmetrics.readthedocs.io/)
- [KaggleHub](https://github.com/Kaggle/kagglehub)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [Pillow](https://python-pillow.org/)
- Google Colab (T4 GPU)

---

## 📁 Project Structure

```
📦 repo
 ┗ 📓 brain_tumor_mri_classifier.ipynb
 ┗ 📄 README.md
```
