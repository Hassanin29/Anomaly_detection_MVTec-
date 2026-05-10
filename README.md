# Anomaly_detection_MVTec-
MVTec dataset 
# MVTec Anomaly Detection with FastFlow

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📊 Project Overview

This project implements an **anomaly detection system** for industrial inspection using a modified **FastFlow** architecture built on **Wide ResNet-50** backbone. The model is trained on the **MVTec Anomaly Detection dataset** to identify defective products in manufacturing quality control with **91.69% accuracy**.

### Key Features

- 🔄 **Transfer Learning**: Utilizes pretrained Wide ResNet-50 weights from ImageNet
- 🎯 **Binary Classification**: Detects normal vs anomalous samples
- 🔥 **Grad-CAM Compatible**: Architecture supports visualization of model focus areas
- 📈 **Comprehensive Metrics**: Tracks accuracy, AUC-ROC, and F1 score
- 💾 **Model Export**: Automatic model saving and zipping for deployment

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 91.69% |
| **AUC-ROC** | 94.62% |
| **F1 Score** | 80.18% |

## 🏗️ Architecture
Input Image (256×256)
↓
Wide ResNet-50
(frozen first 10 layers)
↓
Layer4 Features
↓
AdaptiveAvgPool2d(1)
↓
Flatten
↓
Linear(2048, 1)
↓
Sigmoid Output
(0=Normal, 1=Anomaly)

## 📁 Dataset Structure

The MVTec dataset is organized as:
mvtec_anomaly_detection/
├── bottle/ # Category example
│ ├── train/
│ │ └── good/ # Normal training samples
│ ├── test/
│ │ ├── good/ # Normal test samples
│ │ ├── broken_large/ # Anomaly: large break
│ │ └── contamination/ # Anomaly: contamination
│ └── ground_truth/ # Pixel-level segmentation masks
│ ├── broken_large/
│ └── contamination/
└── cable/ # Another category
└── ...


## 🛠️ Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
Pillow>=8.0.0
tqdm>=4.62.0
