# Anomaly_detection_MVTec-
MVTec dataset 
📊 Project Overview
This project implements an anomaly detection system for industrial inspection using a modified FastFlow architecture built on Wide ResNet-50 backbone. The model is trained on the MVTec Anomaly Detection dataset to identify defective products in manufacturing quality control.

Key Features
Transfer Learning: Utilizes pretrained Wide ResNet-50 weights from ImageNet

Binary Classification: Detects normal vs anomalous samples

Grad-CAM Compatible: Architecture supports visualization of model focus areas

Comprehensive Metrics: Tracks accuracy, AUC-ROC, and F1 score

📈 Model Performance
Metric	Value
Accuracy	91.69%
AUC-ROC	94.62%
F1 Score	80.18%
🏗️ Architecture
text
Input Image (256x256)
       ↓
   Wide ResNet-50
       ↓
AdaptiveAvgPool2d(1)
       ↓
      Flatten
       ↓
  Linear(2048, 1)
       ↓
  Binary Output
📁 Dataset Structure
The MVTec dataset is organized as:

text
mvtec_anomaly_detection/
├── category/
│   ├── train/
│   │   └── good/          # Normal training samples
│   └── test/
│       ├── good/          # Normal test samples
│       └── defect_type/   # Anomaly samples
└── ground_truth/          # Segmentation masks
🛠️ Requirements
bash
torch
torchvision
numpy
scikit-learn
matplotlib
seaborn
PIL
tqdm
