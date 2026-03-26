# 🚁 UAV Powerline Detection: A Hybrid CV & Deep Learning Approach

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?logo=opencv)

## 📌 Project Overview
Detecting thin, diagonal power lines against highly cluttered backgrounds (trees, buildings, urban environments) is a critical challenge for autonomous Unmanned Aerial Vehicles (UAVs). Standard object detection models (like YOLO) rely on bounding boxes, which are mathematically inefficient for thin wire structures.

This project implements a **Hybrid Pipeline** to solve this:
1. **Semantic Segmentation (Deep Learning):** A lightweight U-Net (MobileNetV2 backbone) extracts a pixel-perfect binary mask of the wires, filtering out background noise.
2. **Mathematical Extraction (Classical CV):** The mask is processed using morphological thinning and Probabilistic Hough Transforms to extract precise geometric vectors for flight controllers. Custom filtering logic merges fragmented lines and removes false positives (like straight tree branches).

## 📊 Dataset Acknowledgement
This project was trained and validated using the **PLD-UAV (Power Line Dataset for UAVs)**. 
Huge thanks to the original authors for providing pixel-level annotations for both mountain (PLDM) and urban (PLDU) environments.
* **Repository:** [SnorkerHeng/PLD-UAV](https://github.com/SnorkerHeng/PLD-UAV)

## 🏗️ Project Structure
```text
drone-wire-detection/
├── data/                   # (Not tracked in Git) PLD-UAV Dataset
├── src/                    
│   ├── dataset.py          # Custom PyTorch Dataset loader (dynamic pairing)
│   ├── train.py            # U-Net training loop and validation
│   ├── inference.py        # Single image testing with 4-panel dashboard
│   └── batch_inference.py  # Automated batch testing on random images
├── best_drone_wire_model.pth # (Not tracked) Trained model weights
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md