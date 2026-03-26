# 🚁 UAV Powerline Detection: A Hybrid CV & Deep Learning Approach

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg?logo=opencv)

## 📌 Project Overview
Detecting thin, diagonal power lines against highly cluttered backgrounds (trees, buildings, urban environments) is a critical challenge for autonomous Unmanned Aerial Vehicles (UAVs). Standard object detection models (like YOLO) rely on bounding boxes, which are mathematically inefficient for thin, elongated wire structures.

This project implements a **Hybrid Pipeline** to solve this problem:
1. **Semantic Segmentation (Deep Learning):** A lightweight U-Net (MobileNetV2 backbone) extracts a pixel-perfect binary mask of the wires, actively filtering out background noise.
2. **Mathematical Extraction (Classical CV):** The predicted mask is processed using morphological thinning and Probabilistic Hough Transforms to extract precise geometric vectors.
3. **Custom Filtering Logic:** Deterministic algorithms merge fragmented line segments and remove angular false positives (like straight tree branches) to provide clean data for flight controllers.

## 📊 Dataset Acknowledgement
This project was trained and validated using the **PLD-UAV (Power Line Dataset for UAVs)**. 
Special thanks to the original researchers for providing the pixel-level annotations for both mountain (PLDM) and urban (PLDU) environments that made this segmentation approach possible.
* **Original Repository:** [SnorkerHeng/PLD-UAV](https://github.com/SnorkerHeng/PLD-UAV)

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Wik19/drone-wire-detection.git
   cd drone-wire-detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### 1. Training the Model
To train the U-Net model from scratch on the PLD dataset:
```bash
python src/train.py
```
This will process the data in `data/`, train the model using a MobileNetV2 backbone, and save the best weights to `best_drone_wire_model.pth`.

### 2. Batch Inference
To run batch inference on random test images from the PLD dataset:
```bash
python src/inference.py
```
This script loads the trained model, performs predictions on random images, applies the classical CV post-processing (skeletonization and Hough Lines), and displays the results side-by-side using Matplotlib.

### 3. Testing the Classical Line Extractor 
To test only the classical CV pipeline on known ground-truth masks (without using the neural network):
```bash
python src/line_extractor.py
```
This script acts as a test bench for the line extraction logic, visualizing the thinning, Hough transform, and line merging processes.

## 📁 Project Structure

* **`src/`**: Project source code.
  * `dataset.py`: PyTorch `PowerlineDataset` loader. Dynamically pairs augmented training images with their ground-truth `.png` masks.
  * `train.py`: Main U-Net training loop, including validation and model checkpointing.
  * `inference.py`: Full batch inference script running the neural network segmentation followed by the clean geometrical line extraction.
  * `line_extractor.py`: A sandbox script for the classical CV math behind turning binary segmentations into parameterized lines.
* **`data/`**: The PLD Dataset (Mountain and Urban subset folders go here).
* **`best_drone_wire_model.pth`**: Saved model weights (after training).
