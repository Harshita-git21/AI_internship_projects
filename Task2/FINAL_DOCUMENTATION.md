# Comprehensive Project Report: CIFAR-10 Image Classification with MobileNetV2

## 1. Project Title & Overview
This project implements a computer vision pipeline to classify images from the CIFAR-10 dataset into 10 distinct categories. It leverages Transfer Learning by fine-tuning a pre-trained deep neural network to achieve high accuracy efficiently.

## 2. Dataset Description
- **Source:** CIFAR-10 (automatically downloaded via TensorFlow/Keras).
- **Classes:** 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- **Format:** 32x32 color images.

## 3. Setup & Installation
### Prerequisites
- Python 3.8+
### Installation
Navigate to this directory and install the required dependencies:
```bash
pip install -r requirements.txt
```

## 4. Methodology & Pipeline
- **Data Preprocessing & Augmentation:** 
  - Applied random horizontal flips, rotations, and zooms to artificially expand the training dataset and prevent overfitting.
  - Resized inputs dynamically to 96x96 to better fit the MobileNetV2 architecture.
- **Model Architecture (Transfer Learning):** 
  - Loaded the pre-trained **MobileNetV2** (trained on ImageNet).
  - Froze the base model weights to retain its feature-extraction capabilities.
  - Added a custom classification head (GlobalAveragePooling, Dropout, and a Dense Softmax layer) optimized for our 10 classes.

## 5. Results & Evaluation
The model successfully learned to distinguish between the 10 classes. 
- **Training Curves:** Validation accuracy and loss were tracked across epochs to ensure the model was learning without overfitting.
*(Refer to `training_curves.png` in this folder for visual performance).*

## 6. Usage / How to Run
**To train the model from scratch:**
```bash
python train.py
```

**To run inference on a new image (e.g., the provided OIP.jpg):**
```bash
python inference.py --image_path OIP.jpg
```
