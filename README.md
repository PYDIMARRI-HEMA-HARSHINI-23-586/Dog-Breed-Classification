# 🐶 Dog Breed Classification

Deep learning model that predicts dog breeds from images using transfer learning.

## 📘 Google Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/PYDIMARRI-HEMA-HARSHINI-23-586/ea2a39ff1670e14d800435c620b5f73a/untitled4.ipynb)


---

## 🐕 Project Description
A deep learning model that predicts **120 dog breeds** from images using transfer learning, data augmentation, and a custom inference pipeline. Built using the **Kaggle Dog Breed Identification** dataset.

---

## 📌 Overview
This project implements an end-to-end image classification system capable of identifying dog breeds from photos. It uses **InceptionV3** for feature extraction, trains a custom classifier head, applies strong augmentation, and supports inference on user-uploaded images with an optional **out-of-distribution (non-dog) detector**.

---

## 🧬 Models Used

### **1. InceptionV3 (Pretrained on ImageNet)**
- Used as the **feature extraction backbone**  
- All convolutional layers **frozen during training**  
- Excellent for fine-grained image recognition  
- Provides high-level feature embeddings for dog images  

### **2. Custom Classification Head**
Trained on top of InceptionV3:
- `Flatten()`  
- `Dense(256, ReLU)`  
- `BatchNormalization()`  
- `Dense(256, ReLU)`  
- `Dropout(0.3)`  
- `BatchNormalization()`  
- `Dense(120, Softmax)` *(predicts 120 breeds)*  

Maps extracted features → final breed prediction.

### **3. Out-of-Distribution Detector (Simple Thresholding)**
- Confidence threshold: **0.10**  
- If highest probability < 0.10 → **Not a Dog**  
- Prevents incorrect predictions on humans or unrelated objects  

---

## 📂 Dataset

**Source:** Kaggle — Dog Breed Identification  
Includes:
- 10,000+ labeled training images  
- 10,000+ test images  
- 120 dog breeds  
- Labels provided via `labels.csv`  

Dataset link:  
https://www.kaggle.com/competitions/dog-breed-identification

---

## 🧩 Training Pipeline

- Preprocessing & augmentation using **Albumentations**:
  - Horizontal & vertical flip  
  - Coarse dropout  
  - Gamma adjustment  
  - Brightness/contrast shift  
- Input size: **128 × 128**  
- Training with **EarlyStopping** + **ReduceLROnPlateau**  
- Batched + prefetched using `tf.data` for efficiency  
- Achieved **~0.99 AUC** on validation data  

---

## 🔍 Testing & Inference

You can test the model on:
- Kaggle test images  
- Any dog image from the internet  
- Your own photos (`.jpg` / `.jpeg`)  

### **Inference with Non-Dog Detection**

Behavior:

If top confidence < 0.10 → Not a Dog

Else → returns predicted breed + confidence score

---

## 📊 Results

- Validation AUC: ~0.99
- Strong generalization to real-world images
- Misclassifications occur mainly between visually similar breeds
(e.g., German Shepherd vs Belgian Malinois)

---

## 🛠️ Tech Stack

- TensorFlow / Keras
- InceptionV3 Transfer Learning
- Albumentations
- OpenCV
- NumPy
- Pandas
- Matplotlib

---

## 🚀 Future Improvements

- Upgrade backbone to EfficientNetV2 or ConvNeXt
- Increase input resolution to 224×224
- Add Grad-CAM visualizations
- Deploy using Streamlit or FastAPI
