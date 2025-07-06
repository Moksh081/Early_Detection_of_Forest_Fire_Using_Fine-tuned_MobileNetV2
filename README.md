# 🔥 Forest Fire Detection using Fine-Tuned MobileNetV2

> A lightweight deep learning model for real-time forest fire detection using RGB images.  
> 📌 Accepted for publication in SCOPUS-indexed Springer LNNS Proceedings (ISMS 2024-25).

---

## 🧠 Overview

Forest fires cause massive ecological and economic damage. This project presents a fine-tuned MobileNetV2 model trained to detect forest fires from RGB images. The model is optimized for real-time deployment on edge devices like drones, IoT sensors, and low-power embedded systems in remote areas.


![image](https://github.com/user-attachments/assets/0f627f2a-692c-4218-8705-8836e6e7e914)


---

## 📌 Key Features

- 🚀 **High Accuracy:** Achieved **93.08% training** and **87.02% validation accuracy**
- 🧠 **Transfer Learning:** Based on MobileNetV2 pre-trained on ImageNet
- ⚙️ **Optimized for Edge Devices:** Lightweight architecture suitable for real-time deployment
- 🧪 **Benchmark Dataset:** Trained on Wildfire dataset (2700+ labeled images)
- 📈 **Robust Training Pipeline:** Includes data augmentation and fine-tuning phases

---

## 🖼️ Sample Output

![image](https://github.com/user-attachments/assets/d9f2fa40-ca59-4a7f-b04e-70d03b3b9622)


---

## 🗃️ Dataset

**Name:** Wildfire Dataset  
**Size:** 2700+ RGB images  
**Sources:** Flickr, Unsplash, Public repositories  
**Classes:** Fire / No Fire

> The dataset includes both aerial and ground-level wildfire images with diverse backgrounds and forest types.

---

## 🏗️ Architecture


Input (224x224x3 RGB Image)
↓
Pretrained MobileNetV2 (ImageNet)
↓
GlobalAveragePooling2D
↓
Dense(1024, ReLU)
↓
Dense(2, Softmax)

![image](https://github.com/user-attachments/assets/53964d12-1d81-4e55-b5a6-062ec1777525)

