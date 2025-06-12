# 🧠 Model Distillation: Using CIFAR-10 dataset

## 🔍 What is Model Distillation?
Think of model distillation like **file compression**, but for AI models. Just like you can zip a big folder to save space, **we shrink a large neural network (called the “Teacher”) into a much smaller one (called the “Student”)** while trying to keep it smart.

This is especially useful when you want to run models on phones, edge devices, or any low-resource environments.

---

## 🚀 Project Overview

This project applies **knowledge distillation** to compress a trained deep learning model for the CIFAR-10 image classification task.

| Model               | Accuracy (%) | Size (MB) | Parameters | Inference Time (ms) |
|--------------------|--------------|-----------|------------|---------------------|
| 🧠 Teacher          | 83.13        | 17.81     | 4,664,970  | 111.29              |
| 👶 Student Baseline | 72.70        | 0.41      | 106,826    | 27.76               |
| 📘 Student Distilled | 70.86        | 0.41      | 106,826    | 25.82               |

> 📝 **Key takeaway**: The student model is **~98% smaller and ~4x faster**, but accuracy dropped slightly after distillation.

---

## 🎯 Goals
- Compress a high-performance model to a tiny one using knowledge distillation.
- Compare accuracy, size, and inference speed of:
  - Teacher model
  - Student trained normally
  - Student trained using distillation

---

## 🛠️ How It Works

1. **Dataset**: CIFAR-10 (10 categories of tiny images).
2. **Teacher Model**: A deep, accurate neural network trained for image classification.
3. **Student Model**: A shallow model with fewer parameters.
4. **Distillation**: We train the student on two things:
   - Real labels (hard targets)
   - Softened outputs from the teacher (soft targets) using **temperature scaling**.

---

## 📦 Results Summary

### 🎯 Accuracy
- Teacher: **83.13%**
- Student Baseline: **72.70%**
- Student after Distillation: **70.86%**

### ⚡ Speed & Size
- Size dropped from **17.81MB → 0.41MB**
- Inference time improved from **111ms → 26ms**

---

## 🔧 Technologies Used
- Python
- PyTorch
- TorchVision
- NumPy & Matplotlib

---

## 📚 Learn More
If you're new to model distillation, check out:
- [Hinton et al. (2015) - Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Distillation in PyTorch (Tutorial)](https://github.com/peterliht/knowledge-distillation-pytorch)

---

