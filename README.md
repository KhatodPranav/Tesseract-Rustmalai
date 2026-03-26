# 🛡️ CargoSight AI (SixRay Detection Project)
**Intelligent X-Ray Cargo Inspection & Border Security Terminal**

## 🔗 Project Resources
* **Project Files, ML Modal & Assets:** [Access the Google Drive Folder](https://drive.google.com/drive/folders/1NjCckFIbuuI7864bk9nRbynqcCgcodYT?usp=drive_link)

## 📖 Overview
CargoSight AI is a real-time, human-in-the-loop threat detection system designed for customs and border security. It assists human operators by instantly analyzing cargo X-ray images to detect concealed objects, calculate a cumulative risk score, and facilitate continuous model improvement through an active-learning feedback loop.

While pure "black-box" AI automation cannot be trusted with critical border security decisions, CargoSight acts as an **AI Co-Pilot**. It highlights potential threats and allows officers to flag AI mistakes (like dense material occlusions), instantly routing the data to retrain and improve the model.

---

## ✨ Key Features

* **Dual-Mode Data Ingestion:** Scan static digital X-ray files via drag-and-drop, or use the "Live Scanner" to capture frames directly from a camera feed.
* **Real-Time Threat Detection:** Instantly draws bounding boxes around prohibited items (Guns, Knives, Scissors, Pliers, Wrenches) with confidence scores.
* **Cumulative Risk Scoring Engine:** Applies a weighted algorithm (e.g., Gun = 1.0 multiplier, Wrench = 0.25 multiplier) to calculate a dynamic 0-100% Bag Risk Score.
* **Active Learning Feedback Loop:** Operators can flag inaccurate scans with specific constraints (Object Occlusion, Material Too Dense, Poor Bounding Box) to automatically sort images for future retraining.
* **High-Performance Dashboard:** A responsive, dark-mode, glassmorphism UI built with Tailwind CSS.

---

## 🔄 Project Workflow
Our system operates on a continuous, 3-step pipeline:
1. **Data Preprocessing:** Auto-orientation, grayscale conversion, auto-contrast equalization, and augmentation (flips, blur, noise) of X-ray imagery.
2. **Model Training:** Training the Convolutional Neural Network (CNN) on the augmented dataset to recognize high-risk silhouettes.
3. **Detection & Feedback:** Live inference via the Flask API, followed by human-in-the-loop validation to catch false positives/negatives.

---

## 📊 Dataset & Pre-processing
This project utilizes the **SIXray** dataset.

* **Kaggle Dataset:** [khanhbtq99/sixray](https://www.kaggle.com/datasets/khanhbtq99/sixray)
* **Roboflow Export:** [khanhs/sixray-yg9ii](https://universe.roboflow.com/khanhs/sixray-yg9ii) (License: CC BY 4.0)
* **Total Images:** 14,131 images annotated in YOLO format.

**Pre-processing & Augmentations Applied:**
To simulate degraded real-world sensor data and make the model more robust, the following techniques were applied:
* Grayscale conversion (simulating CRT phosphor displays)
* 50% probability of Horizontal / Vertical flips
* 90-degree rotations (Clockwise, Counter-Clockwise, Upside-down)
* Random Gaussian blur (0 to 1.7 pixels)
* Salt and pepper noise applied to 1% of pixels 

---

## 🛠️ Technology Stack

* **Machine Learning:** Python, Ultralytics YOLO11, CNN (Convolutional Neural Networks), PyTorch, OpenCV
* **Backend Engine:** Python, Flask
* **Frontend UI:** HTML5, Tailwind CSS, Vanilla JavaScript

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system. 

### 2. Installation
Open your terminal, navigate to the project folder, and install the required Python dependencies:

```bash
pip install flask ultralytics Pillow
