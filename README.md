# 🧠 Brain Tumor MRI Classification & Explainable AI (XAI) System

A medical-AI application for brain tumor classification (Glioma, Meningioma, No Tumor, Pituitary) with multi-modal visual explainability using **Grad-CAM** and **LIME**. 

Designed with a strict architectural decoupling between the **GUI layer** and the **core ML/XAI engine**, allowing seamless deployment on laptops and straightforward future migration to a **Raspberry Pi-based handheld diagnostic assistant**.

---

## 📁 Project Structure

```
BrainTumor_XAI/
├── app.py                          # Streamlit Laptop GUI (Medical-AI Theme)
├── model/
│   └── brain_tumor_model.keras     # Place your trained Kaggle model here
├── inference/
│   ├── __init__.py
│   └── predictor.py                # Standalone prediction & probability engine
├── xai/
│   ├── __init__.py
│   ├── gradcam.py                  # Grad-CAM attention heatmap & overlay generator
│   ├── lime_explainer.py           # LIME superpixel perturbation explainer
│   └── superpixel.py               # Superpixel weight ranking bar chart
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py            # Image loaders & model normalizers
│   └── generate_samples.py         # Synthetic brain MRI test sample generator
├── sample_data/                    # Sample MRI scans for instant testing
│   ├── sample_glioma.png
│   ├── sample_meningioma.png
│   ├── sample_notumor.png
│   └── sample_pituitary.png
├── requirements.txt                # Python dependencies
├── run_app.bat                     # 1-Click Windows launcher
└── README.md                       # Documentation & Raspberry Pi guide
```

---

## 🎯 Target Classes

The classifier distinguishes between 4 diagnostic categories:
1. **Glioma** (Class index `0`)
2. **Meningioma** (Class index `1`)
3. **No Tumor** (Class index `2`)
4. **Pituitary** (Class index `3`)

---

## 🚀 Quickstart Guide (Laptop Demonstration)

### 1. Install Dependencies
Ensure you have Python 3.10 installed:
```bash
pip install -r requirements.txt
```

### 2. Add Your Trained Model Checkpoint
Copy your trained model (`.keras` or `.h5` format, e.g. `efficientnetb0_p3.keras` / `efficientnet_final.keras`) into the `model/` folder:
```bash
model/brain_tumor_model.keras
```
*(The system automatically discovers `.keras` / `.h5` files in `model/` or workspace root).*

### 3. Launch the Application
Run the Streamlit application:
```bash
streamlit run app.py
```
Or double-click `run_app.bat` on Windows.

---

## 🔍 Explainable AI (XAI) Methods

### 1. Grad-CAM (Visual Attention)
* Computes gradients with respect to the final convolutional feature maps (`top_conv` for EfficientNet, `conv5_block3_out` for ResNet, `block5_conv3` for VGG16).
* Identifies visual regions that produced the strongest class activations.
* Generates a high-contrast OpenCV `COLORMAP_JET` overlay blended at configurable opacity ($\alpha = 0.5$).

### 2. LIME (Superpixel Perturbations)
* Segments the MRI using the **Quickshift** algorithm (`kernel_size=4`, `max_dist=200`, `ratio=0.2`).
* Generates local linear explanations over perturbed superpixels:
  * **Positive Mask (Green)**: Superpixels supporting the predicted diagnosis.
  * **Positive + Negative Mask (Green/Red)**: Contrasts supporting vs. contradicting regions.
  * **Superpixel Weight Field**: Continuous diverging heatmap of regional feature importance.

### 3. Quantitative Superpixel Ranking
* Visualizes the numerical contribution weight ($w$) for the top 12 most influential superpixels in a comparative horizontal bar chart.

---

## 🍓 Future Raspberry Pi Deployment Guide

The core inference and XAI components (`inference/`, `xai/`, `utils/`) are **completely independent of Streamlit**. They accept raw NumPy image arrays (e.g. directly from a Raspberry Pi camera stream).

### Raspberry Pi Architecture:
```
           Raspberry Pi 4 / 5
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
  Pi Camera v2/v3         Image File
       │                       │
       └───────────┬───────────┘
                   ▼
         inference/predictor.py
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   Classification        XAI Engine
   (Probabilities)     (Grad-CAM / LIME)
        │                     │
        └──────────┬──────────┘
                   ▼
       Local Display / Touchscreen
```

### Raspberry Pi Camera Inference Snippet:
```python
import cv2
from inference.predictor import BrainTumorPredictor
from xai.gradcam import compute_gradcam, build_overlay

# Initialize engine
predictor = BrainTumorPredictor("model/brain_tumor_model.keras")

# Capture frame from Pi Camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

# Run prediction directly on camera frame
results = predictor.predict(frame)
print(f"Predicted: {results['display_name']} ({results['confidence']}%)")

# Compute Grad-CAM
heatmap, _, _, _ = compute_gradcam(predictor.model, results["processed_batch"])
overlay, _ = build_overlay(results["rgb_display"], heatmap)
cv2.imwrite("pi_result.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
```

---

## ⚖️ Research & Educational Disclaimer

This system is developed strictly for research, academic, and educational demonstration purposes. Model predictions and visual explanations (Grad-CAM and LIME) should not be considered a clinical diagnosis or a substitute for assessment by a certified medical professional.
