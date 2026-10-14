# 🧠 Brain Tumor MRI Classification & Explainable AI (XAI) System

A research and educational prototype for brain tumor classification (Glioma, Meningioma, No Tumor, Pituitary) with multi-modal visual explainability using **Grad-CAM** and **LIME**, plus **local LLM-generated patient-friendly explanations** via [Ollama](https://ollama.com) — **no cloud API, no paid service, no API key required**.

> ⚕️ **This is a research prototype, not a medical diagnostic device.** AI predictions must not replace evaluation by a qualified healthcare professional.

---

## 📁 Project Structure

```
BrainTumor_XAI/
├── app.py                          # Streamlit GUI entry point
├── model/
│   ├── efficientnet_final.keras    # EfficientNetB0 model (~50 MB)
│   ├── resnet50_v2_final.keras     # ResNet50 model (~265 MB)
│   └── vgg16_v2_final.keras        # VGG16 model (~112 MB)
├── inference/
│   ├── __init__.py
│   └── predictor.py                # BrainTumorPredictor — model loading & predict()
├── xai/
│   ├── __init__.py
│   ├── gradcam.py                  # Grad-CAM GradientTape implementation
│   ├── lime_explainer.py           # LIME Quickshift superpixel explainer
│   └── superpixel.py               # Quantitative weight ranking bar chart
├── llm/                            # ← NEW: Local LLM integration
│   ├── __init__.py
│   ├── ollama_client.py            # Ollama REST API client (health-check, generate)
│   ├── evidence_builder.py         # Structured evidence from verified outputs only
│   └── explanation_renderer.py     # Prompt builder, parser, deterministic fallback
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py            # Image loaders & model normalizers
│   └── generate_samples.py         # Synthetic sample generator
├── tests/
│   └── test_evidence_builder.py    # Unit tests (no model or Ollama required)
├── sample_data/                    # Sample MRI scans
├── requirements.txt
├── run_app.bat                     # 1-click Windows launcher
└── README.md
```

---

## 🎯 Target Classes

| Index | Class Key | Display Name |
|-------|-----------|--------------|
| 0 | `glioma` | Glioma |
| 1 | `meningioma` | Meningioma |
| 2 | `notumor` | No Tumor |
| 3 | `pituitary` | Pituitary |

---

## 🚀 Quickstart (Laptop)

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama (for AI explanations)

> Skip this if you only want CNN + Grad-CAM + LIME without the plain-language explanation.

**Windows / macOS / Linux:**  
Download and install from [https://ollama.com](https://ollama.com)

**Start the Ollama server** (keep this terminal open):
```bash
ollama serve
```

### 3. Pull a Local Model

Choose one based on your RAM:

| Model | RAM Required | Speed | Command |
|-------|-------------|-------|---------|
| `qwen2.5:3b` *(recommended)* | ~3–4 GB | Fast | `ollama pull qwen2.5:3b` |
| `llama3.2:3b` | ~3–4 GB | Fast | `ollama pull llama3.2:3b` |
| `phi3:mini` | ~2 GB | Very fast | `ollama pull phi3:mini` |
| `llama3.1:8b` | ~8 GB | Slower | `ollama pull llama3.1:8b` |

**Default configured model:** `qwen2.5:3b`  
Change it anytime in the sidebar without editing code.

> 💡 **Raspberry Pi 4 (4 GB)**: Pi 4 is marginal for 3B models; LIME alone can take 30–60 s.  
> **Raspberry Pi 5 (8 GB)**: Should handle 3B models slowly. Use the deterministic fallback for faster response.

### 4. Launch the Application
```bash
streamlit run app.py
```
Or double-click `run_app.bat` on Windows.

---

## 🔍 Explainable AI (XAI) Methods

### 1. Grad-CAM (Visual Attention)
* Uses `GradientTape` to compute gradients w.r.t. the final convolutional layer.
* Target layers: `top_conv` (EfficientNet) · `conv5_block3_out` (ResNet) · `block5_conv3` (VGG16).
* Colormap: `COLORMAP_JET` (blue = low attention → red = high attention).
* **Not a tumor segmentation** — shows model attention, not anatomy.

### 2. LIME (Superpixel Perturbations)
* Segments MRI using **Quickshift** (`kernel_size=4`, `max_dist=200`, `ratio=0.2`).
* Sign convention: **positive weight = supports** predicted class; **negative = opposes**.
* Green boundaries = supporting superpixels; Red boundaries = opposing superpixels.

### 3. Quantitative Superpixel Ranking
* Horizontal bar chart of top 12 superpixels by absolute contribution weight.
* Diverging `RdYlGn` spatial weight map.

### 4. Local LLM Explanation (NEW)
* Uses **Ollama** at `http://localhost:11434` — **no internet, no API key**.
* Receives only structured evidence (class name, model scores, Grad-CAM availability, LIME weights).
* **Never receives raw image pixels or patient data.**
* If Ollama is unavailable: a **clearly labelled deterministic fallback** is shown automatically.
* The LLM cannot change the predicted class or make unsupported clinical claims.

---

## 🤖 Ollama Configuration

| Setting | Default | How to Change |
|---------|---------|---------------|
| Model name | `qwen2.5:3b` | Sidebar text field |
| Generation timeout | 120 s | Sidebar slider |
| API endpoint | `http://localhost:11434` | `llm/ollama_client.py` → `DEFAULT_OLLAMA_BASE_URL` |
| Max output tokens | 1024 | `llm/ollama_client.py` → `MAX_TOKENS_OUT` |
| Temperature | 0.3 | `llm/ollama_client.py` → `payload["options"]["temperature"]` |

**To switch models without restarting:**
1. Pull the new model: `ollama pull <model_name>`
2. Type the model name in the sidebar "Ollama Model Name" field.
3. Click "Check Ollama Status" to verify.

---

## 🧪 How to Test

### Manual Test Checklist

| Test | Expected Behaviour |
|------|-------------------|
| Upload `sample_data/sample_glioma.png` | Predicts Glioma; Grad-CAM + LIME render |
| Upload `sample_data/sample_meningioma.png` | Predicts Meningioma |
| Upload `sample_data/sample_notumor.png` | Predicts No Tumor; explanation says class not claimed as health |
| Upload `sample_data/sample_pituitary.png` | Predicts Pituitary |
| Stop Ollama (`Ctrl+C` in ollama serve terminal), then analyze | Prediction and XAI still work; fallback explanation shown with yellow notice |
| Configure invalid model name (e.g. `notamodel`), keep Ollama running | "Model not found" message + fallback explanation |
| Upload image A, analyze, then upload image B, analyze | Results for image B shown — no stale results from image A |
| Upload a text file or corrupt file | Friendly error message, no crash |
| Click "Check Ollama Status" | Shows green/red status for server + model |

### Unit Tests (no model or Ollama required)
```bash
python -m pytest tests/test_evidence_builder.py -v
```

Tests cover:
- Evidence builder output keys and correctness for all 4 classes
- LIME top-feature extraction and sorting
- Score gap description (clear / moderate / narrow)
- Fallback explanation sections and forbidden content (no invented anatomy)
- LLM response parser (sections found / not found / case-insensitive)
- Ollama client (mocked HTTP: health, model check, generate, timeout)

---

## 🍓 Future Raspberry Pi Deployment

The `inference/`, `xai/`, `llm/`, and `utils/` packages are all **independent of Streamlit** and accept NumPy arrays directly.

### Pi-Specific Considerations

| Component | Pi 4 (4 GB) | Pi 5 (8 GB) |
|-----------|-------------|-------------|
| EfficientNet inference | ~2–4 s | ~1–2 s |
| LIME (250 samples) | ~30–90 s | ~15–40 s |
| Ollama 3B model | Marginal (~10–30 s/token) | Possible (~2–8 s/token) |
| Deterministic fallback | Instant | Instant |

**Recommended Pi strategy**: Use the deterministic fallback by default; offer the LLM explanation as optional slow mode.

---

## 📝 Unresolved / Known Limitations

1. **Grad-CAM failure**: When `success_cam=False`, a zero heatmap is shown. The UI warns the user.
2. **LIME fallback**: When `success=False`, a grid approximation is used. The UI warns the user.
3. **Score label**: "Model Score (softmax output)" — not a calibrated clinical probability.
4. **Anatomical location**: Grad-CAM heatmap location is not computationally verified; the LLM is instructed not to invent anatomical names.
5. **LLM hallucination risk**: Temperature is set to 0.3 and the system prompt forbids clinical claims, but LLM output should always be reviewed by a clinician in any real-world context.
6. **Tests not auto-run**: Run manually with `pytest` before deployment.

---

## ⚖️ Research & Educational Disclaimer

This system is developed strictly for research, academic, and educational demonstration purposes. Model predictions and visual explanations (Grad-CAM and LIME) should not be considered a clinical diagnosis or a substitute for assessment by a certified medical professional.
