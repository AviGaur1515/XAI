"""
=============================================================================
🧠 Brain Tumor MRI Classification & Explainable AI (XAI) System
Laptop Demonstration & Raspberry Pi Ready Architecture
=============================================================================
Changes from original:
  - Added Ollama LLM integration via llm/ package (local, no API key).
  - Added "Understand the Result" patient-friendly explanation section.
  - Relabelled "Prediction Confidence" → "Model Score (softmax output)".
  - Added session-state image hash to invalidate results on image change.
  - Added Ollama status check in sidebar with configuration controls.
  - Added prominent medical disclaimer.
  - Kept all existing CNN, Grad-CAM, LIME and superpixel code unchanged.
"""

import io
import os
import glob
import time
import hashlib
import re
import html as _html
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Import decoupled inference and XAI core (unchanged)
from inference.predictor import BrainTumorPredictor
from xai.gradcam import compute_gradcam, build_overlay
from xai.lime_explainer import run_lime_explanation
from xai.superpixel import generate_superpixel_bar_chart
from utils.preprocessing import CLASS_NAMES, DISPLAY_NAMES

# Import new LLM integration
from llm.ollama_client import OllamaClient, OllamaStatus
from llm.evidence_builder import build_explanation_evidence
from llm.explanation_renderer import generate_explanation, render_fallback_explanation, SECTION_KEYS

# =============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="Brain Tumor MRI — Clinical AI & Explainability",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Medical-AI Styling (Dark Slate Clinical Theme, Glassmorphic Accents, Modern Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Modern Clinical Header */
    .app-header {
        background: linear-gradient(135deg, #0b1329 0%, #111d40 50%, #0c1a38 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 24px 30px;
        color: #ffffff;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6), 0 0 20px rgba(14, 165, 233, 0.12);
        position: relative;
        overflow: hidden;
    }
    .app-header-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(14, 165, 233, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .app-header h1 {
        margin: 0;
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .app-header p {
        margin: 8px 0 0 0;
        font-size: 14px;
        color: #94a3b8;
        line-height: 1.5;
    }

    /* Clinical Disclaimer Box */
    .disclaimer-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-left: 5px solid #ef4444;
        padding: 14px 18px;
        border-radius: 10px;
        font-size: 12.5px;
        color: #cbd5e1;
        margin-top: 10px;
        margin-bottom: 24px;
        line-height: 1.55;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .disclaimer-box strong {
        color: #f87171;
    }

    /* Metric / Prediction Card */
    .pred-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.4);
        position: relative;
    }
    .pred-badge {
        display: inline-block;
        padding: 7px 18px;
        border-radius: 9999px;
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    .badge-glioma { 
        background-color: rgba(239, 68, 68, 0.15); 
        color: #fca5a5; 
        border: 1px solid rgba(239, 68, 68, 0.4); 
    }
    .badge-meningioma { 
        background-color: rgba(245, 158, 11, 0.15); 
        color: #fcd34d; 
        border: 1px solid rgba(245, 158, 11, 0.4); 
    }
    .badge-notumor { 
        background-color: rgba(16, 185, 129, 0.15); 
        color: #6ee7b7; 
        border: 1px solid rgba(16, 185, 129, 0.4); 
    }
    .badge-pituitary { 
        background-color: rgba(99, 102, 241, 0.15); 
        color: #a5b4fc; 
        border: 1px solid rgba(99, 102, 241, 0.4); 
    }

    /* Probability Bar Container */
    .prob-bar-container {
        margin-bottom: 14px;
    }
    .prob-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 13.5px;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 5px;
    }
    .prob-bar-bg {
        background-color: #1f2937;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Explanation Section Cards */
    .explanation-section {
        background: #111827;
        border: 1px solid #1f2937;
        border-left: 4px solid #0284c7;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .explanation-section h4 {
        margin: 0 0 10px 0;
        font-size: 16px;
        font-weight: 700;
        color: #38bdf8;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .explanation-section p {
        margin: 0;
        font-size: 14.5px;
        line-height: 1.7;
        color: #cbd5e1;
    }

    /* Notice badges */
    .fallback-notice {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 13px;
        color: #fde68a;
        margin-bottom: 18px;
        line-height: 1.55;
    }
    .llm-active-notice {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 4px solid #10b981;
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 13px;
        color: #6ee7b7;
        margin-bottom: 18px;
        line-height: 1.5;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL INITIALIZATION (CACHED)
# =============================================================================
@st.cache_resource
def get_predictor():
    """Initializes and caches the prediction engine."""
    return BrainTumorPredictor()

predictor = get_predictor()


# =============================================================================
# SIDEBAR CONTROLS & MODEL SETTINGS
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    # Model Status
    if predictor.is_loaded:
        st.success(f"**Model Loaded**: `{os.path.basename(predictor.model_path)}`")
        st.caption(f"Backbone: **{predictor.model_name.upper()}** | Input: **{predictor.input_shape}**")
    else:
        st.info("💡 **Model Checkpoint**: Place a `.keras` or `.h5` model in the `model/` folder.")
        st.caption("Currently operating in verified pipeline demonstration mode.")

    # Option to select custom model checkpoint if multiple exist
    model_files = glob.glob("model/*.keras") + glob.glob("model/*.h5") + glob.glob("*.keras") + glob.glob("*.h5")
    if model_files:
        def _format_cnn_label(path: str) -> str:
            base = os.path.basename(path)
            clean = base.replace("_final.keras", "").replace(".keras", "").replace(".h5", "").upper()
            if "efficientnet" in base.lower():
                return f"⭐ EfficientNet-B0 (Recommended · {base})"
            elif "resnet" in base.lower():
                return f"🧠 ResNet-50 v2 ({base})"
            elif "vgg" in base.lower():
                return f"🔬 VGG-16 v2 ({base})"
            return f"📁 {clean} ({base})"

        selected_model_file = st.selectbox(
            "Select CNN Architecture",
            options=model_files,
            index=0,
            format_func=_format_cnn_label,
            help="Select which pre-trained deep learning backbone to use for inference and XAI."
        )
        if st.button("🔄 Reload Selected Model", use_container_width=True):
            try:
                predictor.load_model_from_file(selected_model_file)
                st.success("Model reloaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    st.markdown("---")
    st.markdown("### 🔬 XAI Parameters")
    gradcam_alpha = st.slider("Grad-CAM Overlay Opacity (α)", min_value=0.1, max_value=0.9, value=0.50, step=0.05)
    lime_samples = st.slider("LIME Perturbation Samples", min_value=100, max_value=800, value=250, step=50, 
                             help="Higher values provide finer explanations but take slightly longer.")
    lime_features = st.slider("Top Superpixels to Highlight", min_value=3, max_value=10, value=5, step=1)
    
    st.markdown("---")

    # ------------------------------------------------------------------
    # Ollama LLM Configuration (Smart Dropdown + Auto-Discovery)
    # ------------------------------------------------------------------
    st.markdown("### 🤖 Local AI Explanation (Ollama)")
    st.caption("Runs entirely on your device. Zero internet or API key required.")
    
    # Query installed models if Ollama is running
    temp_client = OllamaClient()
    installed_models = temp_client.get_installed_models()

    RECOMMENDED_OLLAMA_MODELS = [
        "qwen2.5:3b",
        "llama3.2:3b",
        "llama3.2:1b",
        "phi3:mini",
        "mistral:7b",
    ]
    
    ollama_dropdown_options = list(dict.fromkeys(installed_models + RECOMMENDED_OLLAMA_MODELS + ["Custom (enter manually)"]))
    
    def _format_ollama_label(m: str) -> str:
        if m == "Custom (enter manually)":
            return "✏️ Custom Model (Enter manually)..."
        is_inst = m in installed_models
        badge = "✅ Installed: " if is_inst else "⚡ Recommended: "
        if "qwen2.5:3b" in m:
            return f"{badge}Qwen 2.5 (3B) — Fast & accurate"
        elif "llama3.2:3b" in m:
            return f"{badge}Llama 3.2 (3B) — Meta standard"
        elif "llama3.2:1b" in m:
            return f"{badge}Llama 3.2 (1B) — Ultra-light (low RAM)"
        elif "phi3" in m:
            return f"{badge}Phi-3 Mini (3.8B) — Microsoft model"
        elif "mistral" in m:
            return f"{badge}Mistral (7B) — High capability (needs 6GB+ RAM)"
        return f"{'✅ ' if is_inst else '📦 '}{m}"

    default_llm_idx = 0
    if "qwen2.5:3b" in ollama_dropdown_options:
        default_llm_idx = ollama_dropdown_options.index("qwen2.5:3b")

    selected_llm_choice = st.selectbox(
        "Select Local LLM Model",
        options=ollama_dropdown_options,
        index=default_llm_idx,
        format_func=_format_ollama_label,
        help="Choose a local model running in Ollama. 'qwen2.5:3b' is recommended for clinical reasoning."
    )

    if selected_llm_choice == "Custom (enter manually)":
        ollama_model = st.text_input("Enter Ollama model name", value="qwen2.5:3b", help="e.g. gemma2:2b, deepseek-r1:1.5b")
    else:
        ollama_model = selected_llm_choice

    ollama_timeout = st.slider(
        "Generation Timeout (seconds)", min_value=30, max_value=300, value=120, step=15
    )
    
    # Live status check
    ollama_client = OllamaClient(model_name=ollama_model, timeout=ollama_timeout)
    
    if st.button("🔍 Check Ollama Status", use_container_width=True):
        health = ollama_client.check_health()
        if health == OllamaStatus.OK:
            model_status = ollama_client.check_model()
            if model_status == OllamaStatus.OK:
                st.success(f"✅ Ollama running · Model `{ollama_model}` is ready.")
            elif model_status == OllamaStatus.MODEL_NOT_FOUND:
                installed = ollama_client.get_installed_models()
                st.warning(
                    f"⚠️ Model `{ollama_model}` not found.\n\n"
                    f"Run in terminal: `ollama pull {ollama_model}`\n\n"
                    f"Installed: {', '.join(installed) if installed else 'none'}"
                )
            else:
                st.error(OllamaClient.status_message(model_status))
        else:
            st.error(
                "❌ Ollama is not running.\n\n"
                "Start it with: `ollama serve`\n\n"
                "Install from: https://ollama.com"
            )

    enable_llm = st.checkbox(
        "Enable AI Explanation",
        value=True,
        help="When enabled, the app calls the local Ollama model to generate a plain-language explanation. "
             "CNN prediction and XAI work regardless of this setting."
    )

    st.markdown("---")
    st.markdown("### 📁 Load Sample Scans")
    selected_sample = None
    sample_files = sorted(
        glob.glob("sample_data/*.png") + 
        glob.glob("sample_data/*.jpg") + 
        glob.glob("sample_data/*.jpeg") +
        glob.glob("sample_data/*.PNG") + 
        glob.glob("sample_data/*.JPG") + 
        glob.glob("sample_data/*.JPEG")
    )
    if sample_files:
        sample_names = {os.path.basename(f): f for f in sample_files}
        picked = st.selectbox("Choose a sample MRI:", ["-- None --"] + list(sample_names.keys()))
        if picked != "-- None --":
            selected_sample = sample_names[picked]

    st.markdown("---")
    st.caption("🚀 **Handheld Architecture**: Decoupled inference modules ready for Raspberry Pi camera deployment.")


# =============================================================================
# HEADER SECTION
# =============================================================================
st.markdown("""
<div class="app-header">
    <div class="app-header-tag">🔬 Clinical AI &amp; Explainability Suite</div>
    <h1>🧠 Brain Tumor MRI Classification &amp; Explainable AI</h1>
    <p>Multi-backbone deep neural network classification with gradient attention maps (Grad-CAM), local feature attribution (LIME), and plain-language clinical insights.</p>
</div>
""", unsafe_allow_html=True)

# Prominent disclaimer near top
st.markdown("""
<div class="disclaimer-box" style="margin-top: 0; margin-bottom: 24px;">
    <strong>⚕️ Research &amp; Educational Prototype — Not a Medical Diagnostic Device:</strong><br>
    This application is a research and educational prototype, not a medical diagnostic device.
    AI predictions and Grad-CAM/LIME visualizations must not replace evaluation by a qualified healthcare professional.
    If you have a real health concern, consult a licensed radiologist, neurologist, or physician.
</div>
""", unsafe_allow_html=True)


# =============================================================================
# INPUT SECTION: IMAGE UPLOAD / SELECTION
# =============================================================================

def _compute_image_hash(image_input) -> str:
    """Computes a stable hash of image bytes to detect when the user changes the image."""
    try:
        if hasattr(image_input, "getvalue"):
            data = image_input.getvalue()
        elif isinstance(image_input, str):
            with open(image_input, "rb") as f:
                data = f.read()
        elif isinstance(image_input, bytes):
            data = image_input
        else:
            data = str(image_input).encode()
        return hashlib.md5(data).hexdigest()
    except Exception:
        return str(id(image_input))


col_input, col_preview = st.columns([1.2, 1])

with col_input:
    st.markdown("#### 1. Upload Brain MRI")
    uploaded_file = st.file_uploader(
        "Upload Brain MRI (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload an axial or coronal brain MRI scan."
    )
    
    # Use uploaded file or sidebar sample
    active_image = None
    if uploaded_file is not None:
        active_image = uploaded_file
    elif selected_sample is not None:
        active_image = selected_sample
        st.info(f"Loaded sample scan: **{os.path.basename(selected_sample)}**")

with col_preview:
    st.markdown("#### 2. Scan Preview")
    if active_image is not None:
        try:
            if isinstance(active_image, str):
                preview_pil = Image.open(active_image)
            elif hasattr(active_image, "getvalue"):
                preview_pil = Image.open(io.BytesIO(active_image.getvalue()))
            else:
                if hasattr(active_image, "seek"):
                    active_image.seek(0)
                preview_pil = Image.open(active_image)
                if hasattr(active_image, "seek"):
                    active_image.seek(0)
            
            st.image(preview_pil, caption=f"Input Image ({preview_pil.size[0]}×{preview_pil.size[1]} px)", width=240)
        except Exception as e:
            st.error(f"Error reading image: {e}")
            active_image = None
    else:
        st.markdown(
            """
            <div style="border: 2px dashed #cbd5e1; border-radius: 8px; padding: 40px 20px; text-align: center; color: #94a3b8;">
                📷 No MRI scan selected yet.<br>Please upload an image or choose a sample from the sidebar.
            </div>
            """, 
            unsafe_allow_html=True
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Session-state image hash: invalidate cached results when image changes
# ---------------------------------------------------------------------------
if active_image is not None:
    current_hash = _compute_image_hash(active_image)
else:
    current_hash = None

if "last_image_hash" not in st.session_state:
    st.session_state["last_image_hash"] = None

if current_hash != st.session_state["last_image_hash"]:
    # Image changed — clear any cached analysis results
    for key in ["analysis_results", "explanation_sections", "llm_status", "is_fallback"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["last_image_hash"] = current_hash

# Action Buttons
btn_col1, btn_col2, btn_spacer = st.columns([1.2, 1, 3])
with btn_col1:
    analyze_btn = st.button("🔬 Analyze MRI", type="primary", use_container_width=True, disabled=(active_image is None))
with btn_col2:
    if st.button("🔄 Reset / Clear", use_container_width=True):
        for key in ["analysis_results", "explanation_sections", "llm_status", "is_fallback", "last_image_hash"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# =============================================================================
# ANALYSIS & PREDICTION PIPELINE
# =============================================================================
if analyze_btn and active_image is not None:
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Preprocessing
        status_text.markdown("⏳ **Step 1/4**: Preprocessing MRI scan...")
        progress_bar.progress(20)
        time.sleep(0.15)

        # Step 2: Model Inference
        status_text.markdown("⏳ **Step 2/4**: Running deep learning classifier...")
        img_payload = active_image.getvalue() if hasattr(active_image, "getvalue") else active_image
        pred_res = predictor.predict(img_payload)
        progress_bar.progress(45)
        time.sleep(0.15)

        # Step 3: Grad-CAM Explainability
        status_text.markdown("⏳ **Step 3/4**: Computing Grad-CAM visual attention...")
        heatmap, pred_idx, pred_conf, success_cam = compute_gradcam(
            predictor.model,
            pred_res["processed_batch"],
            model_name=predictor.model_name,
            pred_index=pred_res["pred_index"]
        )
        overlay, hm_resized = build_overlay(pred_res["rgb_display"], heatmap, alpha=gradcam_alpha)
        progress_bar.progress(70)

        # Step 4: LIME Explainability
        status_text.markdown(f"⏳ **Step 4/4**: Performing LIME superpixel perturbations ({lime_samples} samples)...")
        lime_res = run_lime_explanation(
            predictor.model,
            pred_res["uint8_img"],
            model_name=predictor.model_name,
            pred_index=pred_res["pred_index"],
            num_samples=lime_samples,
            num_features=lime_features
        )
        progress_bar.progress(100)
        status_text.markdown("✅ **Analysis Complete!**")
        time.sleep(0.3)
        status_text.empty()
        progress_bar.empty()

        # Cache results in session state so they persist and don't stale on re-runs
        st.session_state["analysis_results"] = {
            "pred_res": pred_res,
            "heatmap": heatmap,
            "hm_resized": hm_resized,
            "overlay": overlay,
            "success_cam": success_cam,
            "lime_res": lime_res,
        }

    except Exception as e:
        st.error(f"⚠️ An error occurred during analysis: {e}")
        st.exception(e)


# =============================================================================
# RESULTS DISPLAY (from session state to avoid stale display on re-run)
# =============================================================================
if "analysis_results" in st.session_state:
    r = st.session_state["analysis_results"]
    pred_res = r["pred_res"]
    heatmap = r["heatmap"]
    hm_resized = r["hm_resized"]
    overlay = r["overlay"]
    success_cam = r["success_cam"]
    lime_res = r["lime_res"]

    # =====================================================================
    # PREDICTION RESULT SECTION
    # =====================================================================
    st.markdown("## 📊 Classification Results")
    
    res_col1, res_col2 = st.columns([1, 1.2])
    
    with res_col1:
        badge_class = f"badge-{pred_res['pred_class'].lower()}"
        st.markdown(f"""
        <div class="pred-card">
            <div style="font-size: 11.5px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">
                AI Predicted Class
            </div>
            <div class="pred-badge {badge_class}">
                {pred_res['display_name']}
            </div>
            <div style="margin-top: 20px; font-size: 11.5px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px;">
                Model Score (Softmax Output)
            </div>
            <div style="font-size: 38px; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;">
                {pred_res['confidence']:.2f}%
            </div>
            <div style="font-size: 12px; color: #94a3b8; margin-top: 6px; line-height: 1.4;">
                ⚠️ Uncalibrated model output, not a clinical probability.
            </div>
            <div style="font-size: 11.5px; color: #64748b; margin-top: 12px; padding-top: 8px; border-top: 1px solid #1f2937;">
                Architecture: <strong style="color: #cbd5e1;">{predictor.model_name.upper()}</strong> ({predictor.input_shape[0]}×{predictor.input_shape[1]})
            </div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        prob_html = []
        prob_html.append('<div class="pred-card">')
        prob_html.append(
            '<div style="font-size: 15px; font-weight: 700; color: #f8fafc; margin-bottom: 4px; border-bottom: 1px solid #1f2937; padding-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">'
            '<span>Model Score Distribution</span>'
            '<span style="font-size: 11px; font-weight: 600; color: #94a3b8; background: #1f2937; padding: 2px 8px; border-radius: 10px;">4 Classes</span>'
            '</div>'
        )
        prob_html.append('<div style="font-size: 12px; color: #94a3b8; margin-bottom: 14px; margin-top: 6px;">Comparative softmax scores across all categories.</div>')
        
        bar_colors = {
            "glioma": "#ef4444",
            "meningioma": "#f59e0b",
            "notumor": "#10b981",
            "pituitary": "#6366f1"
        }
        
        for cls in CLASS_NAMES:
            prob = pred_res["probabilities"][cls] * 100.0
            color = bar_colors.get(cls, "#0284c7")
            is_top = (cls == pred_res["pred_class"])
            
            if is_top:
                label_style = "font-weight: 800; color: #f8fafc; font-size: 14px;"
                pct_style = f"font-weight: 800; color: {color}; font-size: 14px;"
                bar_glow = f"box-shadow: 0 0 10px {color}66;"
            else:
                label_style = "font-weight: 500; color: #94a3b8; font-size: 13.5px;"
                pct_style = "font-weight: 600; color: #cbd5e1; font-size: 13.5px;"
                bar_glow = ""
            
            prob_html.append(
                f'<div class="prob-bar-container">'
                f'<div class="prob-bar-label">'
                f'<span style="{label_style}">{DISPLAY_NAMES[cls]}</span>'
                f'<span style="{pct_style}">{prob:.2f}%</span>'
                f'</div>'
                f'<div class="prob-bar-bg">'
                f'<div class="prob-bar-fill" style="width: {prob}%; background-color: {color}; {bar_glow}"></div>'
                f'</div>'
                f'</div>'
            )
            
        prob_html.append('</div>')
        st.markdown("".join(prob_html), unsafe_allow_html=True)

    # =====================================================================
    # PATIENT-FRIENDLY EXPLANATION SECTION
    # =====================================================================
    st.markdown("## 💬 Understand the Result")
    st.caption(
        "Plain-language explanation of what the AI found and what it means. "
        "Generated from the actual model scores and XAI outputs for this image."
    )

    # Build evidence (only verified, computed values)
    evidence = build_explanation_evidence(
        pred_res,
        lime_res=lime_res,
        gradcam_success=success_cam,
        lime_features=lime_features,
    )

    # Generate or retrieve from session state
    if "explanation_sections" not in st.session_state:
        if enable_llm:
            with st.spinner("🤖 Generating plain-language explanation with local Ollama LLM..."):
                sections, llm_status, is_fallback = generate_explanation(evidence, ollama_client)
        else:
            sections = render_fallback_explanation(evidence)
            llm_status = OllamaStatus.OLLAMA_NOT_RUNNING
            is_fallback = True
        st.session_state["explanation_sections"] = sections
        st.session_state["llm_status"] = llm_status
        st.session_state["is_fallback"] = is_fallback
    else:
        sections = st.session_state["explanation_sections"]
        llm_status = st.session_state.get("llm_status", OllamaStatus.OK)
        is_fallback = st.session_state.get("is_fallback", True)

    # Show fallback notice if applicable
    if is_fallback:
        status_msg = OllamaClient.status_message(llm_status) if llm_status != OllamaStatus.OK else ""
        notice_extra = f" ({status_msg})" if status_msg else ""
        st.markdown(
            f'<div class="fallback-notice">'
            f'📋 <strong>Structured Clinical Explanation (Deterministic Engine):</strong> '
            f'The local Ollama instance was offline or unconfigured{notice_extra}. '
            f'The explanation below is synthesized directly from verified model softmax distributions, score margins, and Grad-CAM/LIME spatial attributions. '
            f'It is fully accurate to the computed visual evidence.'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="llm-active-notice">'
            '✅ <strong>Generative Clinical Explanation (Local Ollama LLM):</strong> '
            'Synthesized entirely on-device with zero cloud data transmission.'
            '</div>',
            unsafe_allow_html=True
        )

    # Render each section
    section_icons = {
        "What the AI Predicted": "🔮",
        "What Influenced the Prediction": "🎯",
        "Understanding the Visual Explanations": "🖼️",
        "General Information About This Class": "📚",
        "Important Limitations": "⚠️",
        "Suggested Next Step": "🏥",
    }

    def _format_card_markdown(text: str) -> str:
        """Converts Markdown formatting (bold, italic, code) into safe HTML to prevent raw asterisks."""
        safe = _html.escape(text, quote=False)
        # Bold: **text** -> <strong>
        safe = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color: #38bdf8; font-weight: 700;">\1</strong>', safe)
        # Italic: *text* -> <em>
        safe = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em style="color: #f1f5f9;">\1</em>', safe)
        # Inline code: `code`
        safe = re.sub(r'`([^`]+)`', r'<code style="background: rgba(51, 65, 85, 0.6); padding: 2px 6px; border-radius: 4px; font-size: 13px; color: #38bdf8;">\1</code>', safe)
        # Convert double newlines to paragraphs, single to <br>
        safe = safe.replace("\n\n", "</p><p style=\"margin-top: 10px; margin-bottom: 0;\">").replace("\n", "<br>")
        return safe

    for key in SECTION_KEYS:
        content = sections.get(key, "").strip()
        if not content:
            continue
        icon = section_icons.get(key, "•")
        formatted_html = _format_card_markdown(content)
        st.markdown(
            f'<div class="explanation-section">'
            f'<h4><span>{icon}</span> {key}</h4>'
            f'<p>{formatted_html}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

    # =====================================================================
    # EXPLAINABLE AI SECTION
    # =====================================================================
    st.markdown("## 🔍 Visual Evidence (XAI)")
    
    tab_gradcam, tab_lime, tab_superpixels = st.tabs([
        "🔥 Grad-CAM Visual Attention", 
        "🧩 LIME Superpixel Analysis", 
        "📊 Advanced / Technical XAI"
    ])

    # --- TAB 1: GRAD-CAM ---
    with tab_gradcam:
        st.markdown("### Gradient-Weighted Class Activation Mapping (Grad-CAM)")
        st.markdown(
            f"Grad-CAM shows which parts of the MRI the model paid most attention to when predicting "
            f"**{pred_res['display_name']}**. It does **not** show exact tumor boundaries."
        )
        
        g_col1, g_col2, g_col3 = st.columns(3)
        with g_col1:
            st.image(pred_res["rgb_display"], caption="1. Original MRI Scan", use_container_width=True)
        with g_col2:
            hm_colored = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
            hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
            st.image(hm_colored, caption="2. Grad-CAM Heatmap\nBlue = low attention → Red = high attention", use_container_width=True)
        with g_col3:
            st.image(overlay, caption=f"3. Heatmap Overlay (opacity α={gradcam_alpha})", use_container_width=True)

        if not success_cam:
            st.warning(
                "⚠️ Grad-CAM gradients could not be computed for this model/image combination. "
                "The heatmap shown is a placeholder."
            )
        else:
            st.info(
                "💡 **Reading this heatmap**: Red/yellow areas indicate where the model's internal signals "
                "were strongest for the predicted class. Blue areas had little influence. "
                "These are model-attention signals, not a medical segmentation of tissue."
            )

    # --- TAB 2: LIME ---
    with tab_lime:
        st.markdown("### LIME — Local Interpretable Model-agnostic Explanations")
        st.markdown(
            "LIME divides the MRI into regions (superpixels) and tests which regions matter most "
            "by temporarily hiding each one and observing how the model's output changes."
        )
        
        l_col1, l_col2, l_col3, l_col4 = st.columns(4)
        with l_col1:
            st.image(lime_res["segments_image"], caption=f"1. Superpixel Regions\n({lime_res['num_segments']} segments, yellow borders)", use_container_width=True)
        with l_col2:
            st.image(lime_res["positive_only_image"], caption=f"2. Supporting Regions\n(Green = helped predict {pred_res['display_name']})", use_container_width=True)
        with l_col3:
            st.image(lime_res["pos_neg_image"], caption="3. Supporting vs. Opposing\nGreen = supports  |  Red = opposes", use_container_width=True)
        with l_col4:
            w_map = lime_res["weight_map"]
            vmax = float(np.max(np.abs(w_map))) if float(np.max(np.abs(w_map))) > 1e-8 else 1.0
            norm_w = np.uint8(np.clip(((w_map + vmax) / (2 * vmax)) * 255, 0, 255))
            w_colored = cv2.applyColorMap(norm_w, cv2.COLORMAP_TURBO)
            w_colored = cv2.cvtColor(w_colored, cv2.COLOR_BGR2RGB)
            st.image(w_colored, caption="4. Weight Intensity Map\n(Warm = strongly supporting)", use_container_width=True)

        if not lime_res.get("success", False):
            st.warning(
                "⚠️ LIME ran in fallback mode (a grid approximation was used). "
                "Install `lime` and `scikit-image` for full LIME analysis."
            )
        else:
            st.info(
                "💡 **Reading LIME**: Green-bordered regions increased the model's score for "
                f"**{pred_res['display_name']}** when present. Red-bordered regions lowered it. "
                "These are statistical observations from thousands of random perturbations — not direct tumor markers."
            )

    # --- TAB 3: ADVANCED / TECHNICAL XAI ---
    with tab_superpixels:
        st.markdown("### Advanced / Technical XAI Details")
        st.markdown(
            "Quantitative superpixel contribution weights from LIME. "
            "Each bar shows how much a specific image region increased (green) or decreased (red) "
            "the model's score for the predicted class."
        )
        
        if "explanation" in lime_res and lime_res["explanation"] is not None:
            fig = generate_superpixel_bar_chart(
                lime_res["explanation"],
                pred_index=pred_res["pred_index"],
                top_n=12
            )
            st.pyplot(fig)
        else:
            st.warning("Superpixel explanation data is not available for chart generation.")

        with st.expander("📋 Raw Model Scores"):
            for cls in CLASS_NAMES:
                score = pred_res["probabilities"][cls]
                st.text(f"  {DISPLAY_NAMES[cls]:>15s} ({cls}):  {score:.6f}  ({score*100:.3f}%)")

        with st.expander("📋 LIME Feature Weights (Top Features)"):
            lime_ev = evidence.get("lime", {})
            if lime_ev.get("available") and lime_ev.get("top_features"):
                for f in lime_ev["top_features"]:
                    direction_icon = "✅" if f["direction"] == "supports" else "❌"
                    st.text(
                        f"  Rank {f['rank']:2d} | Superpixel S{f['seg_id']:4d} | "
                        f"weight: {f['weight']:+.5f} | {direction_icon} {f['direction']}"
                    )
            else:
                st.text("LIME feature weights not available (ran in fallback mode).")


# =============================================================================
# MEDICAL & RESEARCH DISCLAIMER
# =============================================================================
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Research &amp; Educational Disclaimer:</strong><br>
    This system is developed strictly for research, academic, and educational demonstration purposes.
    Model predictions and visual explanations (Grad-CAM / LIME) should not be considered a definitive clinical diagnosis
    or a substitute for professional clinical judgment by a certified radiologist, neuro-oncologist, or medical practitioner.
    AI predictions are softmax model outputs and have not been clinically validated. Always consult a qualified professional.
</div>
""", unsafe_allow_html=True)
