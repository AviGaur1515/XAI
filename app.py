"""
=============================================================================
🧠 Brain Tumor MRI Classification & Explainable AI (XAI) System
Laptop Demonstration & Raspberry Pi Ready Architecture
=============================================================================
"""

import io
import os
import glob
import time
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Import decoupled inference and XAI core
from inference.predictor import BrainTumorPredictor
from xai.gradcam import compute_gradcam, build_overlay
from xai.lime_explainer import run_lime_explanation
from xai.superpixel import generate_superpixel_bar_chart
from utils.preprocessing import CLASS_NAMES, DISPLAY_NAMES

# =============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="Brain Tumor MRI — Explainable AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Medical-AI Styling (Light theme, clean cards, modern typography)
st.markdown("""
<style>
    /* Main Background & Text */
    .main {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Title & Header */
    .app-header {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.15);
    }
    .app-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .app-header p {
        margin: 6px 0 0 0;
        font-size: 15px;
        opacity: 0.92;
    }
    
    /* Metric / Prediction Card */
    .pred-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    .pred-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-glioma { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
    .badge-meningioma { background-color: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    .badge-notumor { background-color: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
    .badge-pituitary { background-color: #e0e7ff; color: #4338ca; border: 1px solid #c7d2fe; }

    /* Probability Bar Container */
    .prob-bar-container {
        margin-bottom: 12px;
    }
    .prob-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        font-weight: 600;
        color: #334155;
        margin-bottom: 4px;
    }
    .prob-bar-bg {
        background-color: #e2e8f0;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }
    
    /* Medical Disclaimer Footer */
    .disclaimer-box {
        background-color: #f1f5f9;
        border-left: 4px solid #64748b;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 12px;
        color: #475569;
        margin-top: 30px;
        line-height: 1.5;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
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
        st.info("💡 **Model Checkpoint**: Looking in `model/` directory. Place `brain_tumor_model.keras` in the `model/` folder to run live weights.")
        st.caption("Currently operating in verified pipeline demonstration mode.")

    # Option to select custom model checkpoint if multiple exist
    model_files = glob.glob("model/*.keras") + glob.glob("model/*.h5") + glob.glob("*.keras") + glob.glob("*.h5")
    if model_files:
        selected_model_file = st.selectbox("Select Model Checkpoint", options=model_files, index=0)
        if st.button("Reload Selected Model", use_container_width=True):
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
    st.markdown("### 📁 Load Sample Scans")
    sample_files = glob.glob("sample_data/*.png")
    selected_sample = None
    if sample_files:
        sample_names = {os.path.basename(f): f for f in sample_files}
        picked = st.selectbox("Choose a sample MRI:", ["-- None --"] + list(sample_names.keys()))
        if picked != "-- None --":
            selected_sample = sample_names[picked]

    st.markdown("---")
    st.caption("🚀 **Handheld Architecture**: This engine uses decoupled inference modules ready for Raspberry Pi camera deployment.")


# =============================================================================
# HEADER SECTION
# =============================================================================
st.markdown("""
<div class="app-header">
    <h1>🧠 Brain Tumor MRI Classification & Explainable AI</h1>
    <p>AI-assisted classification and visual explanation of brain MRI images (Grad-CAM & LIME)</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# INPUT SECTION: IMAGE UPLOAD / SELECTION
# =============================================================================
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

# Action Buttons
btn_col1, btn_col2, btn_spacer = st.columns([1.2, 1, 3])
with btn_col1:
    analyze_btn = st.button("🔬 Analyze MRI", type="primary", use_container_width=True, disabled=(active_image is None))
with btn_col2:
    if st.button("🔄 Reset / Clear", use_container_width=True):
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

        # =====================================================================
        # 7. PREDICTION RESULT SECTION
        # =====================================================================
        st.markdown("## 📊 Classification Results")
        
        res_col1, res_col2 = st.columns([1, 1.2])
        
        with res_col1:
            badge_class = f"badge-{pred_res['pred_class'].lower()}"
            st.markdown(f"""
            <div class="pred-card">
                <div style="font-size: 13px; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 6px;">
                    Predicted Diagnosis
                </div>
                <div class="pred-badge {badge_class}">
                    {pred_res['display_name']}
                </div>
                <div style="margin-top: 18px; font-size: 13px; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 4px;">
                    Prediction Confidence
                </div>
                <div style="font-size: 32px; font-weight: 800; color: #0f172a;">
                    {pred_res['confidence']:.2f}%
                </div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
                    Model: {predictor.model_name.upper()} ({predictor.input_shape[0]}×{predictor.input_shape[1]})
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            prob_html = []
            prob_html.append('<div class="pred-card">')
            prob_html.append('<div style="font-size: 15px; font-weight: 700; color: #1e293b; margin-bottom: 16px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;">Class Probability Distribution</div>')
            
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
                    label_style = "font-weight: 800; color: #0f172a; font-size: 14px;"
                    pct_style = "font-weight: 800; color: #0f172a; font-size: 14px;"
                else:
                    label_style = "font-weight: 600; color: #475569; font-size: 14px;"
                    pct_style = "font-weight: 600; color: #475569; font-size: 14px;"
                
                # Single-line string concatenation with zero leading space avoids markdown code-block triggers
                prob_html.append(
                    f'<div class="prob-bar-container" style="margin-bottom: 16px;">'
                    f'<div class="prob-bar-label" style="display: flex; justify-content: space-between; margin-bottom: 4px;">'
                    f'<span style="{label_style}">{DISPLAY_NAMES[cls]}</span>'
                    f'<span style="{pct_style}">{prob:.2f}%</span>'
                    f'</div>'
                    f'<div class="prob-bar-bg" style="background-color: #e2e8f0; border-radius: 6px; height: 12px; overflow: hidden; width: 100%;">'
                    f'<div class="prob-bar-fill" style="width: {prob}%; background-color: {color}; height: 100%; border-radius: 6px;"></div>'
                    f'</div>'
                    f'</div>'
                )
                
            prob_html.append('</div>')
            st.markdown("".join(prob_html), unsafe_allow_html=True)

        # =====================================================================
        # 8. EXPLAINABLE AI SECTION
        # =====================================================================
        st.markdown("## 🔍 Explainable AI (XAI) Diagnosis Rationale")
        
        tab_gradcam, tab_lime, tab_superpixels = st.tabs([
            "🔥 Grad-CAM Visual Attention", 
            "🧩 LIME Superpixel Analysis", 
            "📊 Superpixel Importance Ranking"
        ])

        # --- TAB 1: GRAD-CAM ---
        with tab_gradcam:
            st.markdown("### Gradient-Weighted Class Activation Mapping (Grad-CAM)")
            st.markdown(
                "Grad-CAM visualizes the spatial regions within the MRI scan that produced the highest activation "
                f"gradients for the predicted diagnosis (**{pred_res['display_name']}**)."
            )
            
            g_col1, g_col2, g_col3 = st.columns(3)
            with g_col1:
                st.image(pred_res["rgb_display"], caption="1. Original MRI Scan", use_container_width=True)
            with g_col2:
                # Format heatmap with JET colormap for standalone display
                hm_colored = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
                hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
                st.image(hm_colored, caption="2. Grad-CAM Activation Heatmap", use_container_width=True)
            with g_col3:
                st.image(overlay, caption=f"3. Attention Overlay (α={gradcam_alpha})", use_container_width=True)

            st.info(
                "💡 **Clinical Interpretation Note**: The highlighted red/yellow areas indicate where the deep neural network focused attention. "
                "These regions represent model salience rather than direct physiological tumor margins."
            )

        # --- TAB 2: LIME ---
        with tab_lime:
            st.markdown("### Local Interpretable Model-agnostic Explanations (LIME)")
            st.markdown(
                "LIME segments the MRI scan into discrete superpixels using Quickshift and perturbs regions "
                "to evaluate which areas locally support or oppose the classification."
            )
            
            l_col1, l_col2, l_col3, l_col4 = st.columns(4)
            with l_col1:
                st.image(lime_res["segments_image"], caption=f"1. Quickshift Superpixels ({lime_res['num_segments']} segments)", use_container_width=True)
            with l_col2:
                st.image(lime_res["positive_only_image"], caption=f"2. Positive Supporting Features (Top {lime_features})", use_container_width=True)
            with l_col3:
                st.image(lime_res["pos_neg_image"], caption="3. Green = Supports | Red = Opposes", use_container_width=True)
            with l_col4:
                # Normalize weight map for display
                w_map = lime_res["weight_map"]
                vmax = float(np.max(np.abs(w_map))) if float(np.max(np.abs(w_map))) > 1e-8 else 1.0
                norm_w = np.uint8(np.clip(((w_map + vmax) / (2 * vmax)) * 255, 0, 255))
                w_colored = cv2.applyColorMap(norm_w, cv2.COLORMAP_TURBO)
                w_colored = cv2.cvtColor(w_colored, cv2.COLOR_BGR2RGB)
                st.image(w_colored, caption="4. Superpixel Weight Field", use_container_width=True)

            st.info(
                "💡 **LIME Explanation**: Green boundaries indicate superpixels that increased model confidence for "
                f"**{pred_res['display_name']}**, while red boundaries indicate features that lowered model confidence."
            )

        # --- TAB 3: SUPERPIXEL IMPORTANCE RANKING ---
        with tab_superpixels:
            st.markdown("### Quantitative Superpixel Weight Distribution")
            st.markdown("Ranks the top most influential superpixels by their numerical contribution weight (w).")
            
            if "explanation" in lime_res and lime_res["explanation"] is not None:
                fig = generate_superpixel_bar_chart(
                    lime_res["explanation"],
                    pred_index=pred_res["pred_index"],
                    top_n=12
                )
                st.pyplot(fig)
            else:
                st.warning("Superpixel explanation data is not available for chart generation.")

    except Exception as e:
        st.error(f"⚠️ An error occurred during analysis: {e}")
        st.exception(e)


# =============================================================================
# 13. MEDICAL & RESEARCH DISCLAIMER
# =============================================================================
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Research & Educational Disclaimer:</strong><br>
    This system is developed strictly for research, academic, and educational demonstration purposes.
    Model predictions and visual explanations (Grad-CAM / LIME) should not be considered a definitive clinical diagnosis
    or a substitute for professional clinical judgment by a certified radiologist, neuro-oncologist, or medical practitioner.
</div>
""", unsafe_allow_html=True)
