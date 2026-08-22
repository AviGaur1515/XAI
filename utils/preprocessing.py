"""
=============================================================================
Brain Tumor MRI Classification & Explainable AI
Module: utils.preprocessing
=============================================================================
Handles image loading, color formatting, resizing, and model-specific 
preprocessing pipelines derived from the validated Kaggle notebook.
"""

import io
import os
import cv2
import numpy as np
from PIL import Image

# Constants matching final-updated.ipynb
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary"
}

MODEL_CONFIGS = {
    "efficientnetb0": {
        "img_size": (240, 240),
        "target_layer": "top_conv"
    },
    "resnet50": {
        "img_size": (224, 224),
        "target_layer": "conv5_block3_out"
    },
    "vgg16": {
        "img_size": (224, 224),
        "target_layer": "block5_conv3"
    }
}


def load_image_as_rgb(image_input):
    """
    Converts various input formats (file path, bytes, BytesIO, UploadedFile, PIL Image, np.ndarray)
    into a standard RGB numpy array (uint8).
    
    Args:
        image_input: File path (str), bytes/BytesIO, UploadedFile, PIL.Image, or np.ndarray.
        
    Returns:
        np.ndarray: RGB image with dtype uint8 of shape (H, W, 3).
    """
    if image_input is None:
        raise ValueError("Image input is None.")

    # Streamlit UploadedFile or object with .getvalue()
    if hasattr(image_input, "getvalue"):
        data = image_input.getvalue()
        return load_image_as_rgb(data)

    # File path string
    if isinstance(image_input, (str, os.PathLike)):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image path does not exist: {image_input}")
        pil_img = Image.open(image_input).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)

    # Raw bytes
    if isinstance(image_input, bytes):
        if len(image_input) == 0:
            raise ValueError("Provided image byte stream is empty (0 bytes).")
        try:
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
            return np.array(pil_img, dtype=np.uint8)
        except Exception as e:
            # Fallback decoding via OpenCV
            nparr = np.frombuffer(image_input, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            raise ValueError(f"Could not decode image from byte stream: {e}")

    # File-like object with .read()
    if hasattr(image_input, "read"):
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        content = image_input.read()
        if hasattr(image_input, "seek"):
            image_input.seek(0)
        if not content:
            raise ValueError("File stream is empty.")
        return load_image_as_rgb(content)

    # PIL Image
    if isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"), dtype=np.uint8)

    # NumPy Array
    if isinstance(image_input, np.ndarray):
        arr = image_input.copy()
        if arr.ndim == 2:  # Grayscale
            return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3:
            if arr.shape[2] == 4:  # RGBA
                return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2RGB)
            elif arr.shape[2] == 3:
                return arr.astype(np.uint8)
        raise ValueError(f"Unsupported numpy image shape: {arr.shape}")

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def preprocess_for_model(img_rgb, model_name="efficientnetb0"):
    """
    Applies the specific normalization function required by the backbone model.
    """
    arr = img_rgb.astype(np.float32).copy()
    m_name = model_name.lower().replace("-", "").replace("_", "")

    try:
        if "efficientnet" in m_name:
            from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
            return eff_pre(arr)
        elif "resnet" in m_name:
            from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
            return res_pre(arr)
        elif "vgg" in m_name:
            from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
            return vgg_pre(arr)
        else:
            return arr / 255.0
    except ImportError:
        if "efficientnet" in m_name:
            return arr
        elif "resnet" in m_name or "vgg" in m_name:
            arr_bgr = arr[..., ::-1].copy()
            arr_bgr[..., 0] -= 103.939
            arr_bgr[..., 1] -= 116.779
            arr_bgr[..., 2] -= 123.68
            return arr_bgr
        else:
            return arr / 255.0


def prepare_mri_inputs(image_input, model_name="efficientnetb0", target_size=None):
    """
    Complete pipeline to load, resize, and prepare all representations needed
    for inference, visualization, and XAI explainability.
    """
    raw_rgb = load_image_as_rgb(image_input)
    orig_h, orig_w = raw_rgb.shape[:2]

    if target_size is None:
        m_key = "efficientnetb0"
        for k in MODEL_CONFIGS:
            if k in model_name.lower():
                m_key = k
                break
        target_size = MODEL_CONFIGS[m_key]["img_size"]

    resized_rgb = cv2.resize(raw_rgb, target_size, interpolation=cv2.INTER_AREA)
    processed = preprocess_for_model(resized_rgb, model_name=model_name)
    processed_batch = np.expand_dims(processed, axis=0).astype(np.float32)

    return {
        "processed_batch": processed_batch,
        "rgb_display": resized_rgb.astype(np.uint8),
        "uint8_img": resized_rgb.astype(np.uint8),
        "orig_size": (orig_w, orig_h),
        "target_size": target_size
    }
