"""
=============================================================================
Brain Tumor MRI Classification & Explainable AI
Module: xai.gradcam
=============================================================================
Gradient-Weighted Class Activation Mapping (Grad-CAM) implementation
fully compatible with Keras 3 and TensorFlow 2.x functional models.
"""

import cv2
import numpy as np


def find_base_model(model):
    """
    Finds the base backbone model (e.g. EfficientNetB0, ResNet50, VGG16)
    if wrapped inside a custom classification model.
    """
    if model is None:
        return None
    for layer in model.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 5:
            return layer
    return model


def find_gradcam_target_layer(base_model, model_name="efficientnetb0"):
    """
    Identifies the target convolutional layer name for Grad-CAM.
    Matches layer names used in the Kaggle notebook:
      - EfficientNet: 'top_conv'
      - ResNet50: 'conv5_block3_out', 'conv5_block3_add', or 'add_15'
      - VGG16: 'block5_conv3'
    """
    if base_model is None:
        return "top_conv"

    layer_names = [l.name for l in base_model.layers]
    m_lower = model_name.lower()

    if "resnet" in m_lower:
        for name in ["conv5_block3_out", "conv5_block3_add", "add_15", "conv5_block3_3_conv"]:
            if name in layer_names:
                return name
    elif "efficientnet" in m_lower:
        if "top_conv" in layer_names:
            return "top_conv"
    elif "vgg" in m_lower:
        if "block5_conv3" in layer_names:
            return "block5_conv3"

    # Dynamic fallback: search backwards for Conv2D / DepthwiseConv2D
    import tensorflow as tf
    for layer in reversed(base_model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Add)):
            return layer.name

    return layer_names[-1]


def compute_gradcam(model, img_array, model_name="efficientnetb0", target_layer_name=None, pred_index=None):
    """
    Computes Grad-CAM activation heatmap for a target class using GradientTape.
    Compatible with Keras 3 and nested functional architectures.
    
    Args:
        model: Loaded Keras model.
        img_array: 4D numpy array (1, H, W, 3).
        model_name: Backbone name.
        target_layer_name: Optional explicit layer name.
        pred_index: Target class index. If None, uses top predicted class.
        
    Returns:
        tuple: (heatmap: 2D np.ndarray [0, 1], pred_index: int, pred_conf: float, success: bool)
    """
    if model is None:
        # Fallback synthetic circular heatmap for standalone demo
        h, w = img_array.shape[1:3]
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        mock_hm = np.exp(-r2 / (2 * (h / 4) ** 2))
        mock_hm = (mock_hm - mock_hm.min()) / (mock_hm.max() - mock_hm.min() + 1e-8)
        return mock_hm, (pred_index if pred_index is not None else 0), 94.6, False

    import tensorflow as tf

    base_model = find_base_model(model)
    if target_layer_name is None:
        target_layer_name = find_gradcam_target_layer(base_model, model_name)

    try:
        # 1. Locate target convolutional layer inside base_model
        top_conv_idx = None
        for idx, l in enumerate(base_model.layers):
            if l.name == target_layer_name:
                top_conv_idx = idx
                break

        if top_conv_idx is not None:
            target_layer = base_model.layers[top_conv_idx]
            
            # Sub-model mapping base_model.inputs -> target_layer.output
            conv_model = tf.keras.Model(inputs=base_model.inputs, outputs=target_layer.output)
            
            # Remaining base_model layers after target_layer
            base_remaining_layers = base_model.layers[top_conv_idx + 1:]
            
            # Classification head layers
            head_layers = []
            if base_model != model:
                for layer in model.layers:
                    if not isinstance(layer, tf.keras.layers.InputLayer) and layer.name != base_model.name:
                        head_layers.append(layer)

            img_tensor = tf.cast(img_array, tf.float32)

            with tf.GradientTape() as tape:
                # Forward pass to target conv layer
                conv_outputs = conv_model(img_tensor, training=False)
                tape.watch(conv_outputs)

                # Pass through remaining base_model layers
                feat = conv_outputs
                for layer in base_remaining_layers:
                    feat = layer(feat, training=False)

                # Pass through classifier head layers
                for layer in head_layers:
                    feat = layer(feat, training=False)

                predictions = feat
                if pred_index is None:
                    pred_index = int(tf.argmax(predictions[0]).numpy())
                class_score = predictions[:, pred_index]

            # Compute gradients of class score w.r.t. conv outputs
            grads = tape.gradient(class_score, conv_outputs)

            if grads is not None:
                weights = tf.reduce_mean(grads, axis=(0, 1, 2))
                cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
                cam = tf.nn.relu(cam)
                cam_np = cam.numpy()

                mn, mx = cam_np.min(), cam_np.max()
                if (mx - mn) > 1e-8:
                    cam_np = (cam_np - mn) / (mx - mn)
                else:
                    cam_np = np.zeros_like(cam_np)

                pred_conf = float(predictions[0][pred_index].numpy()) * 100.0
                print(f"[Grad-CAM Success] Layer: {target_layer_name} | Max activation: {cam_np.max():.4f}")
                return cam_np, int(pred_index), pred_conf, True

    except Exception as e:
        print(f"[Grad-CAM Exception] {e}")

    # Fallback if gradient tape fails
    try:
        predictions = model(img_array, training=False).numpy()
        if pred_index is None:
            pred_index = int(np.argmax(predictions[0]))
        pred_conf = float(predictions[0][pred_index]) * 100.0
        return np.zeros((7, 7), dtype=np.float32), int(pred_index), pred_conf, False
    except Exception:
        return np.zeros((7, 7), dtype=np.float32), 0, 0.0, False


def build_overlay(orig_rgb, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Blends the Grad-CAM heatmap with the original RGB MRI image.
    Matches Kaggle build_overlay function.
    
    Args:
        orig_rgb (np.ndarray): Original image (H, W, 3) uint8.
        heatmap (np.ndarray): 2D float heatmap in range [0, 1].
        alpha (float): Heatmap opacity (0.0 to 1.0).
        colormap: OpenCV colormap (default cv2.COLORMAP_JET).
        
    Returns:
        tuple: (overlay: np.ndarray uint8, hm_resized: np.ndarray float32)
    """
    H, W = orig_rgb.shape[:2]
    hm_up = cv2.resize(heatmap.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    hm_up = np.clip(hm_up, 0.0, 1.0)
    hm_uint8 = np.uint8(255 * hm_up)
    
    # Apply colormap and convert BGR -> RGB
    hm_jet = cv2.applyColorMap(hm_uint8, colormap)
    hm_jet = cv2.cvtColor(hm_jet, cv2.COLOR_BGR2RGB)
    
    # Alpha blend
    overlay = np.uint8(hm_jet * alpha + orig_rgb * (1.0 - alpha))
    return overlay, hm_up
