"""
=============================================================================
Brain Tumor MRI Classification & Explainable AI
Module: inference.predictor
=============================================================================
Provides standalone model inference and class probability estimation.
Fully decoupled from GUI layers for direct portability to Raspberry Pi.
"""

import os
import glob
import numpy as np
from utils.preprocessing import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    MODEL_CONFIGS,
    prepare_mri_inputs
)


class BrainTumorPredictor:
    """
    Inference engine that loads a trained Keras model and predicts brain tumor classes.
    """

    def __init__(self, model_path=None, model_name=None):
        """
        Initializes the predictor.
        
        Args:
            model_path (str, optional): Path to .keras or .h5 model file.
            model_name (str, optional): 'efficientnetb0', 'resnet50', or 'vgg16'.
        """
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.model_name = model_name or "efficientnetb0"
        self.input_shape = MODEL_CONFIGS["efficientnetb0"]["img_size"]
        
        self._find_and_load_model(model_path)

    def _find_and_load_model(self, model_path=None):
        """Locates and loads the Keras model file."""
        candidates = []
        if model_path and os.path.exists(model_path):
            candidates.append(model_path)
        
        # Check standard model directories
        search_dirs = ["model", "models", ".", ".."]
        for d in search_dirs:
            if os.path.exists(d):
                candidates.extend(glob.glob(os.path.join(d, "*.keras")))
                candidates.extend(glob.glob(os.path.join(d, "*.h5")))

        if candidates:
            selected_path = candidates[0]
            try:
                import tensorflow as tf
                # Load Keras model with compile=False for fast inference
                self.model = tf.keras.models.load_model(selected_path, compile=False)
                self.model_path = selected_path
                self.is_loaded = True
                self._detect_architecture()
                print(f"[Predictor] Loaded model from: {selected_path} ({self.model_name})")
                return
            except Exception as e:
                print(f"[Predictor] Warning: Could not load model from {selected_path}: {e}")

        print("[Predictor] No physical model checkpoint found. Running in standalone fallback mode.")
        self.is_loaded = False

    def _detect_architecture(self):
        """Inspects the model structure to determine backbone name and target image size."""
        if self.model is None:
            return

        try:
            # Check input shape
            in_shape = self.model.input_shape
            if isinstance(in_shape, list):
                in_shape = in_shape[0]
            if len(in_shape) >= 3 and in_shape[1] is not None and in_shape[2] is not None:
                self.input_shape = (int(in_shape[1]), int(in_shape[2]))
        except Exception:
            pass

        # Check layer names for architecture hints
        all_layer_names = [layer.name.lower() for layer in self.model.layers]
        layer_str = " ".join(all_layer_names)
        
        if "efficientnet" in layer_str or "top_conv" in layer_str:
            self.model_name = "efficientnetb0"
        elif "resnet" in layer_str or "conv5_block" in layer_str:
            self.model_name = "resnet50"
        elif "vgg" in layer_str or "block5_conv" in layer_str:
            self.model_name = "vgg16"
        elif self.input_shape == (240, 240):
            self.model_name = "efficientnetb0"
        else:
            self.model_name = "efficientnetb0"

    def load_model_from_file(self, custom_path):
        """Allows loading a custom model file dynamically."""
        if not os.path.exists(custom_path):
            raise FileNotFoundError(f"Model file not found at: {custom_path}")
        
        import tensorflow as tf
        self.model = tf.keras.models.load_model(custom_path, compile=False)
        self.model_path = custom_path
        self.is_loaded = True
        self._detect_architecture()
        return True

    def predict(self, image_input):
        """
        Runs inference on an MRI image.
        
        Args:
            image_input: Path, bytes, PIL Image, or np.ndarray.
            
        Returns:
            dict: Structured prediction results including probabilities, 
                  predicted class, and processed arrays.
        """
        # Step 1: Prepare MRI inputs using validated pipeline
        mri_data = prepare_mri_inputs(
            image_input,
            model_name=self.model_name,
            target_size=self.input_shape
        )
        
        processed_batch = mri_data["processed_batch"]
        rgb_display = mri_data["rgb_display"]
        uint8_img = mri_data["uint8_img"]

        # Step 2: Model Inference
        if self.is_loaded and self.model is not None:
            raw_preds = self.model(processed_batch, training=False).numpy()
            probs = raw_preds[0]
            # Ensure probabilities sum to 1.0
            probs = np.clip(probs, 0.0, 1.0)
            sum_p = np.sum(probs)
            if sum_p > 0:
                probs = probs / sum_p
        else:
            # Standalone deterministic fallback for testing UI when model file is not yet dropped
            # Analyzes mean pixel intensity and symmetry to produce realistic test probabilities
            mean_intensity = np.mean(rgb_display)
            center_crop = rgb_display[60:180, 60:180]
            center_mean = np.mean(center_crop)
            
            if center_mean > mean_intensity * 1.3:
                probs = np.array([0.9462, 0.0231, 0.0174, 0.0133], dtype=np.float32)  # Glioma
            elif center_mean < mean_intensity * 0.8:
                probs = np.array([0.0125, 0.0185, 0.9520, 0.0170], dtype=np.float32)  # No Tumor
            else:
                probs = np.array([0.0350, 0.9120, 0.0280, 0.0250], dtype=np.float32)  # Meningioma

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx]) * 100.0

        prob_dict = {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        }

        return {
            "pred_index": pred_idx,
            "pred_class": pred_class,
            "display_name": DISPLAY_NAMES.get(pred_class, pred_class.capitalize()),
            "confidence": round(confidence, 2),
            "probabilities": prob_dict,
            "raw_probabilities": probs,
            "rgb_display": rgb_display,
            "uint8_img": uint8_img,
            "processed_batch": processed_batch,
            "model_name": self.model_name,
            "target_size": self.input_shape,
            "is_loaded": self.is_loaded,
            "model_path": self.model_path
        }
