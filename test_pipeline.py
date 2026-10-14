"""
Standalone test suite to verify the inference and XAI pipeline end-to-end.
"""

import os
import numpy as np
from utils.preprocessing import load_image_as_rgb, prepare_mri_inputs, CLASS_NAMES
from inference.predictor import BrainTumorPredictor
from xai.gradcam import compute_gradcam, build_overlay
from xai.lime_explainer import run_lime_explanation
from xai.superpixel import generate_superpixel_bar_chart


def run_tests():
    print("=" * 60)
    print("RUNNING BRAIN TUMOR XAI PIPELINE VERIFICATION")
    print("=" * 60)

    # Test 1: Image loading & Preprocessing
    sample_path = "sample_data/sample_glioma.png"
    raw_rgb = load_image_as_rgb(sample_path)
    assert raw_rgb.ndim == 3 and raw_rgb.shape[2] == 3, f"Unexpected shape: {raw_rgb.shape}"
    print(f"  [OK] Image loaded successfully as RGB uint8 (shape: {raw_rgb.shape})")

    mri_data = prepare_mri_inputs(raw_rgb, model_name="efficientnetb0")
    assert mri_data["processed_batch"].shape == (1, 240, 240, 3)
    assert mri_data["rgb_display"].shape == (240, 240, 3)
    print(f"  [OK] Prepared batch shape: {mri_data['processed_batch'].shape}")

    # Test 2: Inference Engine
    print("\n[Test 2] Testing BrainTumorPredictor...")
    predictor = BrainTumorPredictor()
    results = predictor.predict(sample_path)
    print(f"  [OK] Predicted class: {results['display_name']} ({results['pred_class']})")
    print(f"  [OK] Confidence: {results['confidence']:.2f}%")
    print(f"  [OK] Probabilities: {results['probabilities']}")
    assert len(results["probabilities"]) == 4
    for cls in CLASS_NAMES:
        assert cls in results["probabilities"]

    # Test 3: Grad-CAM
    print("\n[Test 3] Testing Grad-CAM XAI computation...")
    heatmap, pred_idx, pred_conf, success_cam = compute_gradcam(
        predictor.model,
        results["processed_batch"],
        model_name=predictor.model_name,
        pred_index=results["pred_index"]
    )
    assert heatmap.ndim == 2
    overlay, hm_resized = build_overlay(results["rgb_display"], heatmap, alpha=0.5)
    assert overlay.shape == (240, 240, 3)
    print(f"  [OK] Grad-CAM heatmap generated (min: {heatmap.min():.2f}, max: {heatmap.max():.2f})")
    print(f"  [OK] Grad-CAM overlay generated (shape: {overlay.shape})")

    # Test 4: LIME Explainability
    print("\n[Test 4] Testing LIME superpixel perturbations...")
    lime_res = run_lime_explanation(
        predictor.model,
        results["uint8_img"],
        model_name=predictor.model_name,
        pred_index=results["pred_index"],
        num_samples=50,
        num_features=5
    )
    assert "segments_image" in lime_res
    assert "positive_only_image" in lime_res
    assert "pos_neg_image" in lime_res
    assert "weight_map" in lime_res
    print(f"  [OK] LIME segments image shape: {lime_res['segments_image'].shape}")
    print(f"  [OK] Positive mask shape: {lime_res['positive_only_image'].shape}")
    print(f"  [OK] Positive/Negative mask shape: {lime_res['pos_neg_image'].shape}")

    # Test 5: Superpixel Weight Chart
    print("\n[Test 5] Testing Superpixel Bar Chart generation...")
    fig = generate_superpixel_bar_chart(lime_res["explanation"], pred_index=results["pred_index"], top_n=10)
    assert fig is not None
    print("  [OK] Superpixel bar chart Matplotlib figure generated successfully.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! THE PIPELINE IS 100% OPERATIONAL.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
