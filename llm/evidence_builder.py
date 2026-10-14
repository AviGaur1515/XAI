"""
=============================================================================
Brain Tumor MRI Classification & Explainable AI
Module: llm.evidence_builder
=============================================================================
Builds a structured, plain-text evidence context from verified prediction
and XAI outputs.  Only factual, computed values are included.

Rules:
- Never include raw image pixels or patient-identifying data.
- Never invent anatomical locations, tumor boundaries or clinical findings.
- Mark Grad-CAM / LIME data as unavailable when success flags are False.
- All class names use the canonical CLASS_NAMES list from preprocessing.
- Sign convention for LIME: positive weight = supports predicted class.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional

from utils.preprocessing import CLASS_NAMES, DISPLAY_NAMES

# ---------------------------------------------------------------------------
# Educational background text per class (kept brief, class-level only)
# These are general facts and do NOT apply to any individual image.
# ---------------------------------------------------------------------------
_CLASS_EDUCATION: dict[str, str] = {
    "glioma": (
        "Gliomas are tumors that arise from glial cells — the supportive tissue of the brain. "
        "They account for the majority of malignant primary brain tumors and vary widely in grade "
        "and behaviour. Detection and management require specialist neuroimaging and neuro-oncology review."
    ),
    "meningioma": (
        "Meningiomas originate from the meninges (the protective membranes surrounding the brain and spinal cord). "
        "They are the most common primary brain tumor and are most often benign and slow-growing, "
        "though some are atypical or malignant. Management depends on size, location, symptoms and grade."
    ),
    "notumor": (
        "This class represents brain MRI scans where the model did not detect imaging patterns "
        "associated with the three tumor categories it was trained on. "
        "This does not rule out all neurological conditions or subtle findings that require specialist review."
    ),
    "pituitary": (
        "Pituitary tumors (adenomas) arise from the pituitary gland at the base of the brain. "
        "They are usually benign and may affect hormone levels. "
        "Diagnosis and treatment planning require endocrine, ophthalmological and neurosurgical assessment."
    ),
}

_MANDATORY_LIMITATIONS = (
    "This is an academic research prototype and is NOT a medical diagnostic device. "
    "The CNN was trained on a specific public dataset and may not generalise to all MRI scanners, "
    "protocols or patient populations. Model scores are softmax outputs, not calibrated clinical probabilities. "
    "Grad-CAM and LIME explain model behaviour; they are not tumor segmentations or validated biomarkers. "
    "Always consult a qualified radiologist or neurologist for any real health concern."
)


# ---------------------------------------------------------------------------
# LIME helper: extract top weighted superpixels
# ---------------------------------------------------------------------------

def _extract_lime_top_features(
    explanation: Any,
    pred_index: int,
    top_n: int = 5,
) -> Optional[list[dict]]:
    """
    Extracts the top-n LIME superpixel weights for `pred_index`.

    Returns a list of dicts: [{rank, seg_id, weight, direction}, ...]
    or None if data is unavailable.
    """
    if explanation is None:
        return None
    local_exp = getattr(explanation, "local_exp", None)
    if local_exp is None:
        return None
    pairs = local_exp.get(pred_index, [])
    if not pairs:
        return None

    sorted_pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    result = []
    for rank, (seg_id, weight) in enumerate(sorted_pairs, start=1):
        result.append({
            "rank": rank,
            "seg_id": int(seg_id),
            "weight": round(float(weight), 5),
            "direction": "supports" if weight > 0 else "opposes",
        })
    return result


def _describe_score_gap(probs: dict[str, float], pred_class: str) -> str:
    """
    Returns a plain-text description of the margin between the top score
    and the second-highest score — only from actual computed values.
    """
    top_score = probs[pred_class]
    others = {k: v for k, v in probs.items() if k != pred_class}
    second_class = max(others, key=lambda k: others[k])
    second_score = others[second_class]
    gap = top_score - second_score

    top_pct = round(top_score * 100, 1)
    second_pct = round(second_score * 100, 1)
    second_name = DISPLAY_NAMES.get(second_class, second_class)

    if gap > 0.6:
        margin_desc = "clear"
    elif gap > 0.25:
        margin_desc = "moderate"
    else:
        margin_desc = "narrow"

    return (
        f"The model assigned a {margin_desc} margin between '{DISPLAY_NAMES.get(pred_class, pred_class)}' "
        f"(score: {top_pct}%) and the next highest class '{second_name}' (score: {second_pct}%)."
    )


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def build_explanation_evidence(
    pred_res: dict,
    lime_res: Optional[dict] = None,
    gradcam_success: bool = False,
    lime_features: int = 5,
) -> dict:
    """
    Assembles a structured evidence dictionary from verified inference and
    XAI outputs.  This dict is serialised to a structured prompt for the LLM.

    Args:
        pred_res:         Output dict from BrainTumorPredictor.predict().
        lime_res:         Output dict from run_lime_explanation(), or None.
        gradcam_success:  Whether compute_gradcam() returned success=True.
        lime_features:    Number of top LIME features used in the UI.

    Returns:
        dict with keys consumed by explanation_renderer.build_prompt().
    """
    pred_class = pred_res["pred_class"]
    pred_index = pred_res["pred_index"]
    probs = pred_res["probabilities"]         # {class_name: float 0–1}
    model_score_pct = round(float(probs[pred_class]) * 100, 2)

    # All class scores rounded for display
    all_scores: list[dict] = [
        {
            "class_name": DISPLAY_NAMES.get(cn, cn),
            "score_pct": round(float(probs[cn]) * 100, 2),
            "is_predicted": (cn == pred_class),
        }
        for cn in CLASS_NAMES
    ]

    # Score gap description (evidence-based only)
    score_gap_desc = _describe_score_gap(probs, pred_class)

    # ---- Grad-CAM evidence ------------------------------------------------
    gradcam_evidence: dict = {
        "available": gradcam_success,
        "colormap": "JET (blue → cyan → green → yellow → red)",
        "color_meaning": (
            "Red and yellow areas indicate where the model's attention was highest "
            "for the predicted class. Blue areas indicate low model attention."
        ),
        "note": (
            "The heatmap shows where model gradients were strongest. "
            "It is NOT a tumor segmentation and should not be interpreted as exact tumor boundaries."
        ) if gradcam_success else "Grad-CAM data was not successfully computed for this image.",
    }

    # ---- LIME evidence ----------------------------------------------------
    lime_available = (
        lime_res is not None
        and lime_res.get("success", False)
        and lime_res.get("explanation") is not None
    )

    lime_evidence: dict = {"available": lime_available}

    if lime_available:
        top_features = _extract_lime_top_features(
            lime_res["explanation"], pred_index, top_n=lime_features
        )
        lime_evidence.update({
            "num_segments": lime_res.get("num_segments", "unknown"),
            "sign_convention": (
                "Positive weight → superpixel supports the predicted class. "
                "Negative weight → superpixel opposes the predicted class."
            ),
            "color_legend": "Green boundaries = supporting superpixels; Red boundaries = opposing superpixels.",
            "top_features": top_features,
            "note": (
                "LIME randomly perturbs (masks) superpixel regions and trains a local linear model "
                "to approximate the CNN's behaviour in the neighbourhood of this image."
            ),
        })
    else:
        lime_evidence["note"] = (
            "LIME explanation data is not available or was computed in fallback mode for this image."
        )

    # ---- Educational content (class-level, clearly labelled) -------------
    education = _CLASS_EDUCATION.get(pred_class, "")

    return {
        "predicted_class": DISPLAY_NAMES.get(pred_class, pred_class),
        "predicted_class_key": pred_class,
        "predicted_index": pred_index,
        "model_score_pct": model_score_pct,
        "all_scores": all_scores,
        "score_gap_description": score_gap_desc,
        "model_name": pred_res.get("model_name", "unknown").upper(),
        "gradcam": gradcam_evidence,
        "lime": lime_evidence,
        "class_education": education,
        "mandatory_limitations": _MANDATORY_LIMITATIONS,
    }
