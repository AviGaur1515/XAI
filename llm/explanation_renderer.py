"""
=============================================================================
Brain Tumor MRI Classification & Explainable AI
Module: llm.explanation_renderer
=============================================================================
Builds the LLM prompt from structured evidence, calls OllamaClient,
validates/parses the response, and provides deterministic fallback text.

Fallback text is ALWAYS clearly labelled as non-LLM generated content.
"""

from __future__ import annotations

import re
import html
from typing import Optional

from llm.ollama_client import OllamaClient, OllamaStatus

# ---------------------------------------------------------------------------
# System prompt (sent as `system` field in Ollama payload)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You explain the outputs of an academic brain MRI classification and Explainable AI prototype in plain language.

Your rules:
- Do NOT diagnose patients or claim disease has been confirmed or ruled out.
- Explain ONLY the structured evidence supplied in the prompt. Do not add external knowledge about this specific image.
- The predicted class comes from an existing CNN and must NOT be changed.
- Treat model scores as softmax model outputs, NOT calibrated clinical probabilities.
- Grad-CAM shows model contribution/activation and is NOT a tumor segmentation. Do not name specific anatomical structures unless independently supplied and verified.
- LIME describes how perturbing image regions affected the model output according to the supplied weights and legend. Use real feature weights from the evidence; do not invent counts or rankings.
- Never infer exact anatomy, tumor boundaries, grade, malignancy, prognosis or treatment from this output.
- Separate general educational information (clearly labelled as general) from claims about the uploaded image.
- If evidence is insufficient, say so explicitly.
- Use calm, clear, non-alarming language and explain limitations.
- Recommend professional medical assessment for real health concerns.
- Do NOT claim the system has diagnosed or ruled out disease.
- Responses must be in plain English; no markdown headers inside sections.
- Keep the total response under 600 words.
"""

# ---------------------------------------------------------------------------
# Expected section headers (used for parsing)
# ---------------------------------------------------------------------------

SECTION_KEYS = [
    "What the AI Predicted",
    "What Influenced the Prediction",
    "Understanding the Visual Explanations",
    "General Information About This Class",
    "Important Limitations",
    "Suggested Next Step",
]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(evidence: dict) -> str:
    """
    Converts the structured evidence dict into a plain-text prompt.
    No raw images, no patient data.
    """
    e = evidence
    pred = e["predicted_class"]
    score = e["model_score_pct"]

    # All class scores
    score_lines = "\n".join(
        f"  - {s['class_name']}: {s['score_pct']}%{'  <-- predicted' if s['is_predicted'] else ''}"
        for s in e["all_scores"]
    )

    # LIME features section
    lime = e["lime"]
    if lime["available"] and lime.get("top_features"):
        feat_lines = "\n".join(
            f"  Rank {f['rank']}: superpixel S{f['seg_id']}, weight={f['weight']:.4f}, {f['direction']} predicted class"
            for f in lime["top_features"]
        )
        lime_section = (
            f"LIME is available.\n"
            f"Number of superpixel segments: {lime.get('num_segments', 'unknown')}\n"
            f"Sign convention: {lime['sign_convention']}\n"
            f"Color legend: {lime['color_legend']}\n"
            f"Top features by absolute weight:\n{feat_lines}\n"
            f"Note: {lime['note']}"
        )
    else:
        lime_section = f"LIME data: {lime.get('note', 'Not available.')}"

    # Grad-CAM section
    gcam = e["gradcam"]
    if gcam["available"]:
        gcam_section = (
            f"Grad-CAM is available.\n"
            f"Colormap: {gcam['colormap']}\n"
            f"Meaning: {gcam['color_meaning']}\n"
            f"Note: {gcam['note']}"
        )
    else:
        gcam_section = f"Grad-CAM data: {gcam.get('note', 'Not available.')}"

    prompt = f"""Below is structured evidence from an academic brain MRI classification system.
Use ONLY this evidence. Do NOT add any information not contained here.

=== PREDICTION ===
Predicted class: {pred}
Model score for predicted class: {score}%
Score gap description: {e['score_gap_description']}
All class scores (softmax model outputs, not clinical probabilities):
{score_lines}

=== GRAD-CAM EVIDENCE ===
{gcam_section}

=== LIME EVIDENCE ===
{lime_section}

=== GENERAL CLASS INFORMATION (applies broadly to this class, NOT specifically to this image) ===
{e['class_education']}

=== MANDATORY LIMITATIONS ===
{e['mandatory_limitations']}

Please write a patient-friendly explanation organised into exactly these six sections.
Start each section with the exact label on its own line, followed by the content:

1. What the AI Predicted
2. What Influenced the Prediction
3. Understanding the Visual Explanations
4. General Information About This Class
5. Important Limitations
6. Suggested Next Step
"""
    return prompt


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_llm_response(raw_text: str) -> Optional[dict[str, str]]:
    """
    Attempts to split the LLM response into the six expected sections.
    Returns a dict {section_label: content} or None if parsing fails.
    """
    if not raw_text or len(raw_text.strip()) < 50:
        return None

    sections: dict[str, str] = {}
    # Build a pattern that matches any of the numbered section headers
    pattern = r"(?:^|\n)\s*(?:\d+\.\s*)?" + r"(" + "|".join(
        re.escape(k) for k in SECTION_KEYS
    ) + r")\s*[\:\n]"

    parts = re.split(pattern, raw_text, flags=re.IGNORECASE)

    # parts = [pre-text, key1, content1, key2, content2, ...]
    if len(parts) < 3:
        return None

    i = 1
    while i < len(parts) - 1:
        key_found = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        # Match to canonical key (case-insensitive)
        for canonical in SECTION_KEYS:
            if canonical.lower() in key_found.lower():
                sections[canonical] = content
                break
        i += 2

    # Require at least 3 sections to accept the parse
    if len(sections) < 3:
        return None

    return sections


def sanitize_text(text: str) -> str:
    """Escapes HTML special characters for safe Streamlit markdown rendering."""
    # Allow common markdown but escape angle brackets
    text = html.escape(text, quote=False)
    # Re-allow markdown bold/italic markers that html.escape doesn't touch
    return text


# ---------------------------------------------------------------------------
# Fallback explanation (deterministic, evidence-based)
# ---------------------------------------------------------------------------

def render_fallback_explanation(evidence: dict) -> dict[str, str]:
    """
    Produces a deterministic, evidence-based fallback explanation when Ollama
    is unavailable or generation fails.

    ALWAYS clearly labelled as non-LLM content.
    """
    pred = evidence["predicted_class"]
    score = evidence["model_score_pct"]
    gap_desc = evidence["score_gap_description"]
    lime = evidence["lime"]
    gcam = evidence["gradcam"]
    education = evidence["class_education"]
    limits = evidence["mandatory_limitations"]

    # What was predicted
    sec1 = (
        f"The AI model assigned this MRI scan to the class **{pred}** "
        f"with a model score (softmax output) of **{score}%**. "
        f"{gap_desc} "
        f"Note: These are model outputs, not clinically validated confidence values."
    )

    # Influences
    lime_influence = ""
    if lime["available"] and lime.get("top_features"):
        top = lime["top_features"][0]
        direction = top["direction"]
        lime_influence = (
            f" LIME analysis identified superpixel S{top['seg_id']} as the most influential region "
            f"(weight: {top['weight']:.4f}), which {direction} the predicted class."
        )
    else:
        lime_influence = " LIME region data is not available for this image."

    sec2 = (
        f"The model's decision was driven by patterns it detected in the image pixels. "
        f"The specific features that contributed most are visualised using Grad-CAM and LIME below.{lime_influence}"
    )

    # Visual explanations
    if gcam["available"]:
        gcam_txt = (
            f"Grad-CAM: {gcam['color_meaning']} "
            f"Colormap used: {gcam['colormap']}. {gcam['note']}"
        )
    else:
        gcam_txt = "Grad-CAM data was not available for this image."

    if lime["available"]:
        lime_txt = (
            f"LIME: {lime.get('color_legend', '')} "
            f"Positive weight means the superpixel supported the prediction; negative means it opposed it. "
            f"{lime.get('note', '')}"
        )
    else:
        lime_txt = f"LIME: {lime.get('note', 'Not available.')}"

    sec3 = f"{gcam_txt}\n\n{lime_txt}"

    # Education
    sec4 = education if education else f"General information for class '{pred}' is not available."

    # Limitations
    sec5 = limits

    # Next step
    sec6 = (
        "If this scan relates to a real health concern, please consult a qualified radiologist, "
        "neurologist or neurosurgeon who can review the original imaging study and your full clinical history. "
        "Do not rely on this AI prototype for any medical decision."
    )

    return {
        "What the AI Predicted": sec1,
        "What Influenced the Prediction": sec2,
        "Understanding the Visual Explanations": sec3,
        "General Information About This Class": sec4,
        "Important Limitations": sec5,
        "Suggested Next Step": sec6,
        "_is_fallback": True,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_explanation(
    evidence: dict,
    client: OllamaClient,
) -> tuple[dict[str, str], OllamaStatus, bool]:
    """
    Attempts to generate an LLM explanation via Ollama.
    Falls back to deterministic text on any failure.

    Returns:
        (sections_dict, ollama_status, is_fallback)
        sections_dict keys: SECTION_KEYS + optionally "_is_fallback"
        is_fallback: True when the returned text is deterministic fallback.
    """
    prompt = build_prompt(evidence)
    status, raw_text = client.generate(prompt, system_prompt=SYSTEM_PROMPT)

    if status == OllamaStatus.OK:
        parsed = parse_llm_response(raw_text)
        if parsed is not None:
            parsed["_is_fallback"] = False
            return parsed, status, False
        # Parsing failed — use raw text in a single section + fallback for rest
        fallback = render_fallback_explanation(evidence)
        fallback["_is_fallback"] = True
        return fallback, OllamaStatus.RESPONSE_EMPTY, True

    # Any other status: use fallback
    fallback = render_fallback_explanation(evidence)
    return fallback, status, True
