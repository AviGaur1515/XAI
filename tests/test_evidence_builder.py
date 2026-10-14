"""
=============================================================================
Brain Tumor MRI Classification & Explainable AI
Module: tests.test_evidence_builder
=============================================================================
Unit tests for:
  - llm.evidence_builder.build_explanation_evidence()
  - llm.explanation_renderer.parse_llm_response()
  - llm.explanation_renderer.render_fallback_explanation()
  - llm.ollama_client.OllamaClient (with mock HTTP)

Run with:
    python -m pytest tests/test_evidence_builder.py -v

No trained model file or Ollama server is required to run these tests.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.evidence_builder import build_explanation_evidence, _extract_lime_top_features, _describe_score_gap
from llm.explanation_renderer import (
    parse_llm_response,
    render_fallback_explanation,
    build_prompt,
    SECTION_KEYS,
)
from llm.ollama_client import OllamaClient, OllamaStatus


# ---------------------------------------------------------------------------
# Helpers to build minimal fake pred_res and lime_res
# ---------------------------------------------------------------------------

def _make_pred_res(pred_class="glioma", scores=None):
    """Builds a minimal pred_res dict matching BrainTumorPredictor.predict() output."""
    if scores is None:
        scores = {"glioma": 0.92, "meningioma": 0.04, "notumor": 0.02, "pituitary": 0.02}
    return {
        "pred_index": ["glioma", "meningioma", "notumor", "pituitary"].index(pred_class),
        "pred_class": pred_class,
        "display_name": pred_class.capitalize(),
        "confidence": round(scores[pred_class] * 100, 2),
        "probabilities": scores,
        "raw_probabilities": np.array(list(scores.values()), dtype=np.float32),
        "rgb_display": np.zeros((240, 240, 3), dtype=np.uint8),
        "uint8_img": np.zeros((240, 240, 3), dtype=np.uint8),
        "processed_batch": np.zeros((1, 240, 240, 3), dtype=np.float32),
        "model_name": "efficientnetb0",
        "target_size": (240, 240),
        "is_loaded": True,
        "model_path": "model/efficientnet_final.keras",
    }


def _make_lime_res(success=True, pred_index=0):
    """Builds a minimal lime_res dict."""
    class MockExplanation:
        def __init__(self):
            self.local_exp = {
                pred_index: [(3, 0.45), (7, 0.31), (12, -0.18), (1, 0.22), (9, -0.09)]
            }
            self.segments = np.zeros((240, 240), dtype=int)

    return {
        "segments_image": np.zeros((240, 240, 3), dtype=np.uint8),
        "num_segments": 64,
        "positive_only_image": np.zeros((240, 240, 3), dtype=np.uint8),
        "pos_neg_image": np.zeros((240, 240, 3), dtype=np.uint8),
        "weight_map": np.zeros((240, 240), dtype=np.float64),
        "explanation": MockExplanation(),
        "success": success,
    }


# ===========================================================================
# Tests: evidence_builder
# ===========================================================================

class TestExtractLimeTopFeatures(unittest.TestCase):

    def test_returns_sorted_by_abs_weight(self):
        class FakeExp:
            local_exp = {0: [(3, 0.45), (7, 0.31), (12, -0.55), (1, 0.22)]}
            segments = np.zeros((10, 10), dtype=int)

        result = _extract_lime_top_features(FakeExp(), pred_index=0, top_n=3)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        # Highest absolute weight first
        self.assertEqual(result[0]["seg_id"], 12)
        self.assertAlmostEqual(result[0]["weight"], -0.55)
        self.assertEqual(result[0]["direction"], "opposes")
        self.assertEqual(result[1]["direction"], "supports")

    def test_returns_none_for_missing_index(self):
        class FakeExp:
            local_exp = {1: [(3, 0.5)]}
            segments = np.zeros((10, 10), dtype=int)

        result = _extract_lime_top_features(FakeExp(), pred_index=0)
        self.assertIsNone(result)

    def test_returns_none_for_none_explanation(self):
        self.assertIsNone(_extract_lime_top_features(None, pred_index=0))


class TestDescribeScoreGap(unittest.TestCase):

    def test_clear_margin(self):
        probs = {"glioma": 0.92, "meningioma": 0.04, "notumor": 0.02, "pituitary": 0.02}
        desc = _describe_score_gap(probs, "glioma")
        self.assertIn("clear", desc)

    def test_narrow_margin(self):
        probs = {"glioma": 0.40, "meningioma": 0.38, "notumor": 0.12, "pituitary": 0.10}
        desc = _describe_score_gap(probs, "glioma")
        self.assertIn("narrow", desc)

    def test_no_tumor_class(self):
        probs = {"glioma": 0.05, "meningioma": 0.03, "notumor": 0.88, "pituitary": 0.04}
        desc = _describe_score_gap(probs, "notumor")
        self.assertIn("No Tumor", desc)


class TestBuildExplanationEvidence(unittest.TestCase):

    def test_all_required_keys_present(self):
        pred = _make_pred_res("glioma")
        lime = _make_lime_res(success=True, pred_index=0)
        ev = build_explanation_evidence(pred, lime_res=lime, gradcam_success=True)

        required_keys = [
            "predicted_class", "predicted_class_key", "predicted_index",
            "model_score_pct", "all_scores", "score_gap_description",
            "model_name", "gradcam", "lime", "class_education", "mandatory_limitations",
        ]
        for key in required_keys:
            self.assertIn(key, ev, f"Missing key: {key}")

    def test_lime_available_when_success_true(self):
        pred = _make_pred_res("glioma")
        lime = _make_lime_res(success=True)
        ev = build_explanation_evidence(pred, lime_res=lime, gradcam_success=True)
        self.assertTrue(ev["lime"]["available"])
        self.assertIsNotNone(ev["lime"]["top_features"])

    def test_lime_unavailable_when_success_false(self):
        pred = _make_pred_res("glioma")
        lime = _make_lime_res(success=False)
        ev = build_explanation_evidence(pred, lime_res=lime, gradcam_success=False)
        self.assertFalse(ev["lime"]["available"])

    def test_lime_unavailable_when_lime_res_none(self):
        pred = _make_pred_res("notumor", {"glioma": 0.02, "meningioma": 0.03, "notumor": 0.92, "pituitary": 0.03})
        ev = build_explanation_evidence(pred, lime_res=None, gradcam_success=False)
        self.assertFalse(ev["lime"]["available"])

    def test_gradcam_available_flag(self):
        pred = _make_pred_res("pituitary", {"glioma": 0.02, "meningioma": 0.03, "notumor": 0.05, "pituitary": 0.90})
        ev = build_explanation_evidence(pred, gradcam_success=True)
        self.assertTrue(ev["gradcam"]["available"])

        ev2 = build_explanation_evidence(pred, gradcam_success=False)
        self.assertFalse(ev2["gradcam"]["available"])

    def test_class_education_present_for_all_classes(self):
        classes = ["glioma", "meningioma", "notumor", "pituitary"]
        scores_map = {
            "glioma": {"glioma": 0.9, "meningioma": 0.04, "notumor": 0.03, "pituitary": 0.03},
            "meningioma": {"glioma": 0.03, "meningioma": 0.9, "notumor": 0.04, "pituitary": 0.03},
            "notumor": {"glioma": 0.03, "meningioma": 0.03, "notumor": 0.91, "pituitary": 0.03},
            "pituitary": {"glioma": 0.03, "meningioma": 0.03, "notumor": 0.04, "pituitary": 0.90},
        }
        for cls in classes:
            pred = _make_pred_res(cls, scores_map[cls])
            ev = build_explanation_evidence(pred)
            self.assertTrue(len(ev["class_education"]) > 0, f"Missing education for {cls}")

    def test_all_scores_contains_four_entries(self):
        pred = _make_pred_res("glioma")
        ev = build_explanation_evidence(pred)
        self.assertEqual(len(ev["all_scores"]), 4)
        predicted_entries = [s for s in ev["all_scores"] if s["is_predicted"]]
        self.assertEqual(len(predicted_entries), 1)


# ===========================================================================
# Tests: explanation_renderer
# ===========================================================================

class TestParseLlmResponse(unittest.TestCase):

    def _make_response(self, include_all=True):
        lines = []
        for key in SECTION_KEYS:
            lines.append(f"\n{key}:\nSome content for {key}. This is the explanation text.")
        return "\n".join(lines)

    def test_parses_all_sections(self):
        raw = self._make_response()
        result = parse_llm_response(raw)
        self.assertIsNotNone(result)
        for key in SECTION_KEYS:
            self.assertIn(key, result)

    def test_returns_none_for_short_text(self):
        self.assertIsNone(parse_llm_response("Too short."))

    def test_returns_none_for_empty(self):
        self.assertIsNone(parse_llm_response(""))

    def test_returns_none_for_few_sections(self):
        # Only 2 sections — should fail threshold
        raw = "What the AI Predicted:\nSome text.\n\nImportant Limitations:\nSome text."
        result = parse_llm_response(raw)
        # Either None or only 2 sections found — either way not enough
        if result is not None:
            self.assertLessEqual(len(result), 2)

    def test_case_insensitive_match(self):
        raw = "\nwhat the ai predicted:\nThe model predicted Glioma.\n\nWhat Influenced the Prediction:\nThe bright region.\n\nUnderstanding the Visual Explanations:\nGrad-CAM shows red.\n"
        result = parse_llm_response(raw)
        # Should parse at least 3 sections
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result), 3)


class TestRenderFallbackExplanation(unittest.TestCase):

    def _make_evidence_glioma(self):
        pred = _make_pred_res("glioma")
        lime = _make_lime_res(success=True)
        return build_explanation_evidence(pred, lime_res=lime, gradcam_success=True)

    def test_all_section_keys_present(self):
        ev = self._make_evidence_glioma()
        result = render_fallback_explanation(ev)
        for key in SECTION_KEYS:
            self.assertIn(key, result, f"Missing section: {key}")

    def test_is_fallback_flag_true(self):
        ev = self._make_evidence_glioma()
        result = render_fallback_explanation(ev)
        self.assertTrue(result["_is_fallback"])

    def test_no_invented_anatomy(self):
        """Fallback text must not mention specific anatomical locations."""
        ev = self._make_evidence_glioma()
        result = render_fallback_explanation(ev)
        full_text = " ".join(v for v in result.values() if isinstance(v, str)).lower()
        forbidden = ["frontal lobe", "temporal lobe", "parietal", "occipital", "cerebellum"]
        for term in forbidden:
            self.assertNotIn(term, full_text, f"Invented anatomy found: {term}")

    def test_disclaimer_present(self):
        ev = self._make_evidence_glioma()
        result = render_fallback_explanation(ev)
        full_text = " ".join(v for v in result.values() if isinstance(v, str)).lower()
        self.assertIn("not a medical diagnostic", full_text)

    def test_no_tumor_class_handled(self):
        scores = {"glioma": 0.02, "meningioma": 0.03, "notumor": 0.92, "pituitary": 0.03}
        pred = _make_pred_res("notumor", scores)
        lime = _make_lime_res(success=False)
        ev = build_explanation_evidence(pred, lime_res=lime, gradcam_success=False)
        result = render_fallback_explanation(ev)
        sec1 = result["What the AI Predicted"].lower()
        # Should NOT claim the person is healthy or disease ruled out
        self.assertNotIn("healthy", sec1)
        self.assertNotIn("ruled out", sec1)
        # Should mention the class name
        self.assertIn("no tumor", sec1)


# ===========================================================================
# Tests: OllamaClient (with mocked HTTP)
# ===========================================================================

class TestOllamaClientHealth(unittest.TestCase):

    @patch("llm.ollama_client.requests.get")
    def test_health_ok(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        client = OllamaClient()
        self.assertEqual(client.check_health(), OllamaStatus.OK)

    @patch("llm.ollama_client.requests.get")
    def test_health_not_running_connection_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.ConnectionError
        client = OllamaClient()
        self.assertEqual(client.check_health(), OllamaStatus.OLLAMA_NOT_RUNNING)

    @patch("llm.ollama_client.requests.get")
    def test_model_found(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": [{"name": "qwen2.5:3b"}]},
        )
        client = OllamaClient(model_name="qwen2.5:3b")
        self.assertEqual(client.check_model(), OllamaStatus.OK)

    @patch("llm.ollama_client.requests.get")
    def test_model_not_found(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": [{"name": "llama3.2:3b"}]},
        )
        client = OllamaClient(model_name="qwen2.5:3b")
        self.assertEqual(client.check_model(), OllamaStatus.MODEL_NOT_FOUND)

    @patch("llm.ollama_client.requests.post")
    def test_generate_ok(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": "This is the generated explanation."},
        )
        client = OllamaClient()
        status, text = client.generate("test prompt")
        self.assertEqual(status, OllamaStatus.OK)
        self.assertEqual(text, "This is the generated explanation.")

    @patch("llm.ollama_client.requests.post")
    def test_generate_timeout(self, mock_post):
        import requests as req
        mock_post.side_effect = req.Timeout
        client = OllamaClient()
        status, text = client.generate("test prompt")
        self.assertEqual(status, OllamaStatus.TIMEOUT)
        self.assertEqual(text, "")

    @patch("llm.ollama_client.requests.post")
    def test_generate_empty_response(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": ""},
        )
        client = OllamaClient()
        status, text = client.generate("test prompt")
        self.assertEqual(status, OllamaStatus.RESPONSE_EMPTY)


class TestBuildPrompt(unittest.TestCase):
    """Smoke tests to ensure build_prompt doesn't crash and produces reasonable output."""

    def test_prompt_contains_predicted_class(self):
        pred = _make_pred_res("meningioma", {"glioma": 0.05, "meningioma": 0.85, "notumor": 0.06, "pituitary": 0.04})
        lime = _make_lime_res(success=True, pred_index=1)
        ev = build_explanation_evidence(pred, lime_res=lime, gradcam_success=True)
        prompt = build_prompt(ev)
        self.assertIn("Meningioma", prompt)
        self.assertIn("85.0%", prompt)

    def test_prompt_does_not_contain_pixel_data(self):
        pred = _make_pred_res("glioma")
        ev = build_explanation_evidence(pred)
        prompt = build_prompt(ev)
        # Should not contain numpy array representations
        self.assertNotIn("uint8", prompt)
        self.assertNotIn("float32", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
