"""
explainer.py — AI-Powered Anomaly Explanations

Translates raw statistical anomalies into short, plain-English insights that
non-technical stakeholders can immediately understand.

Two modes:
  1. **LLM mode** — Uses Google Gemini (free tier) to generate contextual,
     natural-language explanations.
  2. **Rule-based fallback** — If no API key is configured or the LLM call
     fails, a deterministic template produces a reasonable explanation.

Security:  The API key is loaded exclusively from the GEMINI_API_KEY
environment variable (via python-dotenv). It is never hard-coded.
"""

import os

import google.generativeai as genai
from dotenv import load_dotenv

# Load .env file located at the project root
load_dotenv()


class AnomalyExplainer:
    """
    Generates human-readable explanations for detected anomalies.
    Gracefully degrades to rule-based output when no LLM is available.
    """

    def __init__(self):
        """Initialise the explainer, configuring Gemini if a valid key exists."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.use_llm = False

        if self.api_key and self.api_key != "your_api_key_here":
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.use_llm = True

    def generate_explanation(
        self,
        date: str,
        actual_value: float,
        expected_value: float,
        z_score: float,
    ) -> str:
        """
        Produce a concise anomaly explanation.

        Args:
            date:           Date string of the anomalous observation.
            actual_value:   The recorded metric value.
            expected_value: The rolling-average expectation.
            z_score:        Statistical severity indicator.

        Returns:
            A 1–2 sentence explanation prefixed with its source
            ([AI Explanation] or [Rule-Based Explanation]).
        """
        direction = "spike" if actual_value > expected_value else "dip"
        deviation_pct = abs((actual_value - expected_value) / expected_value) * 100

        if self.use_llm:
            return self._llm_explanation(date, actual_value, expected_value, direction, deviation_pct)

        return self._fallback_explanation(direction, deviation_pct)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _llm_explanation(
        self,
        date: str,
        actual_value: float,
        expected_value: float,
        direction: str,
        deviation_pct: float,
    ) -> str:
        """Call Gemini to produce a natural-language insight."""
        prompt = (
            f"You are an expert, direct business analyst AI. On {date}, a business "
            f"operating metric experienced an unexpected {direction}. The actual data "
            f"point was {actual_value:.1f}, while the baseline expected value was "
            f"{expected_value:.1f} (a {deviation_pct:.0f}% variance). Write exactly "
            f"ONE short sentence explaining that this is a statistically rare event "
            f"and suggest one or two realistic, non-technical business causes "
            f"(e.g. localised holiday, transient outage, marketing push). "
            f"Keep it extremely simple."
        )
        try:
            response = self.model.generate_content(prompt)
            explanation_text = response.text.replace("\n", " ").strip()
            return f"[AI Explanation] {explanation_text}"
        except Exception:
            # LLM quota exceeded or network failure → deterministic fallback
            return self._fallback_explanation(direction, deviation_pct)

    @staticmethod
    def _fallback_explanation(direction: str, deviation_pct: float) -> str:
        """Deterministic template used when no LLM is available."""
        if direction == "spike":
            return (
                f"[Rule-Based Explanation] This rapid {deviation_pct:.0f}% increase "
                f"deviates from normal trends, likely driven by a temporary promotional "
                f"campaign or a localised surge in demand."
            )
        return (
            f"[Rule-Based Explanation] This sharp {deviation_pct:.0f}% drop represents "
            f"an unexpected anomaly, suggesting a potential temporary outage, data "
            f"delay, or regional holiday impacting normal volumes."
        )
