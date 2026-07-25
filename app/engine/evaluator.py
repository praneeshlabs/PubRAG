"""
app/engine/evaluator.py

LLM-based evaluation of RAG pipeline outputs.

Metrics implemented (inspired by RAGAS methodology)

1. Faithfulness         — Are all answer claims grounded in the retrieved context?
2. Answer Relevance     — Does the answer actually address the original question?
3. Context Precision    — Are the retrieved documents relevant to the question?

These three metrics are weighted into an Overall Score.

Design notes
- Each metric calls Claude with a strict JSON-only output prompt so results
  are machine-parseable.
- temperature=0.0 is used for all evaluation calls to maximise reproducibility.
- All LLM calls are wrapped in try/except; failures return a neutral 0.5 score
  rather than crashing the UI.
- The evaluator is optional in the UI (checkbox) to avoid extra API cost.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import anthropic # type: ignore

from app.core.config import Config # type: ignore
from app.templates.prompts import ( # type: ignore 
    EVAL_CONTEXT_PRECISION_PROMPT,
    EVAL_FAITHFULNESS_PROMPT,
    EVAL_RELEVANCE_PROMPT,
)

logger = logging.getLogger(__name__)

# Result dataclass

DEFAULT_WEIGHTS: dict[str, float] = {
    "faithfulness": 0.40,
    "answer_relevance": 0.40,
    "context_precision": 0.20,
}


@dataclass
class EvaluationResult:
    """
    Typed container for all RAG evaluation scores and supporting reasoning.

    Scores are in [0.0, 1.0] where 1.0 is best.
    """

    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    overall_score: float = 0.0

    faithfulness_reasoning: str = ""
    relevance_reasoning: str = ""
    precision_reasoning: str = ""

    # Claims identified as not supported by any retrieved document
    unsupported_claims: list[str] = field(default_factory=list)

    # Raw LLM response strings (useful for debugging)
    _raw_responses: dict[str, str] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON export."""
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "context_precision": round(self.context_precision, 4),
            "overall_score": round(self.overall_score, 4),
            "faithfulness_reasoning": self.faithfulness_reasoning,
            "relevance_reasoning": self.relevance_reasoning,
            "precision_reasoning": self.precision_reasoning,
            "unsupported_claims": self.unsupported_claims,
        }

    @property
    def grade(self) -> str:
        """Return a letter grade for the overall score."""
        score = self.overall_score
        if score >= 0.85:
            return "A"
        if score >= 0.70:
            return "B"
        if score >= 0.55:
            return "C"
        if score >= 0.40:
            return "D"
        return "F"


# Evaluator

class RAGEvaluator:
    """
    LLM-based RAG evaluator.

    Parameters
    config : Config
        Validated application configuration.
    """

    # Max characters of context shown to the eval LLM to keep prompts bounded
    _MAX_CONTEXT_CHARS: int = 3_000
    _MAX_CONTEXT_DOCS: int = 5

    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        logger.info("RAGEvaluator initialised (model: %s)", config.LLM_MODEL)

    # Internal helpers

    def _call_eval_llm(self, user_prompt: str) -> str:
        """
        Call Claude with temperature=0 for a deterministic evaluation.

        Returns
        -------
        str
            Raw text response from the model.

        Raises
        ------
        anthropic.APIError on unrecoverable API failures.
        """
        try:
            response = self._client.messages.create(
                model=self.config.LLM_MODEL,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()
        except anthropic.APIStatusError as exc:
            logger.error(
                "Anthropic API error (status %s) during evaluation: %s",
                exc.status_code,
                exc.message,
            )
            raise
        except anthropic.APIConnectionError as exc:
            logger.error("Network error during evaluation: %s", exc)
            raise

    @staticmethod
    def _parse_json_response(raw: str) -> dict:
        """
        Parse a JSON response from Claude, stripping markdown code fences if present.

        Returns an empty dict on failure (allows safe .get() by callers).
        """
        try:
            # Strip ```json ... ``` or ``` ... ``` wrappers
            clean = raw.strip()
            if clean.startswith("```"):
                lines = clean.splitlines()
                clean = "\n".join(lines[1:-1]) if len(lines) > 2 else clean
            return json.loads(clean)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse failed for evaluation response: %s", exc)
            return {}

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp a score to [0.0, 1.0]."""
        return max(0.0, min(1.0, float(value)))

    def _truncate_contexts(self, contexts: list[str]) -> str:
        """
        Join and truncate context strings to keep the evaluation prompt bounded.
        """
        joined = "\n\n---\n\n".join(contexts[: self._MAX_CONTEXT_DOCS])
        if len(joined) > self._MAX_CONTEXT_CHARS:
            joined = joined[: self._MAX_CONTEXT_CHARS] + "\n[… truncated for evaluation]"
        return joined

    # Individual metric methods

    def evaluate_faithfulness(
        self,
        answer: str,
        contexts: list[str],
    ) -> tuple[float, str, list[str]]:
        """
        Assess whether every factual claim in the answer is supported by context.

        Returns
        -------
        tuple[float, str, list[str]]
            (score, reasoning, list_of_unsupported_claims)
        """
        context_str = self._truncate_contexts(contexts)
        prompt = EVAL_FAITHFULNESS_PROMPT.format(
            context=context_str,
            answer=answer,
        )

        try:
            raw = self._call_eval_llm(prompt)
            data = self._parse_json_response(raw)
            score = self._clamp(data.get("score", 0.5))
            reasoning = str(data.get("reasoning", "Parsing failed."))
            unsupported = [str(c) for c in data.get("unsupported_claims", [])]
            logger.info("Faithfulness score: %.3f", score)
            return score, reasoning, unsupported

        except Exception as exc:  # noqa: BLE001
            logger.warning("Faithfulness evaluation failed: %s", exc)
            return 0.5, f"Evaluation unavailable ({exc})", []

    def evaluate_answer_relevance(
        self,
        question: str,
        answer: str,
    ) -> tuple[float, str]:
        """
        Assess whether the answer actually addresses the research question.

        Returns
        -------
        tuple[float, str]
            (score, reasoning)
        """
        prompt = EVAL_RELEVANCE_PROMPT.format(
            question=question,
            answer=answer,
        )

        try:
            raw = self._call_eval_llm(prompt)
            data = self._parse_json_response(raw)
            score = self._clamp(data.get("score", 0.5))
            reasoning = str(data.get("reasoning", "Parsing failed."))
            logger.info("Answer relevance score: %.3f", score)
            return score, reasoning

        except Exception as exc:  # noqa: BLE001
            logger.warning("Answer relevance evaluation failed: %s", exc)
            return 0.5, f"Evaluation unavailable ({exc})"

    def evaluate_context_precision(
        self,
        question: str,
        contexts: list[str],
    ) -> tuple[float, str]:
        """
        Assess whether the retrieved contexts are topically relevant to the question.

        Returns
        -------
        tuple[float, str]
            (score, reasoning)
        """
        if not contexts:
            return 0.0, "No contexts were provided."

        # Show only a short excerpt of each context to keep the prompt focused
        excerpts = "\n---\n".join(ctx[:400] for ctx in contexts[: self._MAX_CONTEXT_DOCS])
        prompt = EVAL_CONTEXT_PRECISION_PROMPT.format(
            question=question,
            context_excerpts=excerpts,
        )

        try:
            raw = self._call_eval_llm(prompt)
            data = self._parse_json_response(raw)
            score = self._clamp(data.get("score", 0.5))
            reasoning = str(data.get("reasoning", "Parsing failed."))
            logger.info("Context precision score: %.3f", score)
            return score, reasoning

        except Exception as exc:  # noqa: BLE001
            logger.warning("Context precision evaluation failed: %s", exc)
            return 0.5, f"Evaluation unavailable ({exc})"

    # Full evaluation suite

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        weights: Optional[dict[str, float]] = None,
    ) -> EvaluationResult:
        """
        Run all three evaluation metrics and compute a weighted overall score.

        Parameters
        ----------
        question  : The original research question.
        answer    : The RAG system's generated answer.
        contexts  : Plain-text contents of the retrieved (and reranked) documents.
        weights   : Optional custom weights.  Defaults to DEFAULT_WEIGHTS.

        Returns
        -------
        EvaluationResult
            All metric scores, reasoning strings, and the weighted overall score.
        """
        _weights = weights if weights is not None else DEFAULT_WEIGHTS

        # Normalise weights in case caller provides non-unit-sum values
        total_weight = sum(_weights.values())
        normalised = {k: v / total_weight for k, v in _weights.items()}

        result = EvaluationResult()

        # Faithfulness
        logger.info("Evaluating faithfulness…")
        (
            result.faithfulness,
            result.faithfulness_reasoning,
            result.unsupported_claims,
        ) = self.evaluate_faithfulness(answer, contexts)

        # Answer relevance 
        logger.info("Evaluating answer relevance…")
        result.answer_relevance, result.relevance_reasoning = (
            self.evaluate_answer_relevance(question, answer)
        )

        # Context precision 
        logger.info("Evaluating context precision…")
        result.context_precision, result.precision_reasoning = (
            self.evaluate_context_precision(question, contexts)
        )

        # Overall score
        result.overall_score = (
            normalised["faithfulness"] * result.faithfulness
            + normalised["answer_relevance"] * result.answer_relevance
            + normalised["context_precision"] * result.context_precision
        )

        logger.info(
            "Evaluation complete | faithfulness=%.3f | relevance=%.3f | "
            "precision=%.3f | overall=%.3f | grade=%s",
            result.faithfulness,
            result.answer_relevance,
            result.context_precision,
            result.overall_score,
            result.grade,
        )

        return result
