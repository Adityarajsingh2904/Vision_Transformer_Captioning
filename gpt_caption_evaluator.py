"""Utilities for evaluating captions with GPT models."""

from typing import Dict, List
import argparse
import json
import os

import openai


def _query_score(prompt: str, model: str = "gpt-4") -> float:
    """Query OpenAI ChatCompletion API and return the numeric score."""

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1,
    )
    try:
        return float(response["choices"][0]["message"]["content"].strip())
    except (KeyError, ValueError):
        return float("nan")


def evaluate_with_gpt(captions: List[str], *, api_key: str | None = None) -> Dict[str, float]:
    """Score multiple captions using GPT-4.

    Parameters
    ----------
    captions:
        List of captions to score.
    api_key:
        Optional OpenAI API key. If not provided, the ``OPENAI_API_KEY``
        environment variable will be used.
    """

    openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OpenAI API key not provided")

    scores: Dict[str, float] = {}
    for caption in captions:
        prompt = (
            "You are an expert image caption evaluator."
            " Rate the following caption for grammar fluency and relevance on a 1-10 scale."
            " Respond with only the number.\n\nCaption: "
            f"{caption}\nRating:"
        )
        scores[caption] = _query_score(prompt)

    return scores


def evaluate_caption(caption: str) -> float:
    """Convenience wrapper to score a single caption."""

    return evaluate_with_gpt([caption])[caption]


def evaluate_file(path: str) -> List[Dict[str, float]]:
    """Evaluate captions stored line by line in ``path``."""

    with open(path, "r", encoding="utf-8") as fh:
        captions = [line.strip() for line in fh if line.strip()]

    scores = evaluate_with_gpt(captions)
    return [{"caption": c, "score": scores[c]} for c in captions]


def main() -> None:
    """CLI entry point for scoring captions in a text file."""

    parser = argparse.ArgumentParser(description="Evaluate caption fluency using GPT-4")
    parser.add_argument("captions", help="Path to text file with one caption per line")
    parser.add_argument("--output", default="gpt_scores.json", help="Path to output JSON file")
    parser.add_argument("--api_key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    args = parser.parse_args()

    results = evaluate_file(args.captions)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved scores to {args.output}")


if __name__ == "__main__":
    main()
