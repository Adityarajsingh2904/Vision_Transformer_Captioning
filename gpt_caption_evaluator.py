import argparse
import json
import os
from typing import List, Dict
import openai


def evaluate_caption(caption: str, model: str = "gpt-4") -> float:
    """Query GPT model to rate caption fluency from 1-10."""
    prompt = (
        "Rate the fluency of the following image caption on a scale of 1-10. "
        "Only return the number.\nCaption: \"%s\"" % caption
    )
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        score = float(response.choices[0].message["content"].strip())
    except ValueError:
        score = float("nan")
    return score


def evaluate_file(path: str) -> List[Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        captions = [line.strip() for line in f if line.strip()]
    results = []
    for caption in captions:
        score = evaluate_caption(caption)
        results.append({"caption": caption, "score": score})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate caption fluency using GPT-4")
    parser.add_argument("captions", help="Path to text file containing captions, one per line")
    parser.add_argument("--output", default="gpt_scores.json", help="Output JSON file")
    parser.add_argument("--api_key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    args = parser.parse_args()

    openai.api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OpenAI API key not provided")

    results = evaluate_file(args.captions)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved scores to {args.output}")


if __name__ == "__main__":
    main()
