from typing import List, Dict
import os
import openai


def evaluate_with_gpt(captions: List[str]) -> Dict[str, float]:
    """Score captions using GPT-4 for fluency and relevance.

    Each caption is rated on a 1-10 scale. The function expects an
    ``OPENAI_API_KEY`` environment variable to be set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    openai.api_key = api_key

    scores: Dict[str, float] = {}
    for caption in captions:
        prompt = (
            "You are an expert image caption evaluator."
            " Rate the following caption for grammar fluency and relevance on a 1-10 scale."
            " Respond with only the number.\n\nCaption: "
            f"{caption}\nRating:"
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1,
            )
            score_text = response["choices"][0]["message"]["content"].strip()
            scores[caption] = float(score_text)
        except Exception:
            scores[caption] = 0.0
    return scores
