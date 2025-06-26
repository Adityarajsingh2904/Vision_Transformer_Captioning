import tempfile
from typing import List

import streamlit as st
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from gpt_caption_evaluator import evaluate_with_gpt


# Placeholder for your trained model's caption generation
# Replace this with actual inference code.
def generate_captions(image_path: str) -> List[str]:
    """Generate caption(s) for the given image path.

    This stub should be replaced with model inference using the trained
    captioning model contained in this repository.
    """
    # TODO: integrate with inference_caption.py or your model loader
    return ["Caption generation not implemented"]


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU score for a single hypothesis-reference pair."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=SmoothingFunction().method1)


def main() -> None:
    st.title("Caption Viewer with GPT Evaluation")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    reference_caption = st.text_input("Reference caption (optional)")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            image.save(tmp.name)
            captions = generate_captions(tmp.name)

        scores = evaluate_with_gpt(captions)
        for cap in captions:
            line = f"**{cap}** - GPT Score: {scores.get(cap, 'N/A'):.1f}"
            if reference_caption:
                bleu = compute_bleu(reference_caption, cap)
                line += f" | BLEU: {bleu:.3f}"
            st.write(line)


if __name__ == "__main__":
    main()
