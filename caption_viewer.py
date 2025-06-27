import tempfile
from pathlib import Path

from inference_caption import generate_caption

import streamlit as st
from PIL import Image
import openai


def score_caption(caption: str, api_key: str, model: str = "gpt-4") -> float:
    """Use GPT model to score caption fluency."""
    openai.api_key = api_key
    prompt = (
        "Rate the fluency of the following image caption on a scale of 1-10. "
        "Only return the number.\nCaption: \"%s\"" % caption
    )
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    except openai.error.OpenAIError as e:
        raise RuntimeError(f"OpenAI API request failed: {e}") from e

    return float(response.choices[0].message["content"].strip())


st.title("Image Caption Viewer")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"]) 
if file:
    tmp_dir = tempfile.mkdtemp()
    image_path = Path(tmp_dir) / file.name
    with open(image_path, "wb") as f:
        f.write(file.getbuffer())
    image = Image.open(image_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    caption = generate_caption(str(image_path))
    st.write("**Caption:**", caption)

    if api_key:
        try:
            score = score_caption(caption, api_key)
            st.write(f"**Fluency Score:** {score}")
        except Exception as e:
            st.write(f"Scoring failed: {e}")
    else:
        st.write("Provide an API key to score the caption")
