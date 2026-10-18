import streamlit as st
import numpy as np
from pathlib import Path
import keras

st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="centered")
MODEL_PATH = Path(__file__).resolve().parent / "sentiment_model.keras"

@st.cache_resource
def load_model(model_path: str):
    return keras.saving.load_model(model_path)

st.title("🎬 Movie Review Sentiment")
st.write("Enter a movie review to estimate whether its sentiment is positive or negative.")
st.caption("This educational model can make mistakes, especially with sarcasm, mixed opinions, and complex negation.")

if not MODEL_PATH.exists():
    st.error("Trained model not found: sentiment_model.keras")
    st.info("Run the notebook from top to bottom to train and save the model, then place the saved file beside app.py.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("The model could not be loaded. Re-run the notebook with the installed requirements and regenerate sentiment_model.keras.")
    st.exception(exc)
    st.stop()

review = st.text_area("Movie review", placeholder="Example: The acting was excellent, but the story felt slow.", height=140, max_chars=5000)
if st.button("Analyze sentiment", type="primary", use_container_width=True):
    cleaned = review.strip()
    if not cleaned:
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing review..."):
            score = float(model.predict(np.asarray([cleaned], dtype=object), verbose=0).reshape(-1)[0])
        label = "Positive" if score >= 0.5 else "Negative"
        st.subheader(f"Predicted sentiment: {label}")
        st.metric("Positive sentiment score", f"{score:.1%}")
        st.progress(min(max(score, 0.0), 1.0))
        st.caption("The score is the model's sigmoid output, not a calibrated probability that the prediction is correct.")
