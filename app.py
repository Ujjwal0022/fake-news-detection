import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import os

# Page configuration
st.set_page_config(page_title="Fake News Detector", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .fake-news {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff0000;
    }
    .real-news {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #00cc00;
    }
</style>
""", unsafe_allow_html=True)

# ✅ FIXED MODEL LOADING
@st.cache_resource
def load_model():
    try:
        base_dir = os.path.dirname(__file__)

        model_path = os.path.join(base_dir, "dataset", "model.pkl")
        vectorizer_path = os.path.join(base_dir, "dataset", "vectorizer.pkl")

        model = pickle.load(open(model_path, "rb"))
        vectorizer = pickle.load(open(vectorizer_path, "rb"))

        return model, vectorizer

    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        return None, None


# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# UI
st.title("🔍 Fake News Detection System")
st.write("This app uses Machine Learning to detect fake news.")

model, vectorizer = load_model()

if model is not None and vectorizer is not None:

    news_input = st.text_area("Enter news text:")

    if st.button("Analyze"):
        if news_input.strip():
            cleaned = clean_text(news_input)
            vec = vectorizer.transform([cleaned])

            pred = model.predict(vec)[0]

            if pred == 0:
                st.markdown('<div class="fake-news">🚨 FAKE NEWS</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="real-news">✅ REAL NEWS</div>', unsafe_allow_html=True)
        else:
            st.warning("Enter some text")

else:
    st.error("❌ Model not loaded")