import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Page configuration
st.set_page_config(page_title="Fake News Detector", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
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

# ✅ FIXED PATH (IMPORTANT)
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
        st.error(f"❌ Could not load model: {e}")
        return None, None


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# Main app
st.title("🔍 Fake News Detection System")
st.write("This app uses Machine Learning to detect fake news with high accuracy.")

# Load model
model, vectorizer = load_model()

if model is not None and vectorizer is not None:

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        **Model:** Logistic Regression with TF-IDF Vectorization
        
        **Features:** 5000 most important words
        
        **Training Data:** Combined fake and real news dataset
        """)
        
        st.divider()
        st.header("📊 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Train Accuracy", "0.9950")
        with col2:
            st.metric("Test Accuracy", "0.9847")

    # Tabs
    tabs = st.tabs(["Single News Check", "Batch Upload", "Model Info"])
    
    # 🔹 Tab 1
    with tabs[0]:
        st.subheader("Check if a News Article is Fake")
        
        news_input = st.text_area(
            "Enter news article or headline:",
            placeholder="Paste the news text here...",
            height=200
        )
        
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔍 Analyze", use_container_width=True):
                if news_input.strip():
                    cleaned_text = clean_text(news_input)
                    text_vec = vectorizer.transform([cleaned_text])
                    
                    prediction = model.predict(text_vec)[0]
                    confidence = model.predict_proba(text_vec)[0]
                    
                    st.divider()

                    if prediction == 0:
                        st.markdown(f"""
                        <div class="fake-news">
                            <h3>🚨 FAKE NEWS DETECTED</h3>
                            <p><strong>Confidence: {confidence[0]*100:.2f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="real-news">
                            <h3>✅ LIKELY REAL NEWS</h3>
                            <p><strong>Confidence: {confidence[1]*100:.2f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.subheader("Confidence Breakdown")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Fake Probability", f"{confidence[0]*100:.2f}%")
                    with col2:
                        st.metric("Real Probability", f"{confidence[1]*100:.2f}%")

                else:
                    st.warning("⚠️ Please enter some text")

        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.rerun()

    # 🔹 Tab 2
    with tabs[1]:
        st.subheader("Batch Check - Upload CSV")
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

            column = st.selectbox("Select text column:", df.columns)

            if st.button("Analyze All"):
                results = []
                probs = []

                for text in df[column]:
                    cleaned = clean_text(str(text))
                    vec = vectorizer.transform([cleaned])
                    pred = model.predict(vec)[0]
                    conf = model.predict_proba(vec)[0]

                    results.append("FAKE" if pred == 0 else "REAL")
                    probs.append(conf[pred])

                df["Prediction"] = results
                df["Confidence"] = [f"{p*100:.2f}%" for p in probs]

                st.success("Done ✅")
                st.dataframe(df)

    # 🔹 Tab 3
    with tabs[2]:
        st.subheader("Model Info")
        st.write("""
        - Algorithm: Logistic Regression  
        - Vectorizer: TF-IDF  
        - Classes: Fake (0), Real (1)  
        """)

else:
    st.error("❌ Model not loaded")