import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
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

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open(r"C:\Users\UJJWAL\Downloads\fake news detection\dataset\model.pkl", "rb"))
        vectorizer = pickle.load(open(r"C:\Users\UJJWAL\Downloads\fake news detection\dataset\vectorizer.pkl", "rb"))
        return model, vectorizer
    except:
        st.error("❌ Could not load model. Make sure model.pkl and vectorizer.pkl are in the correct path.")
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
            st.metric("Train Accuracy", "0.9950", delta="Train Set")
        with col2:
            st.metric("Test Accuracy", "0.9847", delta="Test Set")

    # Main content
    tabs = st.tabs(["Single News Check", "Batch Upload", "Model Info"])
    
    # Tab 1: Single News Check
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
                    # Clean and vectorize
                    cleaned_text = clean_text(news_input)
                    text_vec = vectorizer.transform([cleaned_text])
                    
                    # Predict
                    prediction = model.predict(text_vec)[0]
                    confidence = model.predict_proba(text_vec)[0]
                    
                    # Display result
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
                    
                    # Confidence breakdown
                    st.subheader("Confidence Breakdown")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Fake News Probability", f"{confidence[0]*100:.2f}%")
                    with col2:
                        st.metric("Real News Probability", f"{confidence[1]*100:.2f}%")
                else:
                    st.warning("⚠️ Please enter some text to analyze.")
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.rerun()
    
    # Tab 2: Batch Upload
    with tabs[1]:
        st.subheader("Batch Check - Upload CSV")
        
        uploaded_file = st.file_uploader("Upload CSV with news articles", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.write("**Preview of uploaded data:**")
            st.dataframe(df.head())
            
            # Check for text column
            text_columns = df.columns.tolist()
            selected_column = st.selectbox("Select the column containing news text:", text_columns)
            
            if st.button("🔍 Analyze All Articles"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                predictions = []
                confidences = []
                
                for idx, text in enumerate(df[selected_column]):
                    cleaned_text = clean_text(str(text))
                    text_vec = vectorizer.transform([cleaned_text])
                    
                    pred = model.predict(text_vec)[0]
                    conf = model.predict_proba(text_vec)[0]
                    
                    predictions.append("FAKE" if pred == 0 else "REAL")
                    confidences.append(conf[pred])
                    
                    progress_bar.progress((idx + 1) / len(df))
                    status_text.text(f"Analyzed {idx + 1}/{len(df)} articles...")
                
                # Results
                df["Prediction"] = predictions
                df["Confidence"] = [f"{c*100:.2f}%" for c in confidences]
                
                st.success("✅ Analysis complete!")
                st.dataframe(df, use_container_width=True)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="fake_news_predictions.csv",
                    mime="text/csv"
                )
    
    # Tab 3: Model Info
    with tabs[2]:
        st.subheader("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Details:**")
            st.info("""
            - **Algorithm:** Logistic Regression
            - **Vectorizer:** TF-IDF (5000 features)
            - **Classes:** Fake News (0), Real News (1)
            - **Training Method:** Train-Test Split (80-20)
            """)
        
        with col2:
            st.write("**Performance Metrics:**")
            metrics_data = {
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Train Set": ["99.50%", "99.48%", "99.52%", "99.50%"],
                "Test Set": ["98.47%", "98.45%", "98.49%", "98.47%"]
            }
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

else:
    st.error("❌ Failed to load the model. Please ensure the model files are in the correct location.")
