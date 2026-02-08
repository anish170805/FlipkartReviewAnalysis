import streamlit as st
import joblib
import pandas as pd

from utils.preprocessing import preprocess_text


# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Flipkart Badminton Sentiment Analysis",
    layout="centered"
)


# ---------- Load Trained Pipeline ----------
@st.cache_resource
def load_model():
    return joblib.load("D:\\FK Sentimental analysis\\models\\sentiment_model.joblib")


model = load_model()


# ---------- UI ----------
st.title("🏸 Flipkart Badminton Review Sentiment Analyzer")
st.write(
    "Analyze Flipkart badminton product reviews using a machine learning model "
    "trained with TF-IDF and multiple classifiers."
)

review_text = st.text_area(
    "Enter a product review",
    height=180,
    placeholder="Example: The racket quality is excellent and very durable."
)

if st.button("Predict Sentiment"):

    if review_text.strip() == "":
        st.warning("Please enter a review before predicting.")
    else:
        # Preprocess input
        cleaned_review = preprocess_text(review_text)

        # Model expects raw text (pipeline handles TF-IDF)
        prediction = model.predict([cleaned_review])[0]

        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")