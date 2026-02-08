import streamlit as st
import joblib
from utils.preprocessing import preprocess_text
import numpy as np

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("Flipkart Review Sentiment Analysis")

model = joblib.load("D:\\FK Sentimental analysis\\models\\sentiment_model.joblib")

user_input = st.text_area("Enter a product review:")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned_text = preprocess_text(user_input)

        prediction = model.predict([cleaned_text])[0]

        # Confidence handling
        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba([cleaned_text]))
        else:
            confidence = None

        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")

        if confidence is not None:
            st.info(f"Confidence: {confidence:.2f}")

        with st.expander("See cleaned text"):
            st.write(cleaned_text)