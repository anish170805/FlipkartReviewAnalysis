import joblib
from app.utils.preprocessing import preprocess_text

MODEL_PATH = "models/sentiment_model.joblib"

model = joblib.load(MODEL_PATH)

def predict_sentiment(text):
    if not isinstance(text, str) or len(text.strip()) < 3:
        return {"error": "Invalid input"}

    processed = preprocess_text(text)
    pred = model.predict([processed])[0]

    return {
        "prediction": "positive" if pred == 1 else "negative"
    }