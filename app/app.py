from flask import Flask
import joblib
import os

def create_app():
    app = Flask(__name__)

    # Load model once at startup
    model_path = os.path.join("models", "sentiment_model.joblib")
    app.model = joblib.load(model_path)

    # Register routes
    from app.routes import register_routes
    register_routes(app)

    return app