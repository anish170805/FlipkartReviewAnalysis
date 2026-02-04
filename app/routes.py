from flask import request, jsonify

def register_routes(app):

    @app.route("/", methods=["GET"])
    def home():
        return {
            "status": "running",
            "endpoint": "/predict",
            "method": "POST"
        }

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(silent=True)

        if not data or "text" not in data:
            return jsonify({"error": "JSON body with 'text' required"}), 400

        text = data["text"]

        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Invalid input text"}), 400

        # Predict
        pred = app.model.predict([text])[0]

        return jsonify({
            "input": text,
            "prediction": "positive" if pred == 1 else "negative"
        })