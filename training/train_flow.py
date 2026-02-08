from training.data_loader import load_data
from training.cleaning import df_cleaning
from training.model_selection import train_and_select_model
from app.utils.preprocessing import preprocess_text

import joblib
from prefect import flow
import mlflow
import mlflow.sklearn


# Set MLflow experiment
mlflow.set_experiment("Flipkart-Badminton-Sentiment")


@flow(name="Sentiment Training Flow")
def training_flow(csv_path):

    with mlflow.start_run(
        run_name="LogReg_TFIDF_Badminton"
    ):

        # -------- Load and preprocess data --------
        df = load_data(csv_path)
        df = df_cleaning(df)
        df["Review text"] = df["Review text"].apply(preprocess_text)

        # -------- Log dataset parameters --------
        mlflow.log_param("dataset", "Flipkart Badminton Reviews")
        mlflow.log_param("num_samples", len(df))
        mlflow.log_param("task", "Sentiment Analysis")

        # -------- Train model --------
        model, metrics = train_and_select_model(df)

        # -------- Log metrics --------
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])

        # -------- Save model locally --------
        joblib.dump(model, "models/sentiment_model.joblib")

        # -------- Register model in MLflow --------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="FlipkartSentimentModel"
        )

        # -------- Tags for model management --------
        mlflow.set_tag("domain", "E-commerce")
        mlflow.set_tag("product", "Badminton")
        mlflow.set_tag("project", "Flipkart Sentiment Analysis")

        print("Model trained, logged, and registered successfully.")


if __name__ == "__main__":
    training_flow("data/data.csv")